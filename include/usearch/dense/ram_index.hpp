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
#include <type_traits> // `std::is_same_v` - the tiles kit a scan streams through

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref`
#include <forkunion/types.hpp>   // `fu::wait_capped_t`
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
 *  Construction and traversal run simultaneously. Every node's neighbors hold that node's own
 *  lock in their leading slot, placing locks at the start of independent cache lines; the rest
 *  of the book-keeping is a handful of atomics.
 *
 *  A batch publishes whole: one compare-exchange dispenses its rows' IDs as one range, and one
 *  watermark turn makes every row visible, so a reader sees all of a batch or none of it. The
 *  turns run in dispensed order, so the publication gap is bounded by the batches in flight, and
 *  a small batch waits behind an earlier, larger batch's wiring before its turn.
 *
 *  Every node keeps its key, and every key its nodes. A row's key is stored beside its vector, and
 *  the turn holder files the batch's keys right before moving the watermark, so the key map has one
 *  writer and no read-modify-write per key. A lookup drops every node at or past the watermark it
 *  acquired, so a key is found exactly when its node is visible.
 *
 *  The sharded index closes the dispenser around its refines and loads: closing succeeds only
 *  once every dispensed ID is published, and a dispense meeting a closed dispenser answers
 *  @c contended_k, so no batch wires beside a load copying at the watermark or a refine
 *  rewriting rows.
 *
 *  @section dense_ram_index_persistence Persistence
 *
 *  Not meant to be reached from disk - io_uring and SPDK reimplementations cover that.
 *
 *  @section dense_ram_index_malloc Memory Allocation Strategy
 *
 *  Chunks grow through power-of-two sizes up to @b 2^23-row slabs, amortizing allocations between
 *  insertions instead of re-padding one huge matrix. Slabs that wide reach a gigabyte at 128-byte
 *  rows, so large indexes fill whole 1 GiB pages. Compatible with CXL-backed allocators.
 */
template <                                                                    //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,                       //
    typename tiles_kit_type_ = no_tiles_kit_t,                                //
    typename elementwise_kit_type_ = serial_elementwise_kit_t,                //
    typename rmscast_kit_type_ = no_rmscast_kit_t,                            //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,                 //
    typename wait_type_ = dummy_waiter_t,                                     //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = dummy_cache_hints_t,                         //
    typename vectors_allocator_type_ = std::allocator<byte_t>,                //
    typename neighbors_allocator_type_ = std::allocator<byte_t>,              //
    typename traversals_allocator_type_ = std::allocator<byte_t>,             //
    typename key_type_ = uuid128_t,                                           //
    typename key_hash_type_ = std::hash<key_type_>                            //
    >
struct ram_index;

/**
 *  @brief A filtered walk's fold: each scored quad's contenders, the nodes that could still enter
 *      @c admitted, are asked of @c predicate at their own static extent, and what it admits enters
 *      @c admitted, at most @c breadth, while the walk routes through all.
 *
 *  The predicate takes node IDs, usually a @c node_predicate over a key column. An aggregating one
 *  only gathers the scored nodes, answered at @ref settle.
 */
template <typename predicate_type_, typename ranking_type_>
struct predicate_fold {

    predicate_type_& predicate;
    ranking_type_& admitted;
    std::size_t breadth;

    template <typename dot_result_type_>
    walk_demand_t absorb(std::span<scored_node_t const> scored_nodes, std::span<dot_result_type_ const>) noexcept {
        quad<node_id_t> ids;
        quad<scored_node_t> contenders;
        std::size_t count = 0;
        for (scored_node_t const candidate : scored_nodes) {
            if (!contends_(candidate)) continue;
            ids[count] = candidate.node_id;
            contenders[count] = candidate;
            ++count;
        }
        ask_contenders_(ids, contenders, count);
        return walk_demand_t::continue_k;
    }

    /** Answers what an aggregating predicate gathered since the last settle; an immediate one has
     *  nothing left. */
    void settle() noexcept {
        settle_predicate(predicate, [this](scored_node_t const& match) noexcept {
            [[maybe_unused]] bool const inserted = admitted.insert(scored_node_t {match}, breadth);
        });
    }

  private:
    /** Whether @p candidate could still enter @c admitted: always until it fills, then unless its
     *  worst match is closer, so the predicate never sees a lane the candidates would drop; a tie
     *  still merges in. */
    bool contends_(scored_node_t const& candidate) const noexcept {
        return admitted.size() < breadth || !closer_scored_node_t {}(admitted.top(), candidate);
    }

    /** Asks about the @p count leading contenders and merges what an immediate predicate admits as
     *  one ascending quad, as the walk merges its candidates, so ties rank alike in both. */
    void ask_contenders_(quad<node_id_t> const& ids, quad<scored_node_t> const& contenders,
                         std::size_t count) noexcept {
        quad<scored_node_t> matches;
        std::size_t matched = 0;
        auto const gather_match = [&](scored_node_t const& match) noexcept { matches[matched++] = match; };
        with_static_extent<quad_lanes_k>(count, [&]<std::size_t extent_>() noexcept {
            ask_quad(predicate, std::span<node_id_t const, extent_>(ids.data(), extent_),
                     std::span<scored_node_t const, extent_>(contenders.data(), extent_), gather_match);
        });
        std::span<scored_node_t> const admitted_quad = std::span(matches).first(matched);
        network_sort(admitted_quad, closer_scored_node_t {});
        admitted.insert(admitted_quad, breadth);
    }
};

/** Reads every scored quad - live lanes in kernel order beside their native dots, both valid only
 *  for the call - and answers whether the walk should go on. */
template <typename fold_type_, typename dot_result_type_>
concept a_fold = requires(fold_type_& fold, std::span<scored_node_t const> scored_nodes,
                          std::span<dot_result_type_ const> dots) {
    { fold.absorb(scored_nodes, dots) } noexcept -> std::same_as<walk_demand_t>;
};

/** Graph products of one left-hand row as a fold: each scored pair whose resolved ID the predicate
 *  admits appends that ID and its dot as @p product_type_ until the row is full, while @c scored
 *  keeps counting past it. */
template <typename id_type_, typename product_type_, typename resolve_type_ = std::identity,
          typename predicate_type_ = dummy_predicate_t>
struct multiply_fold {

    /** The row's ID output. */
    std::span<id_type_> ids;

    /** The row's product output, positionally paired with @c ids. */
    std::span<product_type_> products;

    /** Maps a walked node ID to the ID the row reports. */
    resolve_type_ resolve = {};

    /** Asked about a quad of resolved IDs; a rejected pair is neither written nor counted. */
    predicate_type_ predicate = {};

    /** Pairs written so far. */
    std::size_t written = 0;

    /** Pairs admitted, written or not - above the row's width when it was truncated. */
    std::size_t scored = 0;

    template <typename dot_result_type_>
    walk_demand_t absorb(std::span<scored_node_t const> scored_nodes, std::span<dot_result_type_ const> dots) noexcept {
        with_static_extent<quad_lanes_k>(scored_nodes.size(), [&]<std::size_t extent_>() noexcept {
            std::array<id_type_, extent_> lane_ids;
            std::array<admission_t, extent_> admissions;
            for (std::size_t lane = 0; lane != extent_; ++lane) lane_ids[lane] = resolve(scored_nodes[lane].node_id);
            predicate(std::span<id_type_ const, extent_>(lane_ids), std::span<admission_t, extent_>(admissions));
            for (std::size_t lane = 0; lane != extent_; ++lane) {
                if (admissions[lane] != admission_t::admit_k) continue;
                if (written != ids.size())
                    ids[written] = lane_ids[lane], products[written++] = static_cast<product_type_>(dots[lane]);
                ++scored;
            }
        });
        return walk_demand_t::continue_k;
    }

    template <typename dot_result_type_>
    walk_demand_t absorb(std::span<scored_node_t const> scored_nodes, std::span<dot_result_type_ const> dots) noexcept
        requires std::same_as<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>
    {
        std::size_t const writable = min_of<std::size_t>(scored_nodes.size(), ids.size() - written);
        for (std::size_t lane = 0; lane != writable; ++lane) {
            ids[written + lane] = resolve(scored_nodes[lane].node_id);
            products[written + lane] = static_cast<product_type_>(dots[lane]);
        }
        written += writable;
        scored += scored_nodes.size();
        return walk_demand_t::continue_k;
    }
};

/** What an exact scan hands an aggregating predicate beside each node it gathers: the row within
 *  the rows one call scores. */
using scan_payload_t = u32_t;

/** Asks @p admits about the @p filled leading @p nodes of a quad at their own static extent,
 *  handing @p keep_entry the payload of each one it admits. */
template <typename admits_type_, typename keep_row_type_>
void ask_contenders(admits_type_& admits, quad<node_id_t> const& nodes, quad<scan_payload_t> const& payloads,
                    std::size_t filled, keep_row_type_& keep_entry) noexcept {
    with_static_extent<quad_lanes_k>(filled, [&]<std::size_t extent_>() noexcept {
        ask_quad(admits, std::span<node_id_t const, extent_>(nodes.data(), extent_),
                 std::span<scan_payload_t const, extent_>(payloads.data(), extent_), keep_entry);
    });
}

/** The scorer of an exact scan's blocks where no matrix engine multiplies the scalar: a block of
 *  queries stays resident in cache while each stored row of a task scores against four of them a
 *  call, through @c score_step_loop under the kit's target. */
template <a_quads_kit quads_kit_type_>
struct quads_scorer {
    using scalar_t = typename quads_kit_type_::scalar_t;
    using norms_t = vector_norms<typename scalar_t::dot_result_t>;
    using raw_dot_t = typename scalar_t::dot_result_t::raw_t;

    /** A quad of rows a task: the most an aggregating predicate gathers between settles. */
    static constexpr std::size_t step_rows_k = quad_lanes_k;

    /** A quad of queries a call, the multiple a scan's groups round to. */
    static constexpr std::size_t step_columns_k = quad_lanes_k;

    /** Dots a task stages per row: none, since the kit finalizes in registers. */
    static constexpr std::size_t dots_columns_k = 0;

    nk_kernel_kind_t kernel;
    nk::matrix_view<scalar_t> queries;
    std::span<norms_t const> query_norms;

    /** Unused: the queries score where they lie. */
    byte_t const* packed;

    /** Unused: the kit finalizes its dots in registers. */
    nk::matrix_span<raw_dot_t> dots;

    /** Queries a block of @p columns holds: half of @p cache_bytes in rows of @p row_bytes, the row
     *  block taking the other half, and every one when no cache is named. */
    static std::size_t block_columns(std::size_t columns, std::size_t cache_bytes, std::size_t row_bytes) noexcept {
        if (!cache_bytes) return columns;
        return min_of(max_of(cache_bytes / (2 * row_bytes), step_columns_k), columns);
    }

    /** Bytes the packs of a block take: none. */
    static std::size_t packs_bytes(std::size_t, std::size_t) noexcept { return 0; }

    /** Packs nothing, as the queries score where they lie. */
    static void pack(nk::matrix_view<scalar_t>, byte_t*) noexcept {}

    /** Scores @p task against the block into @p step_distances, @b [rows,queries]. */
    void score(nk::matrix_view<scalar_t> task, std::span<norms_t const> row_norms,
               nk::matrix_span<f32_t> step_distances) const noexcept {
        invoke_under_target_<score_step_loop, quads_kit_type_>(kernel, task, row_norms, queries, query_norms,
                                                               step_distances);
    }
};

/** The scorer of an exact scan's blocks where a matrix engine multiplies the scalar: a block of
 *  queries is packed once, stored rows stream against the pack a task at a time, and every quad
 *  of a row's dots finalizes through @p quads_kit_type_, so the distances match a walk's. */
template <typename tiles_kit_type_, a_quads_kit quads_kit_type_>
struct tiles_scorer {
    using scalar_t = typename quads_kit_type_::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using norms_t = vector_norms<dot_result_t>;
    using raw_dot_t = typename dot_result_t::raw_t;
    using finalized_vec_t = typename quads_kit_type_::finalized_vec_t;
    static_assert(sizeof(finalized_vec_t) == quad_lanes_k * sizeof(raw_dot_t), "A finalizer takes four dot lanes");

    /** Rows one engine call streams against a pack: the most an aggregating predicate gathers
     *  between settles. */
    static constexpr std::size_t step_rows_k = tiles_kit_type_::stream_rows_k;

    /** Queries one engine tile holds, the multiple a scan's groups round to. */
    static constexpr std::size_t step_columns_k = tiles_kit_type_::tile_columns_k;

    /** Dots a task stages per row: one per query a pack holds. */
    static constexpr std::size_t dots_columns_k = tiles_kit_type_::pack_columns_k;

    nk_kernel_kind_t kernel;
    nk::matrix_view<scalar_t> queries;
    std::span<norms_t const> query_norms;

    /** The block's pack, which @ref pack wrote. */
    byte_t const* packed;

    /** One task's raw dots, @b [rows,pack], read a quad at a time past the block's last query. */
    nk::matrix_span<raw_dot_t> dots;

    /** Queries a block of @p columns holds: one pack, whatever the cache. */
    static std::size_t block_columns(std::size_t columns, std::size_t, std::size_t) noexcept {
        return min_of(dots_columns_k, columns);
    }

    /** Bytes the packs of @p columns queries of @p dimensions take, each pack on its own line. */
    static std::size_t packs_bytes(std::size_t columns, std::size_t dimensions) noexcept {
        nk_size_t pack_bytes = 0;
        [[maybe_unused]] nk_status_t const sized = tiles_kit_type_::pack_size(dots_columns_k, dimensions, &pack_bytes);
        assert(sized == nk_success_k && "A matrix engine sizes any pack");
        return divide_round_up(columns, dots_columns_k) *
               round_up_to_multiple<std::size_t>(pack_bytes, default_alignment_k);
    }

    /** Packs the block of @p queries, at most a pack of them, into @p packed. */
    static void pack(nk::matrix_view<scalar_t> queries, byte_t* packed) noexcept {
        std::size_t const columns = queries.extent(0);
        [[maybe_unused]] nk_status_t const packing = tiles_kit_type_::pack(
            queries.raw_values_data(), columns, queries.extent(1), static_cast<std::size_t>(queries.stride_bytes(0)),
            packed, 0, columns, nullptr);
        assert(packing == nk_success_k && "A matrix engine packs any block");
    }

    /** Scores @p task against the pack into @p step_distances, @b [rows,queries]: one engine call,
     *  then the seats swapped, as the finalizer's are symmetric, the row against four queries. */
    void score(nk::matrix_view<scalar_t> task, std::span<norms_t const> row_norms,
               nk::matrix_span<f32_t> step_distances) const noexcept {
        std::size_t const rows = task.extent(0), columns = queries.extent(0);
        [[maybe_unused]] nk_status_t const scored = tiles_kit_type_::dots_packed(
            task.raw_values_data(), packed, dots.data(), rows, columns, task.extent(1),
            static_cast<std::size_t>(task.stride_bytes(0)), static_cast<std::size_t>(dots.stride_bytes(0)), nullptr, 0,
            nullptr);
        assert(scored == nk_success_k && "A matrix engine scores any call");
        for (std::size_t row = 0; row != rows; ++row) {
            raw_dot_t const* const row_dots = &dots(row, 0);
            dot_result_t const row_sumsq = row_norms.empty() ? dot_result_t {} : row_norms[row].sumsq;
            f32_t* const distances = &step_distances(row, 0);
            for (std::size_t first = 0; first < columns; first += quad_lanes_k) {
                std::size_t const lanes = min_of(quad_lanes_k, columns - first);
                alignas(16) quad<dot_result_t> sumsqs;
                for (std::size_t lane = 0; lane != quad_lanes_k; ++lane)
                    sumsqs[lane] = query_norms[first + min_of(lane, lanes - 1)].sumsq;
                quad<f32_t> const lane_distances = convert_to_distances<quads_kit_type_>(
                    kernel, *reinterpret_cast<finalized_vec_t const*>(row_dots + first), row_sumsq,
                    *reinterpret_cast<finalized_vec_t const*>(sumsqs.data()));
                std::memcpy(distances + first, lane_distances.data(), lanes * sizeof(f32_t));
            }
        }
    }
};

/** The scorer of an exact scan's blocks: the matrix engine's where the class has one for the
 *  scalar, the walk's quads kit otherwise. */
template <typename tiles_kit_type_, a_quads_kit quads_kit_type_>
struct scan_scorer {
    using scorer_t = tiles_scorer<tiles_kit_type_, quads_kit_type_>;
};

template <a_quads_kit quads_kit_type_>
struct scan_scorer<no_tiles_kit_t, quads_kit_type_> {
    using scorer_t = quads_scorer<quads_kit_type_>;
};

/**
 *  @brief Scores every row of @p rows, the first being node @p first, against the block of
 *      queries @p scorer holds, keeping each row that beats some query's threshold and that
 *      @p admits lets in.
 *
 *  Rows go through @p scorer a task at a time into @p step_distances, @b [rows,queries], beside
 *  their @p row_norms, empty when the rows keep none. @p keep takes a query's column and the scored
 *  node, and answers that query's new threshold. @p admits hears each row beating some threshold
 *  once, a quad of them at a time, or gathered until the task's end when it aggregates, while
 *  @p step_distances still hold the task's distances.
 */
template <typename scorer_type_, typename admits_type_, typename keep_type_>
void scan_block(scorer_type_ const& scorer, nk::matrix_view<typename scorer_type_::scalar_t> rows, node_id_t first,
                std::span<typename scorer_type_::norms_t const> row_norms, std::span<f32_t> thresholds,
                nk::matrix_span<f32_t> step_distances, admits_type_& admits, keep_type_& keep) noexcept {
    std::size_t const rows_count = rows.extent(0), columns = thresholds.size();
    for (std::size_t offset = 0; offset < rows_count; offset += scorer_type_::step_rows_k) {
        std::size_t const step_rows = min_of(scorer_type_::step_rows_k, rows_count - offset);
        scorer.score(rows(nk::range(offset, offset + step_rows), nk::slice),
                     row_norms.empty() ? row_norms : row_norms.subspan(offset, step_rows), step_distances);
        auto distances_of = [&](std::size_t row) noexcept {
            return std::span<f32_t const>(&step_distances(row - offset, 0), columns);
        };
        auto keep_entry = [&](scan_payload_t row) noexcept {
            std::span<f32_t const> const distances = distances_of(row);
            node_id_t const node = static_cast<node_id_t>(first + row);
            for (std::size_t column = 0; column != columns; ++column)
                if (distances[column] < thresholds[column])
                    thresholds[column] = keep(column, scored_node_t {node, distances[column]});
        };
        quad<node_id_t> nodes;
        quad<scan_payload_t> payloads;
        std::size_t filled = 0;
        for (std::size_t row = offset; row != offset + step_rows; ++row) {
            std::span<f32_t const> const distances = distances_of(row);
            bool closer = false;
            for (std::size_t column = 0; column != columns; ++column) closer |= distances[column] < thresholds[column];
            if (!closer) continue;
            nodes[filled] = static_cast<node_id_t>(first + row), payloads[filled] = static_cast<scan_payload_t>(row);
            if (++filled != quad_lanes_k) continue;
            ask_contenders(admits, nodes, payloads, filled, keep_entry);
            filled = 0;
        }
        ask_contenders(admits, nodes, payloads, filled, keep_entry);
        settle_predicate(admits, keep_entry);
    }
}

/**
 *  @brief Folds a gathered block of stored @p rows against the query rows packed in @p packed: the
 *      dots of each segment's rows, those below @p segment_ends[s] and at or past the previous end,
 *      raise segment s's row of @p maxima, one column per packed query row.
 *
 *  Rows stream through the matrix engine @c stream_rows_k at a time into @p dots, as a
 *  @c tiles_scorer streams them; a matrix engine answers inner products, which MaxSim takes
 *  as they are.
 */
template <typename tiles_kit_type_, typename scalar_type_>
void fold_tiles(nk::matrix_view<scalar_type_> rows, void const* packed, std::span<std::size_t const> segment_ends,
                nk::matrix_span<typename scalar_type_::dot_result_t> maxima,
                nk::matrix_span<typename scalar_type_::dot_result_t::raw_t> dots) noexcept {
    using dot_result_t = typename scalar_type_::dot_result_t;
    using raw_dot_t = typename dot_result_t::raw_t;
    std::size_t const rows_count = rows.extent(0), dimensions = rows.extent(1), columns = maxima.extent(1);
    std::size_t segment = 0;
    for (std::size_t offset = 0; offset < rows_count; offset += tiles_kit_type_::stream_rows_k) {
        std::size_t const step_rows = min_of(tiles_kit_type_::stream_rows_k, rows_count - offset);
        nk::matrix_view<scalar_type_> const task = rows(nk::range(offset, offset + step_rows), nk::slice);
        [[maybe_unused]] nk_status_t const scored = tiles_kit_type_::dots_packed(
            task.raw_values_data(), packed, dots.data(), step_rows, columns, dimensions,
            static_cast<std::size_t>(task.stride_bytes(0)), static_cast<std::size_t>(dots.stride_bytes(0)), nullptr, 0,
            nullptr);
        assert(scored == nk_success_k && "A matrix engine scores any call");
        for (std::size_t row = 0; row != step_rows; ++row) {
            while (segment_ends[segment] <= offset + row) ++segment;
            std::span<raw_dot_t const> const row_dots = to_span(dots.row(row).as_vector());
            std::span<dot_result_t> const segment_maxima = to_span(maxima.row(segment).as_vector());
            for (std::size_t column = 0; column != columns; ++column)
                segment_maxima[column] = max_of(segment_maxima[column], dot_result_t::from_raw(row_dots[column]));
        }
    }
}

/** Folds stored @p row against @p query_rows a quad of them a call, handing @p update each of
 *  @p maxima beside its dot; the 4-way sibling of @ref fold_tiles. */
template <typename quads_kit_type_, typename scalar_type_, typename update_type_>
void fold_quads(nk::vector_view<scalar_type_> row, vector_norms<typename scalar_type_::dot_result_t> row_norms,
                nk::matrix_view<scalar_type_> query_rows,
                std::span<vector_norms<typename scalar_type_::dot_result_t> const> query_norms,
                std::span<typename scalar_type_::dot_result_t> maxima, update_type_&& update) noexcept {
    using dot_result_t = typename scalar_type_::dot_result_t;
    for (std::size_t first = 0; first < maxima.size(); first += quad_lanes_k) {
        std::size_t const lanes = min_of(quad_lanes_k, maxima.size() - first);
        quad<nk::vector_view<scalar_type_>> lane_rows;
        quad<vector_norms<dot_result_t>> lane_norms;
        for (std::size_t lane = 0; lane != quad_lanes_k; ++lane) {
            std::size_t const source = first + min_of(lane, lanes - 1);
            lane_rows[lane] = query_rows.row(source).as_vector();
            lane_norms[lane] = query_norms[source];
        }
        scores_result<dot_result_t> const scored = scores<quads_kit_type_>(nk_kernel_dot_k, row, row_norms, lane_rows,
                                                                           lane_norms);
        for (std::size_t lane = 0; lane != lanes; ++lane) update(maxima[first + lane], scored.dots[lane]);
    }
}

#pragma region Rescorings

/** The insertion and refine walks' rescoring: scores stay as the kit scored them, and each merged
 *  quad keeps a zero-distance node only while the candidates hold none, so a duplicate class spends
 *  one rank of the pool instead of all of them. */
struct single_zero_rescoring_t {
    template <typename ids_type_, typename keys_type_, typename norms_type_, typename scores_type_>
    void operator()(ids_type_ const&, keys_type_ const&, norms_type_ const&, scores_type_&,
                    std::size_t) const noexcept {}

    /** The ascending @p scored quad past its leading zeros, all of them once @p candidates hold
     *  one, all but the last otherwise. */
    template <typename candidates_type_>
    static std::span<scored_node_t> admitted(std::span<scored_node_t> scored,
                                             candidates_type_ const& candidates) noexcept {
        std::size_t zeros = 0;
        while (zeros != scored.size() && scored[zeros].distance == 0) ++zeros;
        bool const held = !candidates.empty() && candidates[0].distance == 0;
        return scored.subspan(zeros - (zeros && !held));
    }
};

/** Indexer heads as a walk's rescoring: each scored dot becomes Σₕ wₕ·ReLU(⟨qₕ, k⟩) and its
 *  distance 1 − score. */
template <typename quads_kit_type_>
struct indexer_rescoring {
    using scalar_t = typename quads_kit_type_::scalar_t;

    /** The heads of the one query row being walked. */
    indexer_heads<scalar_t> heads;

    template <typename dot_type_, std::size_t fan_out_>
    void operator()(std::array<node_id_t, fan_out_> const&, std::array<nk::vector_view<scalar_t>, fan_out_> const& keys,
                    std::array<vector_norms<dot_type_>, fan_out_> const& norms,
                    scores_result<dot_type_, fan_out_>& scores, std::size_t count) const noexcept {
        static_assert(!nk::is_integral_dtype<scalar_t>(), "Indexer scoring needs floating-point keys");
        std::array<dot_type_, fan_out_> indexed {};
        for (std::size_t head = 0; head != heads.heads.extent(0); ++head) {
            scores_result<dot_type_, fan_out_> const head_scores = dense::scores<quads_kit_type_>(
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
 *  @c status. A node the predicate rejects still ranks, so the walk routes through it, but folds
 *  into no head and resolves no value row. */
template <typename quads_kit_type_, typename elementwise_kit_type_, typename value_lookup_type_,
          typename predicate_type_ = dummy_predicate_t>
struct attention_rescoring {
    using scalar_t = typename quads_kit_type_::scalar_t;
    using vector_view_t = nk::vector_view<scalar_t>;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vector_norms_t = vector_norms<dot_result_t>;

    /** The token's heads, one row each. */
    nk::matrix_view<scalar_t> queries;

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

    /** Asked about a scored quad of lookup keys; only the lanes it admits fold. */
    predicate_type_ predicate = {};

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
            [[maybe_unused]] nk_status_t const scaled = elementwise_kit_type_::scale(row, columns, &normalizer, &zero,
                                                                                     row, nullptr);
            assert(scaled == nk_success_k && "The serial tier scales on every CPU");
            carry[head].maximum_score *= std::numbers::ln2_v<f32_t>;
        }
    }

    /** Folds one scored batch, its value rows resolved by node ID. */
    void operator()(quad<node_id_t> const& ids, quad<vector_view_t> const& keys, quad<vector_norms_t> const& norms,
                    scores_result<dot_result_t>& scores, std::size_t count) const noexcept {
        (*this)(ids, keys, norms, scores, count, std::span<node_id_t const>(ids).first(count));
    }

    /** Folds one scored batch, its value rows resolved by @p lookup_keys: the rows' own keys, read
     *  beside the vectors by a walk that holds no key column to resolve node IDs through. */
    template <typename lookup_key_type_>
    void operator()(quad<node_id_t> const&, quad<vector_view_t> const& keys, quad<vector_norms_t> const& norms,
                    scores_result<dot_result_t>& scores, std::size_t count,
                    std::span<lookup_key_type_ const> lookup_keys) const noexcept {
        if (failed(status)) return;
        std::size_t const columns = queries.extent(1), head_count = carry.size();
        quad<std::size_t> folded_lanes;
        quad<lookup_key_type_> folded_keys;
        std::size_t const folded = fold_lanes_(lookup_keys.first(count), folded_lanes, folded_keys);
        quad<vector_view_t> rows;
        quad<char const*> rows_bytes;
        with_static_extent<quad_lanes_k>(folded, [&]<std::size_t extent_>() noexcept {
            value_lookup(std::span<lookup_key_type_ const, extent_>(folded_keys.data(), extent_),
                         std::span<vector_view_t, extent_>(rows.data(), extent_));
        });
        for (std::size_t position = 0; position != folded; ++position) {
            if (rows[position].size() != columns) {
                status = status_code_t::shape_mismatch_k;
                return;
            }
            rows_bytes[position] = rows[position].byte_data();
        }
        f32_t const logit_scale = score_scale * std::numbers::log2e_v<f32_t>;
        for (std::size_t first = 0; first < head_count; first += 8) {
            std::size_t const group = min_of<std::size_t>(8, head_count - first);
            // One exp2 tile per scored node across the group's heads; the zeros past them
            // exponentiate harmlessly.
            quad<std::array<f32_t, 8>> weights {};
            std::array<f32_t, 8> corrections {};
            for (std::size_t head = 0; head != group; ++head) {
                vector_view_t const head_row = queries.row(first + head).as_vector();
                // A compensated finalize reads the head's own row sum beside the keys' stored ones.
                vector_norms_t head_norms {};
                if constexpr (quads_compensated<quads_kit_type_>)
                    if (first + head) head_norms = dense::norms<quads_kit_type_>(head_row);
                scores_result<dot_result_t> const head_scores =
                    first + head ? dense::scores<quads_kit_type_>(nk_kernel_dot_k, head_row, head_norms, keys, norms)
                                 : scores;
                softmax_state_t& state = carry[first + head];
                f32_t maximum = state.maximum_score;
                for (std::size_t position = 0; position != folded; ++position) {
                    weights[position][head] = logit_scale *
                                              static_cast<f32_t>(head_scores.dots[folded_lanes[position]]);
                    maximum = max_of(maximum, weights[position][head]);
                }
                for (std::size_t lane = 0; lane != count; ++lane)
                    scores.dots[lane] = max_of(scores.dots[lane], head_scores.dots[lane]);
                for (std::size_t position = 0; position != folded; ++position) weights[position][head] -= maximum;
                corrections[head] = state.maximum_score - maximum;
                state.maximum_score = maximum;
            }
            for (std::size_t position = 0; position != folded; ++position)
                invoke_under_target_<exp2_loop_, elementwise_kit_type_>(std::span(weights[position]));
            invoke_under_target_<exp2_loop_, elementwise_kit_type_>(std::span(corrections));
            for (std::size_t head = 0; head != group; ++head) {
                softmax_state_t& state = carry[first + head];
                state.partition *= corrections[head];
                for (std::size_t position = 0; position != folded; ++position)
                    state.partition += weights[position][head];
            }
            values_step<scalar_t, elementwise_kit_type_>(outputs.row(first).data(), group, columns, rows_bytes.data(),
                                                         weights.data(), corrections, folded);
        }
        for (std::size_t lane = 0; lane != count; ++lane)
            scores.distances[lane] = 1 - static_cast<distance_t>(scores.dots[lane]);
    }

  private:

    /** Every lane folds when nothing filters. */
    template <typename lookup_key_type_>
    std::size_t fold_lanes_(std::span<lookup_key_type_ const> lane_keys, quad<std::size_t>& lanes,
                            quad<lookup_key_type_>& keys) const noexcept
        requires std::same_as<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>
    {
        for (std::size_t lane = 0; lane != lane_keys.size(); ++lane) lanes[lane] = lane, keys[lane] = lane_keys[lane];
        return lane_keys.size();
    }

    /** The lanes the predicate admits and their keys, packed in lane order. */
    template <typename lookup_key_type_>
    std::size_t fold_lanes_(std::span<lookup_key_type_ const> lane_keys, quad<std::size_t>& lanes,
                            quad<lookup_key_type_>& keys) const noexcept {
        std::size_t folded = 0;
        with_static_extent<quad_lanes_k>(lane_keys.size(), [&]<std::size_t extent_>() noexcept {
            std::array<admission_t, extent_> admissions;
            predicate(std::span<lookup_key_type_ const, extent_>(lane_keys.data(), extent_),
                      std::span<admission_t, extent_>(admissions));
            for (std::size_t lane = 0; lane != extent_; ++lane)
                if (admissions[lane] == admission_t::admit_k) lanes[folded] = lane, keys[folded++] = lane_keys[lane];
        });
        return folded;
    }
};

#pragma endregion Rescorings

/** Bytes a verb stages for @p rows rows of @p columns imported as @p dtype: the @c f32 row an
 *  @c f64 input widens through, then the rows, each on its own line. */
inline std::size_t staged_rows_bytes(nk_dtype_t dtype, std::size_t rows, std::size_t columns) noexcept {
    using slice_t = arena_allocator<byte_t>;
    return slice_t::reserved_bytes(columns * sizeof(f32_t)) +
           slice_t::reserved_bytes(rows * packed_bytes(dtype, columns));
}

/** Rows @b [first,first+count) of @p source as @p scalar_type_, through @ref import_rows into
 *  @p staged laid out as @ref staged_rows_bytes counts it. */
template <typename elementwise_kit_type_, typename rmscast_kit_type_, nk::a_numeric_dtype scalar_type_>
nk::matrix_view<scalar_type_> import_staged_rows(rows_view_t source, std::size_t first, std::size_t count,
                                                 nk_kernel_kind_t kernel, byte_t* staged) noexcept {
    std::size_t const columns = source.columns;
    byte_t* const rows = staged + staged_rows_bytes(scalar_type_::dtype(), 0, columns);
    return import_rows<elementwise_kit_type_, rmscast_kit_type_>(
        source, first, count, kernel,
        nk::matrix_span<scalar_type_> {reinterpret_cast<char*>(rows), rows_shape_of<scalar_type_>(count, 0, columns)},
        nk::vector_span<f32_t> {reinterpret_cast<f32_t*>(staged), columns});
}

/** Per-thread traversal scratch: candidate queues, visited set, neighbor selection buffers and
 *  the running telemetry counters - one instance per scanning thread. */
template <                                                    //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,       //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t, //
    typename allocator_type_ = std::allocator<byte_t>>
struct traversal_context;

template <typename quads_kit_type_, typename visits_kit_type_, typename allocator_type_>
struct traversal_context {

    using scalar_t = typename quads_kit_type_::scalar_t;
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

    using dot_result_t = typename scalar_t::dot_result_t;
    using raw_dot_t = typename dot_result_t::raw_t;
    using dots_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<raw_dot_t>;
    using distances_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<f32_t>;
    using bytes_tensor_t = nk::tensor<std::byte, visits_allocator_t, 1>;
    using dots_tensor_t = nk::tensor<raw_dot_t, dots_allocator_t, 2>;
    using distances_tensor_t = nk::tensor<f32_t, distances_allocator_t, 2>;
    using thresholds_tensor_t = nk::tensor<f32_t, distances_allocator_t, 1>;

    /** One selection's output: @b [kept] scored nodes. The tensor's live shape is the fill and
     *  @c capacity() the reservation, so no separate count exists. */
    using selection_t = nk::tensor<scored_node_t, candidates_allocator_t, 1>;

    using results_array_t = sorted_buffer<scored_node_t, closer_scored_node_t, candidates_allocator_t>;

    /** 8-ary: eight 8-byte scored nodes fill one cache line per level scan, and the
     *  insert-dominated frontier rides the shallower log_8 @c shift_up depth. */
    using frontier_queue_t = max_heap_dary<8, scored_node_t, further_scored_node_t, candidates_allocator_t>;

    /** The visited set walks its slots with the visits kit: sixteen at a time on the AVX-512
     *  classes, one at a time below them. */
    using visited_set_t = growing_hash_set<node_id_t, std::hash<node_id_t>, visits_allocator_t, visits_kit_type_>;

    /** Nodes we've already scored, kept ascending as the best results so far. */
    results_array_t results;

    /** Nodes we've already re-evaluated for post-insertion refinement. */
    results_array_t refining_candidates;

    /** Nodes we consider promising and want to evaluate next. */
    frontier_queue_t frontier;

    /** Rows popped from @c frontier a hop ahead of their expansion, their neighbor rows already
     *  streaming in. */
    frontier_queue_t ahead;

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

    /** The nodes one walk takes as landmarks ahead of its random draws, in the order handed in:
     *  one per landmark a walk draws, reserved with the context. */
    ids_tensor_t hinted_ids;

    /** One byte per hint key of the query in flight, set once any shard staged a node under it: a
     *  query takes at most one key per landmark, reserved with the context. */
    bytes_tensor_t hinted_held;

    /** The staging surface: the staged rows during scoring, then the kept prefix of the running
     *  selection - scoring drains the rows before any selection writes. */
    views_tensor_t staged_vectors;

    /** Stored norms, parallel to @c staged_vectors. */
    norms_tensor_t staged_norms;

    /** One verb's queries in the stored scalar, laid out as @ref staged_rows_bytes counts them,
     *  then whatever the verb keeps past them: an attended token's rows and states, or a scan's
     *  packed block. */
    bytes_tensor_t staged_queries;

    /** The raw dots one task of stored rows scores against a packed block, @b [rows,queries]. */
    dots_tensor_t step_dots;

    /** The distances one task of stored rows scores against a block of an exact scan's queries,
     *  @b [rows,queries]. */
    distances_tensor_t step_distances;

    /** An exact scan's query norms, one per query of its group. */
    norms_tensor_t query_norms;

    /** An exact scan's worst kept distance of every query of its group. */
    thresholds_tensor_t thresholds;

    /** Per entry of the neighborhood in hand, its nearest neighborhood-mates by position, a
     *  max-heap each: @b [capacity,connectivity]; a gather keeps each entry's nearest hub in the
     *  first @b [capacity]. */
    selection_t neighborhood_mates;

    /** The stored norms of the entries a neighborhood gathers, in its order: @b [capacity]. */
    norms_tensor_t neighborhood_norms;

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

    /** Hint keys no shard held a node under. */
    std::size_t missing_hints = 0;

    /** Rows a walk keeps popped ahead of its merges, from @ref walk_width; one walks the plain
     *  greedy order. */
    std::size_t walk_width = 1;

    traversal_context() noexcept = delete;

    /** The running counters above as one snapshot, whose difference a refine reports. */
    traversal_stats_t telemetry() const noexcept {
        return {computed_distances,
                visited_members,
                restarts,
                saturated_walks,
                computed_distances_in_refines,
                computed_distances_in_reverse_refines,
                missing_hints};
    }
    traversal_context(traversal_context const&) noexcept = delete;
    traversal_context& operator=(traversal_context const&) noexcept = delete;

    /** Constructs every container with a rebound copy of @p alloc, and keeps the allocator for
     *  the lazily-sized neighbor scratches, so all scratch goes where the caller places it. */
    traversal_context(allocator_t alloc) noexcept
        : results(candidates_allocator_t(alloc)), refining_candidates(candidates_allocator_t(alloc)),
          frontier(candidates_allocator_t(alloc)), ahead(candidates_allocator_t(alloc)),
          visited(visits_allocator_t(alloc)), random_generator(), hinted_ids(ids_allocator_t(alloc)),
          hinted_held(visits_allocator_t(alloc)), allocator_(alloc) {}
    traversal_context(traversal_context&&) = default;
    traversal_context& operator=(traversal_context&&) = default;

    /** Sizes every neighbor scratch for @p capacity entries through the context's own allocator;
     *  geometry is fixed at construction, so the tensors allocate once. */
    status_t reserve_neighbors(std::size_t capacity) noexcept {
        if (selected_neighbors.capacity() >= capacity) return status_code_t::success_k;
        auto selected = selection_t::uninitialized({capacity}, candidates_allocator_t(allocator_));
        auto refined = selection_t::uninitialized({capacity}, candidates_allocator_t(allocator_));
        auto ids = ids_tensor_t::uninitialized({capacity}, ids_allocator_t(allocator_));
        auto views = views_tensor_t::uninitialized({capacity}, views_allocator_t(allocator_));
        auto norms = norms_tensor_t::uninitialized({capacity}, norms_allocator_t(allocator_));
        if (!selected || !refined || !ids || !views || !norms) return status_code_t::out_of_memory_k;
        selected_neighbors = std::move(selected.value);
        refined_neighbors = std::move(refined.value);
        staged_ids = std::move(ids.value);
        staged_vectors = std::move(views.value);
        staged_norms = std::move(norms.value);
        return status_code_t::success_k;
    }

    /** Grows @c staged_queries to @p bytes through the context's own allocator; a grown buffer
     *  keeps nothing, so a verb sizes it before staging into it. */
    status_t reserve_queries(std::size_t bytes) noexcept {
        if (staged_queries.numel() >= bytes) return status_code_t::success_k;
        auto staged = bytes_tensor_t::uninitialized({bytes}, visits_allocator_t(allocator_));
        if (!staged) return status_code_t::out_of_memory_k;
        staged_queries = std::move(staged.value);
        return status_code_t::success_k;
    }

    /** Sizes the @p rows by @p columns dots tile a matrix engine answers; geometry is fixed at
     *  construction, so the tile allocates once per context. */
    status_t reserve_tiles(std::size_t rows, std::size_t columns) noexcept {
        if (step_dots.numel() >= rows * columns) return status_code_t::success_k;
        auto dots = dots_tensor_t::uninitialized({rows, columns}, dots_allocator_t(allocator_));
        if (!dots) return status_code_t::out_of_memory_k;
        step_dots = std::move(dots.value);
        return status_code_t::success_k;
    }

    /** Sizes an exact scan's query norms and thresholds for @p columns queries, the distances one
     *  task of @p step_rows rows scores against them, and the dots tile of @p dots_columns a row
     *  a matrix engine answers; each grows to the widest group the context scans. */
    status_t reserve_scan(std::size_t columns, std::size_t step_rows, std::size_t dots_columns) noexcept {
        if (failed(reserve_tiles(step_rows, dots_columns))) return status_code_t::out_of_memory_k;
        if (step_distances.extent(0) < step_rows || step_distances.extent(1) < columns) {
            auto distances = distances_tensor_t::uninitialized({step_rows, columns}, distances_allocator_t(allocator_));
            if (!distances) return status_code_t::out_of_memory_k;
            step_distances = std::move(distances.value);
        }
        if (query_norms.extent(0) >= columns) return status_code_t::success_k;
        auto norms = norms_tensor_t::uninitialized({columns}, norms_allocator_t(allocator_));
        auto limits = thresholds_tensor_t::uninitialized({columns}, distances_allocator_t(allocator_));
        if (!norms || !limits) return status_code_t::out_of_memory_k;
        query_norms = std::move(norms.value);
        thresholds = std::move(limits.value);
        return status_code_t::success_k;
    }

    /** Sizes a neighborhood's scratch for @p capacity entries under @p connectivity; each tensor
     *  grows once per context. */
    status_t reserve_neighborhoods(std::size_t capacity, std::size_t connectivity) noexcept {
        if (neighborhood_mates.numel() >= capacity * connectivity && neighborhood_norms.numel() >= capacity)
            return status_code_t::success_k;
        auto mates = selection_t::uninitialized({capacity * connectivity}, candidates_allocator_t(allocator_));
        auto norms = norms_tensor_t::uninitialized({capacity}, norms_allocator_t(allocator_));
        if (!mates || !norms) return status_code_t::out_of_memory_k;
        neighborhood_mates = std::move(mates.value);
        neighborhood_norms = std::move(norms.value);
        return status_code_t::success_k;
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

template <                                                                              //
    typename quads_kit_type_, typename tiles_kit_type_, typename elementwise_kit_type_, //
    typename rmscast_kit_type_, typename visits_kit_type_,                              //
    typename wait_type_, template <typename> class atomic_ref_template_,                //
    typename cache_hints_type_, typename vectors_allocator_type_, typename neighbors_allocator_type_,
    typename traversals_allocator_type_, typename key_type_, typename key_hash_type_> //
struct ram_index {

    using scalar_t = typename quads_kit_type_::scalar_t;
    inline static constexpr nk_dtype_t vector_dtype_k = scalar_t::dtype();

    using dot_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;
    using candidate_t = scored_node_t;

    /** What a walk hands an aggregating predicate beside each key it gathers. */
    using walk_payload_t = scored_node_t;

    /** Default integer type to use for loops and indexing at sub-index scale. */
    using candidates_count_t = dense::candidates_count_t;

    /** Node identifier within this sub-index - the class-local spelling of @c dense::node_id_t. */
    using node_id_t = dense::node_id_t;

    /** Streams the rows a walk or a scan reaches next toward the core. */
    using prefetcher_t = object_prefetcher<cache_hints_type_>;

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

    /** The caller-visible identity of every node; several nodes may share one. */
    using key_t = key_type_;
    using keys_view_t = nk::vector_view<key_t>;

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
    template <typename, typename, typename, typename, typename, typename, typename, template <typename> class, typename,
              typename>
    friend struct ram_indexes;
    template <typename, template <typename> class shards_atomic_ref_template_, typename, typename>
    friend struct utilities::shard_set;

    /** The distance kit @c src/ picked; its task is the row pitch the chunks round to. */
    using quads_kit_t = quads_kit_type_;

    /** The matrix-engine kit an exact scan streams through, @c no_tiles_kit_t where no engine
     *  multiplies the scalar. */
    using tiles_kit_t = tiles_kit_type_;

    /** Whether a matrix engine multiplies the scalar, so many queries scan through it at once. */
    inline static constexpr bool has_tiles_k = !std::is_same_v<tiles_kit_t, no_tiles_kit_t>;

    /** The scorer every exact-scan block takes, from the kits the class picked. */
    using scan_scorer_t = typename scan_scorer<tiles_kit_t, quads_kit_t>::scorer_t;

    /** The casts, scales and blends rows import and export through, and the unit-RMS narrowing an
     *  angular import from wide floats takes. */
    using elementwise_kit_t = elementwise_kit_type_;
    using rmscast_kit_t = rmscast_kit_type_;

    /** The visited set's kit. */
    using visits_kit_t = visits_kit_type_;

    /** Each thread scanning the graph needs its own object like this. */
    using traversal_context_t = traversal_context<quads_kit_t, visits_kit_t, candidates_allocator_t>;
    using results_array_t = typename traversal_context_t::results_array_t;
    using frontier_queue_t = typename traversal_context_t::frontier_queue_t;
    using visited_set_t = typename traversal_context_t::visited_set_t;
    using selection_t = typename traversal_context_t::selection_t;
    using ids_tensor_t = typename traversal_context_t::ids_tensor_t;

    /** The growth schedule of the vectors, norms & neighbors - 1'024 rows first, doubling up to
     *  8'388'608-row slabs - shared with the owner columns that grow beside the index.
     *  @ref dense_ram_index_malloc */
    using chunk_schedule_t = chunk_schedule<10, 23>;

    /** A row's chunk and its offset within that chunk. */
    using location_t = typename chunk_schedule_t::location_t;

    /** Ceiling on dispensable node IDs: the all-ones ID is the sentinel and marks a closed
     *  dispenser, so no count reaches it and the final chunk holds two rows that stay forever
     *  unclaimed. @ref dense_ram_index_growth */
    inline static constexpr std::size_t max_node_count_k = (u64_t(1) << 32) - 2u;

    /** Upper bound on chunks: every chunk before the one holding the first undispensable ID.
     *  @ref dense_ram_index_growth */
    inline static constexpr candidates_count_t max_chunks_k = static_cast<candidates_count_t>(
        chunk_schedule_t::locate(max_node_count_k).chunk);

    /** Every node's key in a column beside the rows, and every key's nodes in a map placed by node,
     *  growing in the rows' chunks. */
    using keyed_t = keyed_nodes<key_t, node_id_t, chunk_schedule_t, max_node_count_k, vectors_allocator_type_,
                                atomic_ref_template_, key_hash_type_>;

    /** The dimensions and connectivity every row keeps, fixed at construction. */
    using geometry_t = dense::geometry_t;

    /** The allocator the vectors & norms grow through, whose memory domain the sharded index
     *  places by. */
    [[no_unique_address]] vectors_allocator_t vectors_allocator_;

    ram_index(ram_index const&) noexcept = delete;
    ram_index& operator=(ram_index const&) noexcept = delete;

    ram_index(ram_index&&) noexcept = delete;
    ram_index& operator=(ram_index&&) noexcept = delete;

    /** Constructs an empty index of @p geometry walking by @p kernel, its vectors, norms & keys
     *  growing through @p vectors_allocator and its neighbors through @p neighbors_allocator,
     *  streaming rows ahead of the walks @p line_bytes at a time. Nothing is allocated: a malformed
     *  geometry or a @p kernel the dtype's kits cannot finalize is refused at the first growth. */
    explicit ram_index(nk_kernel_kind_t kernel, geometry_t geometry,
                       vectors_allocator_type_ const& vectors_allocator = {},
                       neighbors_allocator_type_ const& neighbors_allocator = {},
                       std::size_t line_bytes = default_alignment_k) noexcept
        : vectors_allocator_(vectors_allocator), vectors_(vectors_allocator), norms_(vectors_allocator),
          neighbors_(neighbors_allocator), keyed_(vectors_allocator), geometry_(geometry), kernel_(kernel),
          prefetcher_ {line_bytes} {}

#pragma region Measures

    /** The distance kind every walk ranks by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether rows keep norms: every metric but dot finalizes through them, and so does dot on a
     *  kit whose finalize subtracts the candidates' row sums. Otherwise the norms chunks stay
     *  empty, and checkpoints hold no norms column. */
    bool stores_norms() const noexcept { return kernel_ != nk_kernel_dot_k || quads_compensated<quads_kit_t>; }

    /** Number of published vectors - the visibility watermark. */
    std::size_t size() const noexcept {
        return atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The dimensions & connectivity together, fixed at construction. */
    geometry_t geometry() const noexcept { return geometry_; }

    /** Neighbor slots per node - the full fixed capacity, none lost to bookkeeping. */
    std::size_t connectivity() const noexcept { return geometry_.connectivity; }

    /** The most candidates a walk asks an aggregating predicate between two settles: one expanded
     *  row of @p connectivity slots, or one batch of landmarks. */
    static constexpr std::size_t round_capacity(std::size_t connectivity) noexcept {
        return max_of<std::size_t>(connectivity, quad_lanes_k);
    }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    candidates_count_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor
                                       : static_cast<candidates_count_t>(default_expansion_factor(connectivity()));
    }

    /** Scalar dimensions per stored vector. */
    std::size_t dimensions() const noexcept { return geometry_.dimensions; }

    /** Rows hosted without growing again - unclaimed plus claimed. The release is read first:
     *  @ref release_capacity deflates the claim before it releases, so the sum can only dip, which
     *  is the conservative side @c reserve compares against. */
    std::size_t capacity() const noexcept {
        std::ptrdiff_t const unclaimed_rows =
            atomic_ref<std::ptrdiff_t const>(remaining_capacity_).load(std::memory_order_acquire);
        std::size_t const claimed_entries =
            atomic_ref<node_id_t const>(last_allocated_id_).load(std::memory_order_acquire);
        return claimed_entries + static_cast<std::size_t>(unclaimed_rows > 0 ? unclaimed_rows : 0);
    }

    /** Bytes the materialized chunks span - vectors, norms & neighbors derived from the hosted
     *  capacity and the growth schedule, keys from their published chunks - so it stays safe under
     *  concurrent growth. Reserved-but-empty rows count from the moment they are granted. */
    std::size_t memory_usage() const noexcept {
        std::size_t const hosted_rows = capacity();
        candidates_count_t chunks = 0;
        while (chunks != max_chunks_k && chunk_schedule_t::nodes_before(chunks) < hosted_rows) ++chunks;
        std::size_t const rows = chunk_schedule_t::nodes_before(chunks);
        std::size_t bytes = sizeof(ram_index);
        bytes += rows * padded_row_values(geometry_) * sizeof(scalar_t);
        if (stores_norms()) bytes += rows * sizeof(vector_norms_t);
        bytes += rows * geometry_.connectivity * sizeof(node_id_t);
        return bytes + keyed_.memory_usage();
    }

    /** The stored bytes of one node's vector - the ID-addressed read a key lookup or an export
     *  resolves through; IDs at or above @ref size() name unpublished rows. */
    vector_view_t vector(node_id_t node) const noexcept {
        return vector_view_t {vector_for(chunk_schedule_t::locate(node)).data(), dimensions()};
    }

    /** Every node's key, and every key's nodes; a reader passes the watermark it acquired. */
    keyed_t const& keys() const noexcept { return keyed_; }

    /**
     *  The key of the published @p node.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p node is at or past the watermark.
     */
    expected<key_t, status_code_t> key_of(node_id_t node) const noexcept {
        if (node >= size()) return {{}, status_code_t::config_mismatch_k};
        return {keyed_[node], status_code_t::success_k};
    }

    /** Invokes @p callback for every published node stored under @p key, as of one acquire of the
     *  watermark - wait-free. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        keyed_.find_each(key, static_cast<node_id_t>(size()), callback);
    }

    /** Landmarks one walk draws before it expands: a row's worth, in whole quads of the scoring
     *  kernels, which also bounds the hint keys a query takes and the nodes they seed. */
    std::size_t landmarks_count() const noexcept {
        return divide_round_up<std::size_t>(connectivity(), quad_lanes_k) * quad_lanes_k;
    }

    /**
     *  @brief Stages into @p slots the published nodes below @p first_invisible_id stored under
     *      each of @p hints, in hint order, as many as fit, marking in @p held each hint held.
     *  @return The nodes staged.
     */
    std::size_t stage_hints_(std::span<key_t const> hints, node_id_t first_invisible_id, std::span<node_id_t> slots,
                             bytes_span_t held) const noexcept {
        std::size_t staged_count = 0;
        for (std::size_t hint = 0; hint != hints.size(); ++hint)
            keyed_.find_each(hints[hint], first_invisible_id, [&](node_id_t node) noexcept {
                held[hint] = std::byte {1};
                if (staged_count != slots.size()) slots[staged_count++] = node;
            });
        return staged_count;
    }

    /** Published nodes stored under @p key, as of one acquire of the watermark. */
    std::size_t count(key_t const& key) const noexcept { return keyed_.count(key, static_cast<node_id_t>(size())); }

    /** Whether any published node is stored under @p key, as of one acquire of the watermark. */
    bool contains(key_t const& key) const noexcept { return keyed_.contains(key, static_cast<node_id_t>(size())); }

    /** Exports up to @p wanted rows stored under @p key into @p vectors in their scalar, and
     *  reports the @b true total under that key - a return above @p wanted means the copy
     *  truncated; zero means absent. Rows at least as wide as the index take its columns and keep
     *  their tails. */
    std::size_t get(key_t const& key, rows_span_t vectors, std::size_t wanted) const noexcept {
        rows_span_t const target {vectors.data, vectors.rows, dimensions(), vectors.stride, vectors.dtype};
        std::size_t total = 0;
        find_each(key, [&](node_id_t node) noexcept {
            if (total < wanted && total < target.rows)
                export_row<elementwise_kit_t>(vector(node).byte_data(), vector_dtype_k, target, total);
            ++total;
        });
        return total;
    }

    /** Storage values per row under @p geometry - the copy width every caller's
     *  matrix must be as wide as. */
    static std::size_t stored_row_values(geometry_t geometry) noexcept {
        return divide_round_up<std::size_t>(geometry.dimensions, nk::dimensions_per_value<scalar_t>());
    }

    /** Storage values per row as the chunks hold it: whole tasks of the picked quads kit,
     *  zero-tailed like the GPU's fragments, so every task load stays aligned. Checkpoints and
     *  copies keep the logical @ref stored_row_values. */
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

        /** Neighbor IDs per neighbors-matrix row - the index's connectivity. */
        std::size_t connectivity = 0;
    };

    /** The shapes a @c save() of the current contents fills. */
    saved_shape_t saved_shape() const noexcept {
        std::size_t const count = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        return {count, geometry_.dimensions, geometry_.connectivity};
    }

    /** Where a save writes: row-major columns the caller sized from @ref saved_shape, handed
     *  over at planning so no task names a destination of its own. */
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

        /** Neighbor IDs per row of @c neighbors, at most the index's connectivity, a load padding
         *  each row with empty slots for a refine to fill; zero reads the index's own. */
        std::size_t connectivity = 0;
    };

#pragma endregion Measures

#pragma region Reservation

    /** Optional tuning knobs for @c insert(); zeros mean "derive from connectivity". */
    using insert_config_t = dense::insert_config_t;

    /** Requested capacity for a @c reserve() call. */
    struct reserve_capacity_t {

        /** Minimal number of entries the index must be able to host. */
        std::size_t capacity = 0;

        /** Rows the index may ever host, clamped to the ID space; a reservation only lowers it, and
         *  past it every claim answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;

        /** A request for @p entries; the ceiling is set by name only, so no positional field goes
         *  on it. */
        reserve_capacity_t(std::size_t entries = 0) noexcept : capacity(entries) {}
    };

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries, growing when
     *      needed, and answers the capacity now in effect - a hint, since every claim grows the
     *      storage it needs.
     *
     *  Safe to call concurrently with insertions, searches and other reservations: growth runs
     *  the same grower-election protocol as @c insert. Constructors allocate nothing, so every
     *  storage failure surfaces at the first growth, here or in a claim, and so does a malformed
     *  geometry or a kernel the dtype's kits cannot finalize.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k The dtype's kits cannot finalize the kernel.
     *  - @c config_mismatch_k The geometry has no dimensions, leaves a storage value partly filled,
     *    or has a connectivity that is not a power of two.
     *  - @c overflow_k The capacity passes the reserved ceiling or the chunk directory.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        // Racing reservations keep the lowest ceiling any of them asked for.
        atomic_ref<std::size_t>(ceiling_nodes_).store_min(config.ceiling, std::memory_order_relaxed);

        while (true) {
            std::size_t const total_nodes = capacity();
            if (total_nodes >= config.capacity) return {total_nodes, status_code_t::success_k};

            // Claim from the claimed rows up to the request, not the deficit alone: a deficit that
            // fits in the free rows is granted without a growth and released, forever. Only a claim
            // crossing the free rows elects a grower, and one growth covers it.
            std::size_t const claimed_entries =
                atomic_ref<node_id_t>(last_allocated_id_).load(std::memory_order_acquire);
            std::size_t const claim_rows = config.capacity - claimed_entries;
            auto const [first_row, status] = claim_capacity(claim_rows);
            if (failed(status)) return {capacity(), status};
            release_capacity(claim_rows);
        }
    }

    /** The five queue reservations a walk starts from, for one expansion factor and
     *  connectivity. */
    struct queue_sizes_t {

        /** The walk's candidates - the expansion factor - plus a re-selected node's own row. */
        std::size_t results;

        /** Every neighbor plus the appended node while back-linking, or, while filtering, the
         *  admitted candidates. */
        std::size_t refining_candidates;

        /** Full candidates' unexpanded members plus one expansion's admissions, which shedding past
         *  the candidates' radius keeps the walk within; only ties at that radius grow it. */
        std::size_t frontier;

        /** One node's neighbors per expansion - the first size a re-table doubles from. */
        std::size_t visited;

        /** Rows held a hop ahead: at most the rows whose neighbors fill a walk's candidates, where
         *  @ref walk_width caps. */
        std::size_t ahead;
    };

    static constexpr queue_sizes_t queue_sizes_(std::size_t expansion_factor, std::size_t connectivity) noexcept {
        return {expansion_factor + connectivity, max_of<std::size_t>(connectivity + 1, expansion_factor),
                expansion_factor + connectivity, expansion_factor * connectivity,
                filling_nodes(expansion_factor, connectivity)};
    }

    static status_t reserve_queues_(traversal_context_t& context, queue_sizes_t sizes) noexcept {
        if (status_t const sized = context.results.reserve(sizes.results); failed(sized)) return sized;
        if (status_t const sized = context.refining_candidates.reserve(sizes.refining_candidates); failed(sized))
            return sized;
        if (status_t const sized = context.frontier.reserve(sizes.frontier); failed(sized)) return sized;
        if (status_t const sized = context.ahead.reserve(sizes.ahead); failed(sized)) return sized;
        return context.visited.reserve(sizes.visited);
    }

    /**
     *  @brief Bytes one context reserves for @p config under the index's geometry.
     *
     *  The first size of every structure a walk touches, reservation for reservation with
     *  @ref reserve_traversal. The visited set may still re-table as a walk widens.
     */
    std::size_t traversal_memory_usage(insert_config_t config = {}) const noexcept {
        std::size_t const node_connectivity = geometry_.connectivity;
        std::size_t const expansion_factor = non_zero_expansion_factor(config);
        using context_t = traversal_context_t;
        using arena_t = arena_allocator<std::byte>;
        queue_sizes_t const sizes = queue_sizes_(expansion_factor, node_connectivity);
        std::size_t bytes =
            arena_t::reserved_bytes(context_t::results_array_t::memory_usage(sizes.results)) +
            arena_t::reserved_bytes(context_t::results_array_t::memory_usage(sizes.refining_candidates)) +
            arena_t::reserved_bytes(context_t::frontier_queue_t::memory_usage(sizes.frontier)) +
            arena_t::reserved_bytes(context_t::frontier_queue_t::memory_usage(sizes.ahead)) +
            arena_t::reserved_bytes(context_t::visited_set_t::memory_usage(sizes.visited)) +
            context_t::neighbors_memory_usage(node_connectivity) +
            arena_t::reserved_bytes(landmarks_count() * sizeof(node_id_t)) +
            arena_t::reserved_bytes(landmarks_count()) + staged_rows_bytes(vector_dtype_k, 1, geometry_.dimensions);
        return bytes;
    }

    /**
     *  @brief Sizes @p context 's scratch for the index's geometry and @p config - the other
     *      fallible half of an insertion, hoistable ahead of a fan-out.
     *
     *  Idempotent and monotone. The candidates and the neighbor scratch never allocate again and
     *  the frontier sheds within its reservation; the visited set alone may re-table as a walk
     *  widens. The staged queries hold one imported row, so an inserted row or a lone query never
     *  makes them grow.
     *
     *  @par Statuses:
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     */
    status_t reserve_traversal(traversal_context_t& context, insert_config_t config = {}) noexcept {
        std::size_t const node_connectivity = geometry_.connectivity;
        std::size_t const expansion_factor = non_zero_expansion_factor(config);

        // `sorted_buffer::insert(.., limit)` never grows on its own, and `growing_hash_set::set`
        // never rehashes, so these reservations are load-bearing rather than an optimization.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, node_connectivity));
            failed(sized))
            return sized;
        if (status_t const sized = context.reserve_neighbors(node_connectivity); failed(sized)) return sized;
        // Hints and stitch seeds stage into these without growing them.
        std::size_t const landmarks = landmarks_count();
        if (nk::failed(context.hinted_ids.reserve(landmarks)) || nk::failed(context.hinted_held.reserve(landmarks)) ||
            nk::failed(context.hinted_ids.resize({landmarks})))
            return status_code_t::out_of_memory_k;
        return context.reserve_queries(staged_rows_bytes(vector_dtype_k, 1, geometry_.dimensions));
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
     *  @brief Inserts a batch of @p new_vectors under @p keys, reporting each one's assigned node
     *      ID.
     *
     *  One compare-exchange dispenses the batch's IDs; every row stores its key and payload, walks
     *  the graph for candidates and is wired, then one watermark turn files the keys and publishes
     *  them all. All fallible scratch allocations happen before the dispense; once IDs are
     *  dispensed, this thread must roll forward to publication. @ref dense_ram_index_concurrency
     *
     *  @param[in] keys One key per row; several rows may share one.
     *  @param[in] new_vectors The batch to insert, one row per vector, in the index's geometry.
     *  @param[out] assigned_ids Either empty, or one slot per incoming vector receiving its node
     *      ID. One batch's IDs are contiguous, in the order of its rows.
     *  @param[in,out] context The calling thread's scratch & telemetry, reused across batches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p keys or a non-empty @p assigned_ids differs in length from the
     *    batch.
     *  - @c config_mismatch_k The rows' width differs from the dimensions, or the geometry is
     *    malformed.
     *  - @c unknown_kind_k The dtype's kits cannot finalize the kernel.
     *  - @c overflow_k A queue capacity of @p context wraps when rounded, or the batch passes the
     *    reserved ceiling or the chunk directory.
     *  - @c contended_k The dispenser is closed by a refine or a load.
     */
    expected<insert_result_t> insert(keys_view_t keys, vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {

        std::size_t const incoming_vectors_count = new_vectors.extent(0); // rows, not scalars
        if (keys.size() != incoming_vectors_count) return {{size()}, status_code_t::config_mismatch_k};
        if (!incoming_vectors_count) return {{size()}, status_code_t::success_k};
        bool const report_assigned_ids = assigned_ids.size() != 0;
        if (report_assigned_ids && assigned_ids.size() != incoming_vectors_count)
            return {{size()}, status_code_t::config_mismatch_k};

        // Snapshot the context's running telemetry, to report this batch's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const refines_before = context.computed_distances_in_refines;
        std::size_t const reverse_refines_before = context.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        // Every row goes into a slot as wide as the geometry, so a wider row would overflow it.
        if (new_vectors.extent(1) != dimensions()) return {{size()}, status_code_t::config_mismatch_k};

        if (status_t const sized = reserve_traversal(context, config); failed(sized)) return {{size()}, sized};

        // From here enough sentinel-filled rows exist for every node, and this thread must roll
        // forward to publishing every ID it claims.
        auto [batch_claim, status] = claim(incoming_vectors_count);
        if (failed(status)) return {{size()}, status};
        if (status_t const inserted = insert(std::move(batch_claim), keys, new_vectors, assigned_ids, context, config);
            failed(inserted))
            return {{size()}, inserted};

        insert_result_t result;
        result.new_size = size();
        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.computed_distances_in_refines = context.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = context.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = context.restarts - restarts_before;
        result.saturated_walks = context.saturated_walks - saturated_walks_before;
        return {result, status_code_t::success_k};
    }

    /** Entry removals are banned to simplify concurrency management. */
    status_t remove(node_ids_view_t, traversal_context_t&) noexcept = delete;

    /**
     *  @brief Storage claimed in this index and not yet inserted against - the fallible half of an
     *      insertion, hoisted so a caller spanning several indexes commits or refuses as a whole.
     *
     *  Releasing is automatic and two-sided, so an abandoned claim leaves the physical-rows sum as
     *  it found it. Rows an ID was dispensed against never come back, which is why consuming a
     *  claim and dispensing against it are the same task. @ref dense_ram_index_concurrency
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

        /** Entries this claim still holds; zero once an insertion has taken it. */
        std::size_t entries() const noexcept { return rows_; }

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

    /**
     *  @brief Claims storage for @p count rows without assigning any identity - the task a batch
     *      spanning several indexes runs on every one of them before touching any.
     *
     *  @par Statuses:
     *  - @c overflow_k @p count passes the reserved ceiling or the chunk directory.
     *  - @c config_mismatch_k or @c unknown_kind_k A growth meets a malformed geometry or kernel.
     */
    expected<capacity_claim_t> claim(std::size_t count) noexcept {
        auto const [first_row, status] = claim_capacity(count);
        if (failed(status)) return {capacity_claim_t {}, status};
        return {capacity_claim_t {*this, count, first_row}, status_code_t::success_k};
    }

    /**
     *  @brief Returns @p count rows no ID was dispensed against.
     *
     *  Deflating the claim watermark before releasing the capacity is load-bearing: a reader taking
     *  the release with its acquire sees the deflation too, so the physical-rows sum it forms stays
     *  conservative. The deflation releases as well - a posted, order-free subtract could arrive
     *  after the release. The release applies only on a non-negative counter: a negative one has a
     *  grower inside, and lifting it back across zero would elect a second. The floor is the count
     *  itself, so <tt>observed + count &gt;= count</tt> admits exactly the non-negative counters and rejects
     *  every negative one.
     */
    void release_capacity(std::size_t count) noexcept {
        atomic_ref<node_id_t>(last_allocated_id_).store_sub(static_cast<node_id_t>(count), std::memory_order_release);
        atomic_ref<std::ptrdiff_t> const remaining {remaining_capacity_};
        std::ptrdiff_t const released = static_cast<std::ptrdiff_t>(count);
        std::ptrdiff_t observed = remaining.load(std::memory_order_acquire);
        while (true) {
            for (; observed < 0; observed = remaining.load(std::memory_order_acquire))
                wait_type_ {}(&remaining_capacity_, observed, 0u, fu::wait_capped_t {});
            observed = remaining.fetch_add_if_at_least(released, released, std::memory_order_release);
            if (observed >= 0) return;
        }
    }

    /**
     *  @brief Inserts against capacity @p claim already obtained and scratch already reserved - the
     *      half of an insertion that fails only on a closed dispenser, before dispensing any ID, so
     *      a batch spanning several indexes commits or is refused as a whole. Consumes @p claim
     *      unless refused; telemetry accrues in @p context.
     *
     *  One compare-exchange dispenses the claim's IDs, every row stores its key of @p keys and is
     *  wired, and one watermark turn files the keys and publishes them together: the turn holder is
     *  the key map's one writer, so filing takes no read-modify-write.
     *
     *  @par Statuses:
     *  - @c contended_k The dispenser is closed by a refine or a load; the claim's death
     *    releases its rows.
     */
    status_t insert(capacity_claim_t&& claim, keys_view_t keys, vectors_view_t new_vectors,
                    node_ids_span_t assigned_ids, traversal_context_t& context, insert_config_t config = {}) noexcept {
        assert(claim.entries() == new_vectors.extent(0) && claim.entries() == keys.size() &&
               "A claim holds exactly its batch's rows and keys");
        // An empty range takes no turn: its first ID opens the next range too, which may publish first.
        if (!claim) return status_code_t::success_k;
        bool const report_assigned_ids = assigned_ids.size() != 0;
        rows_view_t const new_rows = new_vectors;
        auto const [range, dispense_status] = dispense_(std::move(claim));
        if (failed(dispense_status)) return dispense_status;
        for (node_id_t const new_node_id : range) {
            std::size_t const row = new_node_id - range.first;
            if (report_assigned_ids) assigned_ids[row] = new_node_id;
            keyed_.assign(new_node_id, keys[row]);
            wire_(new_node_id, new_rows, row, context, config);
        }
        await_turn_(range);
        keyed_.publish(range);
        publish_(range);
        return status_code_t::success_k;
    }

#pragma endregion Insertion

#pragma region Refinement

    /** Optional tuning knobs for @c refine(); zeros mean "derive from connectivity". */
    using refine_config_t = dense::refine_config_t;

    /** A reshard's knobs: its rounds', and the seed that orders the parts' growth. */
    using reshard_config_t = dense::reshard_config_t;

    using refine_report_t = dense::refine_report_t;
    using reachability_t = dense::reachability_t;

    /**
     *  @brief One reachability audit in flight: the strong component of one drawn root, and the
     *      islands outside it. Read-only on the graph, so concurrent insertions leave fresh nodes
     *      uncounted and nothing else.
     *
     *  Tasks are the published IDs one bitset word at a time, so any number of threads may
     *  @ref sweep different tasks of one pass at once; @ref advance_depth is the boundary, one
     *  caller. A depth expands only the entries the depth before it reached, so depths count from
     *  the root and in-link and shallower in-link counts are exact, whatever order the tasks run
     *  in; the same read marks an entry's way to the root when one of its links leads to an entry
     *  already reaching the root. Once a depth reaches nothing, rechecks re-read the entries
     *  reached but not yet marked. One thread then gathers the islands, the entries outside the
     *  root's component, into a compact adjacency of their links and finds their strong components;
     *  a drawn root whose component is smaller than one of them gives way once to that island's
     *  first entry. A count pass then reads every row once more for the in-link counts, and the
     *  islands' roles and weak components follow.
     */
    struct health_plan_t {

        /** One unit of a pass's fan-out. */
        struct task_t {

            /** The task's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** The IDs this task sweeps. */
            iota_view<node_id_t> nodes;
        };

        using bitset_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<u1x64_t>;
        using growing_bitset_t = growing_buffer<u1x64_t, bitset_allocator_t>;
        using byte_column_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<u8_t>;
        using growing_byte_column_t = growing_buffer<u8_t, byte_column_allocator_t>;
        using island_ids_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<node_id_t>;
        using island_ids_t = growing_buffer<node_id_t, island_ids_allocator_t>;

        /** The depth of a node no sweep reached, 255; the deepest a reached node records is 254,
         *  at which the depths saturate: every node reached from there on reads 254 and counts no
         *  shallower in-links, so no merge drops a link for a depth it cannot tell. */
        inline static constexpr u8_t unknown_depth_k = (std::numeric_limits<u8_t>::max)();

        /** The audit's bitsets, one run of words each in @c bits_, in this order. */
        enum bitset_t : std::size_t {

            /** Reached from the root. */
            reached_k,

            /** Reaches the root through some link in its row. */
            reaches_root_k,

            /** Links to two entries of smaller depth to root, so one link may go
             *  to a reconnection. */
            two_ways_to_root_k,

            /** The bitsets in all. */
            bitsets_k,
        };

        /** An island node's roles, the bits of its byte in @c island_roles_; the first two are its
         *  strong component's. */
        enum island_role_t : u8_t {

            /** No entry outside its component links into it. */
            unreached_island_k = 1,

            /** No link leaves its component. */
            stranded_island_k = 2,

            /** It links into the root's component. */
            links_to_component_k = 4,

            /** An entry of the root's component links to it. */
            linked_from_component_k = 8,
        };

        /** Lanes one word holds - what an ID splits on. */
        inline static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

        /** Words one bitset over @p rows nodes takes. */
        static constexpr std::size_t words_for(std::size_t rows) noexcept {
            return divide_round_up<std::size_t>(rows, lanes_k);
        }

        /** The watermark snapshot the audit covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** An unbound plan - a refused audit's answer, and what @ref widen sizes; it owns its
         *  columns, so it names an allocator. */
        explicit health_plan_t(bitset_allocator_t const& allocator) noexcept
            : bits_(allocator), depths_from_root_(byte_column_allocator_t {allocator}),
              depths_to_root_(byte_column_allocator_t {allocator}),
              shallower_in_links_(byte_column_allocator_t {allocator}), in_links_(byte_column_allocator_t {allocator}),
              island_nodes_(island_ids_allocator_t {allocator}), island_components_(island_ids_allocator_t {allocator}),
              island_parents_(island_ids_allocator_t {allocator}), island_entries_(island_ids_allocator_t {allocator}),
              island_roles_(byte_column_allocator_t {allocator}), island_walk_(island_ids_allocator_t {allocator}) {}
        health_plan_t(health_plan_t&& other) noexcept
            : visible(other.visible), index_(std::exchange(other.index_, nullptr)), bits_(std::move(other.bits_)),
              depths_from_root_(std::move(other.depths_from_root_)), depths_to_root_(std::move(other.depths_to_root_)),
              shallower_in_links_(std::move(other.shallower_in_links_)), in_links_(std::move(other.in_links_)),
              depth_entries_(other.depth_entries_), island_nodes_(std::move(other.island_nodes_)),
              island_components_(std::move(other.island_components_)),
              island_parents_(std::move(other.island_parents_)), island_entries_(std::move(other.island_entries_)),
              island_roles_(std::move(other.island_roles_)), island_walk_(std::move(other.island_walk_)),
              root_(other.root_), drawn_root_(other.drawn_root_), largest_island_(other.largest_island_),
              largest_island_entries_(other.largest_island_entries_), reached_in_depth_(other.reached_in_depth_),
              marked_in_depth_(other.marked_in_depth_), swept_entries_(other.swept_entries_),
              unlinked_(other.unlinked_), underlinked_(other.underlinked_),
              unreached_islands_(other.unreached_islands_), stranded_islands_(other.stranded_islands_),
              components_(other.components_), depth_from_root_(other.depth_from_root_),
              depth_to_root_(other.depth_to_root_), depth_(other.depth_), step_(other.step_),
              islands_status_(other.islands_status_) {}
        health_plan_t& operator=(health_plan_t&& other) noexcept {
            visible = other.visible;
            index_ = std::exchange(other.index_, nullptr);
            bits_ = std::move(other.bits_);
            depths_from_root_ = std::move(other.depths_from_root_);
            depths_to_root_ = std::move(other.depths_to_root_);
            shallower_in_links_ = std::move(other.shallower_in_links_);
            in_links_ = std::move(other.in_links_);
            depth_entries_ = other.depth_entries_;
            island_nodes_ = std::move(other.island_nodes_);
            island_components_ = std::move(other.island_components_);
            island_parents_ = std::move(other.island_parents_);
            island_entries_ = std::move(other.island_entries_);
            island_roles_ = std::move(other.island_roles_);
            island_walk_ = std::move(other.island_walk_);
            root_ = other.root_;
            drawn_root_ = other.drawn_root_;
            largest_island_ = other.largest_island_;
            largest_island_entries_ = other.largest_island_entries_;
            reached_in_depth_ = other.reached_in_depth_;
            marked_in_depth_ = other.marked_in_depth_;
            swept_entries_ = other.swept_entries_;
            unlinked_ = other.unlinked_;
            underlinked_ = other.underlinked_;
            unreached_islands_ = other.unreached_islands_;
            stranded_islands_ = other.stranded_islands_;
            components_ = other.components_;
            depth_from_root_ = other.depth_from_root_;
            depth_to_root_ = other.depth_to_root_;
            depth_ = other.depth_;
            step_ = other.step_;
            islands_status_ = other.islands_status_;
            return *this;
        }
        health_plan_t(health_plan_t const&) = delete;
        health_plan_t& operator=(health_plan_t const&) = delete;

        /** Task @p position of the plan, the nodes of one bitset word, since a task writes its own
         *  nodes' depths to root without contention; an empty task past the end. */
        task_t operator[](std::size_t position) const noexcept {
            task_t task;
            if (position >= tasks().size()) return task;
            std::size_t const first = position * lanes_k;
            std::size_t const bound = min_of(first + lanes_k, static_cast<std::size_t>(visible));
            task.position = position;
            task.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return task;
        }

        /** The plan's tasks, positionally, one per bitset word; every pass sweeps all of them. */
        indexed_range<health_plan_t> tasks() const noexcept { return {this, index_ ? words_for(visible) : 0u}; }

        /** Turns the audit onto @p index 's IDs below @p visible, which its columns must span, and
         *  restarts the sweep from a fresh draw. */
        void rebind(ram_index const& index, node_id_t visible) noexcept {
            index_ = &index;
            this->visible = visible;
            root_ = node_id_unknown_k;
            restart_sweep();
        }

        /**
         *  @brief Takes the IDs up to @p visible in without a sweep.
         *
         *  The rows appended since read unreached both ways, at no depth, with no shallower
         *  in-links or in-links, the bitsets regrow to span every one of them, and room is placed
         *  for an island node per bitset word.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k The bitsets, the depths or the islands' room cannot grow; the audit
         *    stays as it was.
         */
        status_t widen(node_id_t visible) noexcept {
            if (visible <= this->visible) return status_code_t::success_k;
            std::size_t const words = words_(), wider = words_for(visible);
            if (failed(bits_.reserve_exactly(bitsets_k * wider)) ||
                failed(depths_from_root_.reserve_exactly(visible)) ||
                failed(depths_to_root_.reserve_exactly(visible)) ||
                failed(shallower_in_links_.reserve_exactly(visible)) || failed(in_links_.reserve_exactly(visible)) ||
                failed(reserve_islands_(wider)))
                return status_code_t::out_of_memory_k;
            bits_.extend_uninitialized(bitsets_k * wider);
            // Back to front, each run moves up past where the next one lower still reads.
            for (std::size_t bitset = bitsets_k; bitset-- != 0;) {
                std::memmove(bits_.data() + bitset * wider, bits_.data() + bitset * words, words * sizeof(u1x64_t));
                std::memset(bits_.data() + bitset * wider + words, 0, (wider - words) * sizeof(u1x64_t));
            }
            std::size_t const added = visible - this->visible;
            for (growing_byte_column_t* column :
                 {&depths_from_root_, &depths_to_root_, &shallower_in_links_, &in_links_})
                column->extend_uninitialized(max_of<std::size_t>(column->size(), visible));
            std::memset(depths_from_root_.data() + this->visible, unknown_depth_k, added);
            std::memset(depths_to_root_.data() + this->visible, unknown_depth_k, added);
            std::memset(shallower_in_links_.data() + this->visible, 0, added);
            std::memset(in_links_.data() + this->visible, 0, added);
            this->visible = visible;
            return status_code_t::success_k;
        }

        /** Clears every column and starts from the root an earlier sweep kept, or else from a
         *  seeded draw, marked both ways at depth zero: the next @ref sweep starts depth one. */
        void restart_sweep() noexcept {
            // Opening and closing health checks share their root, so a different draw cannot
            // read as progress.
            bool const kept = root_ < visible;
            node_id_t const root = kept      ? root_
                                   : visible ? static_cast<node_id_t>(seeded_bijection(0, visible, 0))
                                             : node_id_unknown_k;
            drawn_root_ = kept ? node_id_unknown_k : root;
            start_from_(root);
        }

        /**
         *  @brief Runs @p task of the current pass: expands the entries of this depth, marking
         *      their way to the root in the same read; re-reads the entries reached but not yet
         *      marked; or counts every entry's links.
         *
         *  Safe on any thread for disjoint tasks: the reached bits, depths, depths to root and
         *  counts other tasks read are written through relaxed atomics, and a task marks only its
         *  own nodes, each once.
         *
         *  @return The nodes this call reached, marked or counted.
         */
        std::size_t sweep(task_t const& task) noexcept {
            if (task.nodes.empty()) return 0;
            if (step_ == step_t::expand_k) return expand_(task);
            if (step_ == step_t::recheck_k) return recheck_reaches_root_(task);
            if (step_ == step_t::count_links_k) return count_in_and_out_links_(task);
            return 0;
        }

        /** The boundary between passes, one caller after every task ran: files the entries a
         *  depth reached, turns to the rechecks once a depth reaches nothing, gathers the
         *  islands and keeps the root or re-roots once a recheck marks nothing, and labels the
         *  islands after the count; zero means the audit converged or could not gather them. */
        std::size_t advance_depth() noexcept {
            if (step_ == step_t::expand_k) {
                std::size_t const reached = std::exchange(reached_in_depth_, 0);
                swept_entries_ += reached + std::exchange(marked_in_depth_, 0);
                if (!reached) {
                    step_ = step_t::recheck_k;
                    return 1;
                }
                depth_entries_[min_of<std::size_t>(++depth_, unknown_depth_k - 1u)] += reached;
                ++depth_from_root_;
                return reached;
            }
            if (step_ == step_t::recheck_k) {
                std::size_t const marked = std::exchange(marked_in_depth_, 0);
                if (!marked) return choose_root_();
                swept_entries_ += marked;
                ++depth_to_root_;
                return 1;
            }
            if (step_ == step_t::count_links_k) {
                label_islands_();
                step_ = step_t::done_k;
            }
            return 0;
        }

        /** The depth from the root the sweep reached @p node_id at, @c unknown_depth_k
         *  for none, which any thread may read between depths. */
        u8_t depth_from_root(node_id_t node_id) const noexcept { return depths_from_root_[node_id]; }

        /** One past the depth to root of the entry that marked @p node_id 's way to the
         *  root, zero for the root, @c unknown_depth_k for none, which any thread may
         *  read between depths. */
        u8_t depth_to_root(node_id_t node_id) const noexcept { return depths_to_root_[node_id]; }

        /** Reached entries linking to @p node_id, saturating at 255, which any thread may read
         *  between depths. */
        u8_t in_links(node_id_t node_id) const noexcept {
            return atomic_ref<u8_t const>(in_links_[node_id]).load(std::memory_order_relaxed);
        }

        /** The depth holding the most entries, the shallowest of equals: the depth the deep stratum
         *  draws past. */
        u8_t most_populated_depth() const noexcept {
            std::size_t most_populated = 0;
            for (std::size_t depth = 1; depth != depth_entries_.size(); ++depth)
                if (depth_entries_[depth] > depth_entries_[most_populated]) most_populated = depth;
            return static_cast<u8_t>(most_populated);
        }

        /** Takes one of @p node_id 's links from the depth before it for a merge that drops it,
         *  answering whether another remains; a refusal leaves it untouched, so no row loses
         *  the last shallower in-link it is reached through. A count saturated at 255 refuses
         *  early, never late. */
        bool take_shallower_in_link_if_spare(node_id_t node_id) noexcept {
            atomic_ref<u8_t> const shallower_in_links {shallower_in_links_[node_id]};
            u8_t observed = shallower_in_links.load(std::memory_order_relaxed);
            while (observed > 1 &&
                   !shallower_in_links.compare_exchange_weak(observed, static_cast<u8_t>(observed - 1),
                                                             std::memory_order_relaxed, std::memory_order_relaxed)) {}
            return observed > 1;
        }

        /** Takes the link from @p owner to @p resident for a swap: one not from the depth before
         *  the resident goes freely, a shallower in-link only while another remains, as @ref
         * take_shallower_in_link_if_spare takes it, so no depth ever rises. */
        bool take_link_for_swap_if_spare(node_id_t owner, node_id_t resident) noexcept {
            u8_t const owner_depth = depth_from_root(owner);
            u8_t const child_depth = owner_depth < unknown_depth_k - 1 ? static_cast<u8_t>(owner_depth + 1)
                                                                       : unknown_depth_k;
            return child_depth == unknown_depth_k || depth_from_root(resident) != child_depth ||
                   take_shallower_in_link_if_spare(resident);
        }

        /** Takes the link from @p owner to @p resident for a swap under the owner's lock,
         *  answering whether the reach rule releases it: never the resident's last shallower
         *  in-link, and a link toward the root only while the owner keeps another at that
         *  depth to root. */
        bool take_link_under_lock_if_spare(node_id_t owner, node_id_t resident) noexcept {
            bool const way_to_root = reaches_root(owner) && depth_to_root(owner) != 0 &&
                                     depth_to_root(resident) < depth_to_root(owner);
            if (way_to_root && !contains_(two_ways_to_root_k, owner)) return false;
            if (!take_link_for_swap_if_spare(owner, resident)) return false;
            if (way_to_root) release_(two_ways_to_root_k, owner);
            return true;
        }

        /** Files a link from @p owner to island entry @p member written since the sweep in the
         *  columns a merge keeps reach by: a member at no depth takes the one past the owner's,
         *  with the owner as its one shallower in-link, and an owner with no way to the root
         *  takes one past the member's depth to root, so no later re-selection drops the link.
         *  Only the task healing that island writes its entry's columns. */
        void record_heal_link(node_id_t owner, node_id_t member) noexcept {
            u8_t const owner_depth = depth_from_root(owner);
            if (depth_from_root(member) == unknown_depth_k && owner_depth < unknown_depth_k - 1) {
                atomic_ref<u8_t>(depths_from_root_[member])
                    .store(static_cast<u8_t>(owner_depth + 1), std::memory_order_relaxed);
                atomic_ref<u8_t>(shallower_in_links_[member]).store(1, std::memory_order_relaxed);
            }
            u8_t const member_depth_to_root = depth_to_root(member);
            if (depth_to_root(owner) == unknown_depth_k && member_depth_to_root < unknown_depth_k - 1)
                atomic_ref<u8_t>(depths_to_root_[owner])
                    .store(static_cast<u8_t>(member_depth_to_root + 1), std::memory_order_relaxed);
        }

        /** Entries the sweeps reached plus entries they marked a way to the root for, of twice the
         *  audited ones: what a caller's progress counts. */
        std::size_t swept() const noexcept { return swept_entries_; }

        /** Counts a link written into @p node_id since the sweep, saturating. */
        void count_in_link(node_id_t node_id) noexcept { static_cast<void>(count_in_link_(node_id)); }

        /** Whether @p node_id hangs by one thread: a single shallower in-link, or a way to the
         *  root through a single entry of the depth to root it was marked at. */
        bool hangs_by_one_link(node_id_t node_id) const noexcept {
            if (atomic_ref<u8_t const>(shallower_in_links_[node_id]).load(std::memory_order_relaxed) == 1) return true;
            return reaches_root(node_id) && depth_to_root(node_id) != 0 && !contains_(two_ways_to_root_k, node_id);
        }

        /** Entries outside the root's component, the islands' nodes, ranked by ID. */
        std::size_t islands() const noexcept { return island_nodes_.size(); }

        /** Whether island node @p rank names its strong component, being its first. */
        bool island_first(std::size_t rank) const noexcept { return island_components_[rank] == rank; }

        /** Island node @p rank 's @ref island_role_t bits, its component's two first among them. */
        u8_t island_roles(std::size_t rank) const noexcept { return island_roles_[rank]; }

        /** The entry standing for island node @p rank 's component: a member a link reaches
         *  or leaves the root's component through when one does, then the least linked, ties
         *  by a keyed draw. */
        node_id_t island_entry(std::size_t rank) const noexcept {
            return island_nodes_[island_entries_[island_components_[rank]]];
        }

        /** The entry of the root's component linking to the entry standing for island node
         *  @p rank 's component, or none. */
        node_id_t island_entry_parent(std::size_t rank) const noexcept {
            return island_parents_[island_entries_[island_components_[rank]]];
        }

        /** Whether @p node_id stands for an unreached or a stranded island. */
        bool component_entry(node_id_t node_id) const noexcept {
            if (in_component(node_id)) return false;
            std::span<node_id_t const> const nodes = to_span(island_nodes_);
            std::size_t const rank = static_cast<std::size_t>(lower_bound(nodes, node_id) - nodes.data());
            if (rank == nodes.size() || nodes[rank] != node_id) return false;
            node_id_t const component = island_components_[rank];
            return island_entries_[component] == rank &&
                   (island_roles_[component] & (unreached_island_k | stranded_island_k)) != 0;
        }

        /** The nodes of @p task outside the root's component, written into @p scratch in ID order;
         *  the filled prefix is returned, truncated to the scratch's length. */
        std::span<node_id_t const> islands(task_t const& task, node_ids_span_t scratch) const noexcept {
            std::size_t count = 0;
            for (node_id_t node_id = task.nodes.first; node_id < task.nodes.bound && count != scratch.size(); ++node_id)
                if (!in_component(node_id)) scratch[count++] = node_id;
            return {scratch.data(), count};
        }

        /** Whether @p node_id was reached - readable by any thread between depths. */
        bool reached(node_id_t node_id) const noexcept { return contains_(reached_k, node_id); }

        /** Whether @p node_id reaches the root - readable by any thread between depths. */
        bool reaches_root(node_id_t node_id) const noexcept { return contains_(reaches_root_k, node_id); }

        /** Whether @p node_id is in the root's strong component: reached from it, reaching it. */
        bool in_component(node_id_t node_id) const noexcept { return reached(node_id) && reaches_root(node_id); }

        /** Takes one of @p node_id 's reached in-links for a reconnection, answering whether
         *  another remains; a refusal leaves it untouched, so no row loses the last link it is
         *  reached by. */
        bool take_in_link_if_spare(node_id_t node_id) noexcept {
            return fu::atomic_fetch_sub_if_at_least(atomic_ref<u8_t>(in_links_[node_id]), u8_t {1}, u8_t {1},
                                                    std::memory_order_relaxed) > 1;
        }

        /**
         *  @brief The report: the entries of the root's component, the depths each direction took,
         *      the in-link counts, and the islands' components. @c internal_k while a pass is
         *      pending, since the counts would then be lower bounds.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The audit is bound to no index.
         *  - @c out_of_memory_k The islands' adjacency could not be placed.
         *  - @c overflow_k The islands hold more links than a node ID counts.
         */
        expected<reachability_t> finish() const noexcept {
            if (!index_) return {{}, status_code_t::uninitialized_k};
            if (step_ != step_t::done_k) return {{}, status_code_t::internal_k};
            if (failed(islands_status_)) return {{}, islands_status_};
            reachability_t result;
            result.audited_nodes = visible;
            std::size_t const words = words_();
            for (std::size_t word = 0; word != words_for(visible); ++word)
                result.reachable_nodes += static_cast<std::size_t>(
                    std::popcount(bits_[reached_k * words + word] & bits_[reaches_root_k * words + word]));
            result.depth_from_root = depth_from_root_;
            result.depth_to_root = depth_to_root_;
            result.unlinked = unlinked_;
            result.underlinked = underlinked_;
            for (std::size_t depth = most_populated_depth() + 1u; depth < depth_entries_.size(); ++depth)
                result.beyond_most_populated_depth += depth_entries_[depth];
            result.unreached_islands = unreached_islands_;
            result.stranded_islands = stranded_islands_;
            result.components = components_;
            return {result, status_code_t::success_k};
        }

      private:

        enum class step_t : u8_t { expand_k, recheck_k, count_links_k, done_k };

        /** The smallest depth to root among the links one read met, and how many links met it. */
        struct way_to_root_t {
            u8_t lowest = unknown_depth_k;
            std::size_t at_lowest = 0;

            void note(u8_t depth_to_root) noexcept {
                if (depth_to_root < lowest) lowest = depth_to_root, at_lowest = 1;
                else at_lowest += depth_to_root == lowest && depth_to_root != unknown_depth_k;
            }
        };

        /** Words per bitset. */
        std::size_t words_() const noexcept { return bits_.size() / bitsets_k; }

        /** The lanes of @p word below the watermark. */
        u1x64_t visible_lanes_(std::size_t word) const noexcept {
            std::size_t const lanes = min_of<std::size_t>(visible - word * lanes_k, lanes_k);
            return lanes == lanes_k ? ~u1x64_t {0} : (u1x64_t {1} << lanes) - 1;
        }

        u1x64_t word_(bitset_t bitset, std::size_t word) const noexcept {
            return atomic_ref<u1x64_t const>(bits_[bitset * words_() + word]).load(std::memory_order_relaxed);
        }

        bool contains_(bitset_t bitset, node_id_t node_id) const noexcept {
            return (word_(bitset, node_id / lanes_k) >> (node_id % lanes_k)) & 1u;
        }

        /** Sets @p node_id 's bit and answers whether it was already set. */
        bool insert_(bitset_t bitset, node_id_t node_id) noexcept {
            return atomic_ref<u1x64_t>(bits_[bitset * words_() + node_id / lanes_k])
                .fetch_test_and_set_bit(static_cast<unsigned>(node_id % lanes_k), std::memory_order_relaxed);
        }

        /** Clears @p node_id 's bit and answers whether it was set. */
        bool release_(bitset_t bitset, node_id_t node_id) noexcept {
            return atomic_ref<u1x64_t>(bits_[bitset * words_() + node_id / lanes_k])
                .fetch_test_and_reset_bit(static_cast<unsigned>(node_id % lanes_k), std::memory_order_relaxed);
        }

        static u8_t depth_of_(growing_byte_column_t const& column, node_id_t node_id) noexcept {
            return atomic_ref<u8_t const>(column[node_id]).load(std::memory_order_relaxed);
        }

        static void add_(std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).store_add(value, std::memory_order_relaxed);
        }

        /** Counts one more link into @p node_id from the depth before it, saturating. */
        void count_shallower_in_link_(node_id_t node_id) noexcept {
            atomic_ref<u8_t> const shallower_in_links {shallower_in_links_[node_id]};
            u8_t observed = shallower_in_links.load(std::memory_order_relaxed);
            while (observed != (std::numeric_limits<u8_t>::max)() &&
                   !shallower_in_links.compare_exchange_weak(observed, static_cast<u8_t>(observed + 1),
                                                             std::memory_order_relaxed, std::memory_order_relaxed)) {}
        }

        /** Counts one more reached link into @p node_id, saturating; answers how many it had. */
        u8_t count_in_link_(node_id_t node_id) noexcept {
            return fu::atomic_fetch_add_if_at_most(atomic_ref<u8_t>(in_links_[node_id]), u8_t {1},
                                                   (std::numeric_limits<u8_t>::max)(), std::memory_order_relaxed);
        }

        /** Calls @p visit with every link of @p node_id 's row below the watermark but its own,
         *  waiting out a row still being published; a forgotten row stays the sentinel and yields
         *  none, the caller's epoch voiding the audit. */
        template <typename visit_type_>
        void for_each_link_(node_id_t node_id, visit_type_&& visit) const noexcept {
            // Slot 0 is the lock word, acquired once and waited out; the tail slots are relaxed
            // reads, tolerating a re-linker's stale-but-valid mix.
            node_ids_view_t const row = index_->neighbors_for(chunk_schedule_t::locate(node_id));
            node_id_t first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            while (first_value == node_id_unknown_k && node_id < index_->size()) {
                wait_type_ {}(row.data(), node_id_unknown_k, candidates_count_t(0), fu::wait_capped_t {});
                first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            }
            if (first_value == node_id_unknown_k || first_value == node_id) return;
            for (std::size_t slot = 0; slot != row.size(); ++slot) {
                node_id_t const target = slot ? atomic_ref<node_id_t const>(row[slot]).load(std::memory_order_relaxed)
                                              : first_value;
                if (target == node_id_unknown_k) break;
                if (target < visible && target != node_id) visit(target);
            }
        }

        /** Clears what one root's sweep fills and marks @p root both ways at depth zero; with no
         *  root the audit is over. */
        void start_from_(node_id_t root) noexcept {
            if (!bits_.empty()) std::memset(bits_.data(), 0, bits_.size() * sizeof(u1x64_t));
            if (visible) {
                std::memset(depths_from_root_.data(), unknown_depth_k, visible);
                std::memset(depths_to_root_.data(), unknown_depth_k, visible);
                std::memset(shallower_in_links_.data(), 0, visible);
                std::memset(in_links_.data(), 0, visible);
            }
            depth_entries_.fill(0);
            for (island_ids_t* column : {&island_nodes_, &island_components_, &island_parents_, &island_entries_})
                column->clear();
            island_roles_.clear();
            largest_island_ = node_id_unknown_k;
            largest_island_entries_ = 0;
            unlinked_ = underlinked_ = unreached_islands_ = stranded_islands_ = components_ = 0;
            reached_in_depth_ = marked_in_depth_ = 0;
            swept_entries_ = root == node_id_unknown_k ? 0 : 2;
            depth_from_root_ = depth_to_root_ = depth_ = 0;
            islands_status_ = status_code_t::success_k;
            root_ = root;
            step_ = root == node_id_unknown_k ? step_t::done_k : step_t::expand_k;
            if (root == node_id_unknown_k) return;
            insert_(reached_k, root);
            insert_(reaches_root_k, root);
            depths_from_root_[root] = 0;
            depths_to_root_[root] = 0;
            depth_entries_[0] = 1;
        }

        /** Certifies @p node_id 's way to the root one past the lowest depth to root it links
         *  to, answering whether it links to any entry reaching back; two links at that depth
         *  spare one link for a reconnection to take. */
        bool mark_reaches_root_(node_id_t node_id, way_to_root_t const& way_to_root) noexcept {
            if (way_to_root.lowest == unknown_depth_k) return false;
            u8_t const depth_to_root = static_cast<u8_t>(
                min_of<unsigned>(way_to_root.lowest + 1u, unknown_depth_k - 1u));
            atomic_ref<u8_t>(depths_to_root_[node_id]).store(depth_to_root, std::memory_order_relaxed);
            insert_(reaches_root_k, node_id);
            if (way_to_root.at_lowest > 1 && way_to_root.lowest < depth_to_root) insert_(two_ways_to_root_k, node_id);
            return true;
        }

        /** Expands the entries of @p task at this depth: each link counts an in-link and the first
         *  reaches its target one depth deeper; the same read marks the entry's way to the root. */
        std::size_t expand_(task_t const& task) noexcept {
            // Past the deepest recorded level an entry marks its expansion in its parents byte,
            // which no link counts into there.
            bool const saturated = depth_ >= unknown_depth_k - 1u;
            u8_t const own_depth = static_cast<u8_t>(min_of<std::size_t>(depth_, unknown_depth_k - 1u));
            u8_t const child_depth = static_cast<u8_t>(min_of<std::size_t>(depth_ + 1, unknown_depth_k - 1u));
            std::size_t reached_count = 0, marked_count = 0;
            for (u1x64_t lanes = word_(reached_k, task.position); lanes; lanes &= lanes - 1) {
                node_id_t const node_id = static_cast<node_id_t>(task.position * lanes_k + std::countr_zero(lanes));
                if (depth_of_(depths_from_root_, node_id) != own_depth) continue;
                if (saturated && std::exchange(shallower_in_links_[node_id], u8_t {1})) continue;
                way_to_root_t way_to_root;
                for_each_link_(node_id, [&](node_id_t target) noexcept {
                    // Every reached in-link counts, saturating; the first one reaches the target.
                    if (!count_in_link_(target) && !insert_(reached_k, target)) {
                        atomic_ref<u8_t>(depths_from_root_[target]).store(child_depth, std::memory_order_relaxed);
                        ++reached_count;
                    }
                    // A depth still unknown is one another task reached this depth and is storing.
                    u8_t const target_depth = depth_of_(depths_from_root_, target);
                    if (child_depth != unknown_depth_k - 1 &&
                        (target_depth == child_depth || target_depth == unknown_depth_k))
                        count_shallower_in_link_(target);
                    way_to_root.note(depth_of_(depths_to_root_, target));
                });
                marked_count += depth_to_root(node_id) == unknown_depth_k && mark_reaches_root_(node_id, way_to_root);
            }
            add_(reached_in_depth_, reached_count);
            add_(marked_in_depth_, marked_count);
            return reached_count + marked_count;
        }

        /** Re-reads the entries of @p task reached but not yet marked, marking each that now
         *  links to an entry reaching back. */
        std::size_t recheck_reaches_root_(task_t const& task) noexcept {
            std::size_t marked_count = 0;
            for (u1x64_t lanes = word_(reached_k, task.position) & ~word_(reaches_root_k, task.position); lanes;
                 lanes &= lanes - 1) {
                node_id_t const node_id = static_cast<node_id_t>(task.position * lanes_k + std::countr_zero(lanes));
                way_to_root_t way_to_root;
                for_each_link_(
                    node_id, [&](node_id_t target) noexcept { way_to_root.note(depth_of_(depths_to_root_, target)); });
                marked_count += mark_reaches_root_(node_id, way_to_root);
            }
            add_(marked_in_depth_, marked_count);
            return marked_count;
        }

        /** Counts the entries of @p task no reached entry links to and those linked to by fewer
         *  entries than they link to, and files the least entry of the root's component linking to
         *  each island node as its parent. */
        std::size_t count_in_and_out_links_(task_t const& task) noexcept {
            std::span<node_id_t const> const islands = to_span(island_nodes_);
            std::size_t unlinked = 0, underlinked = 0;
            for (node_id_t node_id = task.nodes.first; node_id < task.nodes.bound; ++node_id) {
                bool const member = in_component(node_id);
                std::size_t out_links = 0;
                for_each_link_(node_id, [&](node_id_t target) noexcept {
                    ++out_links;
                    if (!member || in_component(target)) return;
                    std::size_t const rank = static_cast<std::size_t>(lower_bound(islands, target) - islands.data());
                    if (rank != islands.size() && islands[rank] == target)
                        fu::atomic_min(atomic_ref<node_id_t>(island_parents_[rank]), node_id,
                                       std::memory_order_relaxed);
                });
                unlinked += in_links_[node_id] == 0;
                underlinked += in_links_[node_id] < out_links;
            }
            add_(unlinked_, unlinked);
            add_(underlinked_, underlinked);
            return task.nodes.size();
        }

        /** Closes one root's sweep once a recheck marked nothing: gathers the islands and their
         *  strong components, then sweeps once more from the first entry of the largest island
         *  when it outgrows the drawn root's component, or turns to the count; an island
         *  adjacency that cannot be placed ends the audit with its status. */
        std::size_t choose_root_() noexcept {
            std::size_t const words = words_();
            std::size_t component = 0;
            for (std::size_t word = 0; word != words_for(visible); ++word)
                component += static_cast<std::size_t>(
                    std::popcount(bits_[reached_k * words + word] & bits_[reaches_root_k * words + word]));
            gather_islands_(visible - component);
            if (succeeded(islands_status_)) close_islands_();
            if (failed(islands_status_)) {
                step_ = step_t::done_k;
                return 0;
            }
            if (root_ == drawn_root_ && largest_island_entries_ > component) {
                drawn_root_ = node_id_unknown_k;
                start_from_(largest_island_);
                return 1;
            }
            step_ = step_t::count_links_k;
            return 1;
        }

        /** Gathers the @p islands entries outside the root's component in ascending order,
         *  marks those linking into the component, and marks the way to the root of those,
         *  which no sweep reached. */
        void gather_islands_(std::size_t islands) noexcept {
            if (failed(reserve_islands_(islands))) {
                islands_status_ = status_code_t::out_of_memory_k;
                return;
            }
            std::size_t const words = words_();
            for (std::size_t word = 0; word != words_for(visible); ++word)
                for (u1x64_t lanes = ~(bits_[reached_k * words + word] & bits_[reaches_root_k * words + word]) &
                                     visible_lanes_(word);
                     lanes; lanes &= lanes - 1)
                    island_nodes_.push_back_reserved(static_cast<node_id_t>(word * lanes_k + std::countr_zero(lanes)));
            island_parents_.extend_uninitialized(islands);
            for (node_id_t& parent : island_parents_) parent = node_id_unknown_k;
            island_roles_.extend_uninitialized(islands);
            if (islands) std::memset(island_roles_.data(), 0, islands);
            std::span<node_id_t const> const nodes = to_span(island_nodes_);
            for (std::size_t rank = 0; rank != islands; ++rank) {
                way_to_root_t way_to_root;
                for_each_link_(nodes[rank], [&](node_id_t target) noexcept {
                    if (!in_component(target)) return;
                    island_roles_[rank] |= links_to_component_k;
                    way_to_root.note(depth_to_root(target));
                });
                // A reached island never links into the component, or a recheck marked it.
                mark_reaches_root_(nodes[rank], way_to_root);
            }
        }

        /** Places the columns of @p islands island nodes; their links stay in the graph's rows. */
        status_t reserve_islands_(std::size_t islands) noexcept {
            if (failed(island_nodes_.reserve_exactly(islands)) || failed(island_components_.reserve_exactly(islands)) ||
                failed(island_parents_.reserve_exactly(islands)) || failed(island_entries_.reserve_exactly(islands)) ||
                failed(island_roles_.reserve_exactly(islands)) || failed(island_walk_.reserve_exactly(5 * islands)))
                return status_code_t::out_of_memory_k;
            return status_code_t::success_k;
        }

        /** The rank of island node @p node_id among the islands' nodes. */
        std::size_t island_rank_(node_id_t node_id) const noexcept {
            std::span<node_id_t const> const nodes = to_span(island_nodes_);
            return static_cast<std::size_t>(lower_bound(nodes, node_id) - nodes.data());
        }

        /** Calls @p visit with the rank of every island node that island node @p rank links to. */
        template <typename visit_type_>
        void for_each_island_link_(std::size_t rank, visit_type_&& visit) const noexcept {
            for_each_link_(island_nodes_[rank], [&](node_id_t target) noexcept {
                if (!in_component(target)) visit(island_rank_(target));
            });
        }

        /** The rank of the next island node that island node @p rank links to from slot @p cursor
         *  on, moving the cursor past it, or the islands' count once the row ends; Tarjan's walk
         *  resumes rows this way rather than keep a copy of the islands' links. */
        std::size_t next_island_link_(std::size_t rank, node_id_t& cursor) const noexcept {
            node_id_t const node_id = island_nodes_[rank];
            node_ids_view_t const row = index_->neighbors_for(chunk_schedule_t::locate(node_id));
            while (cursor < row.size()) {
                node_id_t const target = atomic_ref<node_id_t const>(row[cursor++]).load(std::memory_order_relaxed);
                // A sentinel ends the row: past its last link, or in slot zero a row held
                // by a writer.
                if (target == node_id_unknown_k) break;
                if (target < visible && target != node_id && !in_component(target)) return island_rank_(target);
            }
            cursor = static_cast<node_id_t>(row.size());
            return island_nodes_.size();
        }

        /** Pops the strong component Tarjan's walk closed at @p rank off @p stack, names it by
         *  the rank of its first node, marks its nodes done in @p lowest, and marks the way to
         *  the root of its nodes, its successors' being final. */
        void close_component_(node_id_t* stack, std::size_t& stacked, std::size_t rank, node_id_t* lowest) noexcept {
            std::size_t from = stacked;
            while (stack[--from] != rank) {}
            std::span<node_id_t const> const members {stack + from, stacked - from};
            stacked = from;
            node_id_t first = members[0];
            for (node_id_t const member : members) first = min_of(first, member);
            if (members.size() > largest_island_entries_)
                largest_island_entries_ = members.size(), largest_island_ = island_nodes_[first];
            for (node_id_t const member : members) {
                island_components_[member] = first;
                lowest[member] = node_id_unknown_k;
            }
            for (bool marked = true; marked;) {
                marked = false;
                for (node_id_t const member : members) {
                    if (depth_to_root(island_nodes_[member]) != unknown_depth_k) continue;
                    way_to_root_t way_to_root;
                    for_each_island_link_(member, [&](std::size_t next) noexcept {
                        way_to_root.note(depth_to_root(island_nodes_[next]));
                    });
                    marked |= mark_reaches_root_(island_nodes_[member], way_to_root);
                }
            }
        }

        /** Finds the islands' strong components by Tarjan's walk over their links, on one thread
         *  and without recursion, noting the largest. */
        void close_islands_() noexcept {
            std::size_t const islands = island_nodes_.size();
            if (!islands) return;
            if (failed(island_walk_.resize(5 * islands)) || failed(island_components_.resize(islands))) {
                islands_status_ = status_code_t::out_of_memory_k;
                return;
            }
            // Visit order from one, zero unvisited; the least order each node reaches on the
            // stack, unknown once its component closed; the stack; and the walk's path beside
            // the slot each task on it reads next.
            node_id_t* const order = island_walk_.data();
            node_id_t* const lowest = order + islands;
            node_id_t* const stack = lowest + islands;
            node_id_t* const path = stack + islands;
            node_id_t* const cursors = path + islands;
            std::size_t visits = 0, stacked = 0, depth = 0;
            auto visit = [&](std::size_t rank) noexcept {
                order[rank] = lowest[rank] = static_cast<node_id_t>(++visits);
                stack[stacked++] = path[depth] = static_cast<node_id_t>(rank);
                cursors[depth++] = 0;
            };
            for (std::size_t start = 0; start != islands; ++start) {
                if (order[start]) continue;
                visit(start);
                while (depth) {
                    std::size_t const rank = path[depth - 1];
                    if (std::size_t const next = next_island_link_(rank, cursors[depth - 1]); next != islands) {
                        if (!order[next]) visit(next);
                        else if (lowest[next] != node_id_unknown_k) lowest[rank] = min_of(lowest[rank], order[next]);
                        continue;
                    }
                    --depth;
                    if (lowest[rank] == order[rank]) close_component_(stack, stacked, rank, lowest);
                    if (depth) lowest[path[depth - 1]] = min_of(lowest[path[depth - 1]], lowest[rank]);
                }
            }
        }

        /** Marks the unreached and stranded islands, picks the entry standing for each, and counts
         *  the weak components, the root's among them, by union-find. */
        void label_islands_() noexcept {
            std::size_t const islands = island_nodes_.size();
            if (failed(islands_status_)) return;
            components_ = 1;
            if (!islands) return;
            if (failed(island_entries_.resize(islands))) {
                islands_status_ = status_code_t::out_of_memory_k;
                return;
            }
            constexpr u8_t component_roles_k = unreached_island_k | stranded_island_k;
            for (std::size_t rank = 0; rank != islands; ++rank)
                if (island_components_[rank] == rank) island_roles_[rank] |= component_roles_k;
            for (std::size_t rank = 0; rank != islands; ++rank) {
                node_id_t const component = island_components_[rank];
                if (island_parents_[rank] != node_id_unknown_k) {
                    island_roles_[rank] |= linked_from_component_k;
                    island_roles_[component] &= static_cast<u8_t>(~unreached_island_k);
                }
                if (island_roles_[rank] & links_to_component_k)
                    island_roles_[component] &= static_cast<u8_t>(~stranded_island_k);
                for_each_island_link_(rank, [&](std::size_t linked) noexcept {
                    node_id_t const next = island_components_[linked];
                    if (next == component) return;
                    island_roles_[component] &= static_cast<u8_t>(~stranded_island_k);
                    island_roles_[next] &= static_cast<u8_t>(~unreached_island_k);
                });
            }
            for (std::size_t rank = 0; rank != islands; ++rank) {
                node_id_t const component = island_components_[rank];
                u8_t const roles = island_roles_[component] & component_roles_k;
                if (component != rank) island_roles_[rank] |= roles;
                else {
                    unreached_islands_ += (roles & unreached_island_k) != 0;
                    stranded_islands_ += (roles & stranded_island_k) != 0;
                }
            }

            // A member a link crosses through stands first, as the heal links through it; then the
            // fewest in-links, ties by a keyed draw, so the ID order never decides.
            auto stands_before = [&](std::size_t rank, std::size_t other) noexcept {
                auto crosses = [&](std::size_t member) noexcept {
                    return island_parents_[member] != node_id_unknown_k ||
                           (island_roles_[member] & links_to_component_k) != 0;
                };
                if (crosses(rank) != crosses(other)) return crosses(rank);
                node_id_t const node = island_nodes_[rank], other_node = island_nodes_[other];
                if (in_links_[node] != in_links_[other_node]) return in_links_[node] < in_links_[other_node];
                return mix_bits(node) < mix_bits(other_node);
            };
            for (std::size_t rank = 0; rank != islands; ++rank)
                if (island_components_[rank] == rank) island_entries_[rank] = static_cast<node_id_t>(rank);
            for (std::size_t rank = 0; rank != islands; ++rank) {
                node_id_t& entry = island_entries_[island_components_[rank]];
                if (stands_before(rank, entry)) entry = static_cast<node_id_t>(rank);
            }

            // The walk's columns again, as union-find parents; the last slot stands for
            // the root's one.
            node_id_t* const weak = island_walk_.data();
            for (std::size_t rank = 0; rank <= islands; ++rank) weak[rank] = static_cast<node_id_t>(rank);
            auto find = [&](std::size_t rank) noexcept {
                while (weak[rank] != rank) rank = weak[rank] = weak[weak[rank]];
                return rank;
            };
            auto unite = [&](std::size_t first, std::size_t second) noexcept {
                first = find(first), second = find(second);
                if (first != second) weak[max_of(first, second)] = static_cast<node_id_t>(min_of(first, second));
            };
            for (std::size_t rank = 0; rank != islands; ++rank) {
                if (island_roles_[rank] & (links_to_component_k | linked_from_component_k)) unite(rank, islands);
                for_each_island_link_(rank, [&](std::size_t linked) noexcept { unite(rank, linked); });
            }
            components_ = 0;
            for (std::size_t rank = 0; rank <= islands; ++rank) components_ += weak[rank] == rank;
        }

        /** The audited index; null on a moved-from plan. */
        ram_index const* index_ = nullptr;

        /** Every bitset, one bit per published node each, laid out as @ref bitset_t orders them. */
        growing_bitset_t bits_;

        /** Per published node, its depth from the root. */
        growing_byte_column_t depths_from_root_;

        /** Per published node, one past the depth to root of the entry that marked its way
         *  to the root. */
        growing_byte_column_t depths_to_root_;

        /** Per published node, its links from the depth before it, saturating. */
        growing_byte_column_t shallower_in_links_;

        /** Per published node, the reached entries linking to it, saturating. */
        growing_byte_column_t in_links_;

        /** Entries each depth reached; the fullest is the depth the deep stratum draws past. */
        std::array<std::size_t, unknown_depth_k> depth_entries_ {};

        /** The nodes outside the root's component, ascending. */
        island_ids_t island_nodes_;

        /** Per island node, its strong component: the rank of its first node. */
        island_ids_t island_components_;

        /** Per island node, the least entry of the root's component linking to it, or none. */
        island_ids_t island_parents_;

        /** Per island node naming its strong component, the rank of the member standing for it. */
        island_ids_t island_entries_;

        /** Per island node, its @ref island_role_t bits. */
        growing_byte_column_t island_roles_;

        /** Tarjan's walk over the islands, then their union-find. */
        island_ids_t island_walk_;

        /** The node the sweeps start from, a seeded draw or the first entry of a larger island. */
        node_id_t root_ = node_id_unknown_k;

        /** The root this audit drew, which may give way once to a larger island; none for a root
         *  kept from an earlier audit, which never does. */
        node_id_t drawn_root_ = node_id_unknown_k;

        /** The first entry of the largest island the last gather found, and its entries. */
        node_id_t largest_island_ = node_id_unknown_k;
        std::size_t largest_island_entries_ = 0;

        /** Nodes the current depth reached and marked so far, summed through @c atomic_ref. */
        std::size_t reached_in_depth_ = 0;
        std::size_t marked_in_depth_ = 0;

        /** Entries the sweeps reached plus entries they marked a way to the root for, so far. */
        std::size_t swept_entries_ = 0;

        /** Entries no reached entry links to, and entries linked to by fewer than they link to. */
        std::size_t unlinked_ = 0;
        std::size_t underlinked_ = 0;

        /** Island components nothing outside links into, and those with no link out. */
        std::size_t unreached_islands_ = 0;
        std::size_t stranded_islands_ = 0;

        /** Components when direction is ignored, the root's among them. */
        std::size_t components_ = 0;

        /** Depths of the kept root's sweep that reached a node. */
        std::size_t depth_from_root_ = 0;

        /** Rechecks of the kept root's sweep that marked a node. */
        std::size_t depth_to_root_ = 0;

        /** The depth the next forward sweep expands. */
        std::size_t depth_ = 0;

        /** The pass the next tasks run. */
        step_t step_ = step_t::done_k;

        /** Whether the islands' adjacency could be placed. */
        status_t islands_status_ = status_code_t::success_k;
    };

    /** Opens a reachability audit over the published nodes both ways from one drawn root, whose
     *  strong component it measures, and the islands outside it. The caller sweeps every task per
     *  pass until @c advance_depth() answers zero, then reads @c finish(). Its columns come from
     *  the index's own allocator. */
    expected<health_plan_t> plan_health() const noexcept {
        return plan_audit_(atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire));
    }

    /** Bytes an audit over @p rows nodes takes from the index's allocator, every bitset in one
     *  slice, the four byte columns per row in one each, and room for an island node per bitset
     *  word, ten times the share of islands measured on billion-entry graphs, in six slices, each
     *  padded to the default alignment: what a caller sizing that allocator sets aside for
     *  @ref plan_health. More islands than that grow past it, and a refused growth is the audit's
     *  status, never a report of none. */
    static std::size_t audit_memory_usage(std::size_t rows) noexcept {
        std::size_t const words = health_plan_t::words_for(rows);
        std::size_t const island_bytes = 9 * sizeof(node_id_t) + sizeof(u8_t);
        return health_plan_t::bitsets_k * words * sizeof(u1x64_t) + 4 * rows + words * island_bytes +
               11 * default_alignment_k;
    }

    struct reshard_plan_t;

    /**
     *  @brief One refinement round in flight, from its opening health check to its closing one.
     *
     *  A plan from @ref plan_refine runs one round: the opening health check, the islands it named
     *  healed without a distance, a seeded draw of about the square root of the neighborhoods it
     *  needs as hubs, a gather streaming every stored entry past them into the neighborhood of its
     *  nearest hub, and again inside every neighborhood over its capacity, a restitch of each
     *  neighborhood in one core's cache, every entry keeping the diverse links among its own and
     *  its nearest neighborhood-mates and no others, and the closing health check from the same
     *  root, which decides the verdict. When the heal left islands with no link to or from the
     *  root's component, such as a moved part, the round heals and re-stitches again from that
     *  check, while each heal leaves fewer such islands than the one before it; no phase walks.
     *
     *  Tasks of the current phase are disjoint ranges, so any number of threads may @ref run them
     *  at once; @ref advance is the single-caller barrier between phases and answers whether
     *  another phase follows. Planning reserves everything the plan keeps and @ref reserve grows a
     *  context to what a task touches, so neither @ref run, @ref advance nor @ref finish allocates.
     *  Every entry the opening health check reached both ways stays reached both ways throughout,
     *  so searches run throughout and a round stopped between tasks leaves the graph correct, as
     *  long as no insertion overlaps it.
     */
    struct refine_plan_t {
        using task_t = typename health_plan_t::task_t;
        using part_bases_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<node_id_t>;
        using part_bases_t = growing_buffer<node_id_t, part_bases_allocator_t>;
        using bitset_allocator_t = typename health_plan_t::bitset_allocator_t;
        using growing_bitset_t = typename health_plan_t::growing_bitset_t;
        using bytes_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<byte_t>;
        using norms_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<vector_norms_t>;
        using neighborhoods_allocator_t =
            typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<u16_t>;

        /** The health check the restitch reads its depths from and the heal sweeps. */
        health_plan_t health;

        /** An unbound plan over @p health, its buffers placed by @p allocator - what a refused plan
         *  answers with, and what planning fills. */
        refine_plan_t(health_plan_t&& health, part_bases_allocator_t const& allocator) noexcept
            : health(std::move(health)), changed_ {bitset_allocator_t {allocator}}, hub_nodes_(allocator),
              hub_vectors_ {bytes_allocator_t {allocator}}, hub_norms_ {norms_allocator_t {allocator}},
              entry_neighborhoods_ {neighborhoods_allocator_t {allocator}}, neighborhood_entries_(allocator),
              neighborhood_offsets_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : health(std::move(other.health)), index_(std::exchange(other.index_, nullptr)),
              changed_(std::move(other.changed_)), phase_(other.phase_), swept_after_(other.swept_after_),
              islands_waiting_(other.islands_waiting_), waiting_before_(other.waiting_before_), tally_(other.tally_),
              status_(other.status_), cache_bytes_(other.cache_bytes_), hubs_per_draw_(other.hubs_per_draw_),
              neighborhood_capacity_(other.neighborhood_capacity_), hub_nodes_(std::move(other.hub_nodes_)),
              hub_vectors_(std::move(other.hub_vectors_)), hub_norms_(std::move(other.hub_norms_)),
              entry_neighborhoods_(std::move(other.entry_neighborhoods_)),
              neighborhood_entries_(std::move(other.neighborhood_entries_)),
              neighborhood_offsets_(std::move(other.neighborhood_offsets_)), round_seed_(other.round_seed_) {}
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** Task @p position of the current phase, an empty one past the end: a health check's
         *  bitset word, the islands healed, the entries of one gather, or one neighborhood. */
        task_t operator[](std::size_t position) const noexcept {
            if (phase_ == phase_t::health_k) return health[position];
            task_t task;
            if (position >= tasks().size()) return task;
            task.position = position;
            iota_view<node_id_t> const range = phase_nodes_();
            std::size_t const width = phase_width_();
            std::size_t const first = range.first + (position * width);
            std::size_t const bound = min_of(first + width, static_cast<std::size_t>(range.bound));
            task.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return task;
        }

        /** The current phase's tasks, positionally; none once the plan is over. */
        indexed_range<refine_plan_t> tasks() const noexcept {
            if (!index_ || phase_ == phase_t::done_k) return {this, 0u};
            if (phase_ == phase_t::health_k) return {this, health.tasks().size()};
            return {this, divide_round_up<std::size_t>(phase_nodes_().size(), phase_width_())};
        }

        /**
         *  @brief Runs @p task of the current phase on the calling thread - safe for disjoint tasks
         *      of one phase, and never failing on a bound plan once @ref reserve grew @p context,
         *      since planning claimed and reserved everything a task writes.
         *
         *  A heal task links the round's islands, a gather task files a neighborhood capacity of
         *  entries under their nearest hubs, and a restitch task re-stitches one neighborhood in
         *  one core's cache.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         */
        status_t run(task_t const& task, traversal_context_t& context) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            switch (phase_) {
            case phase_t::health_k: health.sweep(task); break;
            case phase_t::heal_k: heal_(task); break;
            case phase_t::gather_k: gather_neighborhoods_(task, context); break;
            case phase_t::restitch_k: restitch_neighborhoods_(task, context); break;
            default: break;
            }
            return status_code_t::success_k;
        }

        /** The barrier between phases, one caller after every task of the current one ran: packs
         *  the hubs, sorts the entries by neighborhood, or picks the next phase, and answers
         *  whether one follows. */
        bool advance() noexcept {
            if (!index_ || failed(status_)) return false;
            if (phase_ == phase_t::health_k) {
                if (!health.advance_depth()) converge_();
            }
            else advance_round_();
            if (failed(status_)) return false;
            return phase_ != phase_t::done_k;
        }

        /** Skips what is left of the round's phase in hand, at a barrier: an opening health check,
         *  or a heal not yet run, ends the round with nothing written; a hubs draw, a gather or a
         *  restitch sends the round to the closing health check, the heal's links and the
         *  neighborhoods already re-stitched standing; the closing health check runs on, and the
         *  verdict reads stopped. A no-op once the round is over. */
        void stop() noexcept {
            if (rounding()) stop_round_();
        }

        /** The first failure the plan met, which ended it; success otherwise. */
        status_t status() const noexcept { return status_; }

        /**
         *  @brief Grows @p context to what a gather or restitch task touches, which a driver calls
         *      before a round's first task and after every @ref next_round.
         *
         *  That is a neighborhood capacity of gathered entries and a gather's hubs with the packs
         *  of either, their norms and thresholds, every gathered entry's nearest
         *  neighborhood-mates, and the merge of one entry's links with them. An unbound plan needs
         *  none of it.
         *
         *  @par Statuses:
         *  - @c out_of_memory_k @p context cannot grow to a round's scratch.
         */
        status_t reserve(traversal_context_t& context) const noexcept {
            if (!index_ || !neighborhood_capacity_) return status_code_t::success_k;
            std::size_t const connectivity = index_->connectivity(), hubs = hubs_for_(health.visible);
            std::size_t const packed = scan_scorer_t::packs_bytes(max_of(hubs, neighborhood_capacity_),
                                                                  index_->dimensions());
            if (failed(context.reserve_scan(max_of(hubs, neighborhood_capacity_), scan_scorer_t::step_rows_k,
                                            scan_scorer_t::dots_columns_k)) ||
                failed(context.reserve_neighborhoods(neighborhood_capacity_, connectivity)) ||
                failed(context.reserve_queries(gathered_packs_offset_() + packed)) ||
                failed(context.results.reserve(2 * connectivity)))
                return status_code_t::out_of_memory_k;
            return status_code_t::success_k;
        }

        /** Whether the plan is inside a round, whose every barrier a caller's progress hears. */
        bool rounding() const noexcept { return phase_ >= phase_t::health_k && phase_ <= phase_t::restitch_k; }

        /** The phase the plan is in. */
        phase_t phase() const noexcept { return phase_; }

        /** The position of the round's phase in hand among its six, from the opening health check
         *  to the closing one; zero outside a round's phases. */
        std::size_t ordinal() const noexcept { return round_ordinal(phase_, swept_after_); }

        /** Units of the round's phase in hand done so far, at a barrier before it advances:
         *  entries swept and marked by an audit, and otherwise all of the phase's, since each of
         *  its passes runs to its end. */
        std::size_t completed() const noexcept {
            if (phase_ == phase_t::health_k) return health.swept();
            return total();
        }

        /** Units of the round's phase in hand in all, which @ref completed counts toward: islands
         *  healed, hubs drawn, entries one gather pass files, or entries re-stitched. */
        std::size_t total() const noexcept {
            switch (phase_) {
            case phase_t::health_k: return 2 * std::size_t {health.visible};
            case phase_t::heal_k: return health.islands();
            case phase_t::hubs_k: return first_hubs_(health.visible);
            case phase_t::gather_k: return gathered_entries_();
            case phase_t::restitch_k: return health.visible;
            default: return 0;
            }
        }

        /** Opens the next round once one ended, from the root the closing health check kept:
         *  the count and the changed marks start over, the health check and the round's columns
         *  grow to the entries published since, and the opening health check sweeps again;
         *  columns that cannot grow fail the plan. A no-op mid-round. Contexts then need
         *  @ref reserve again. */
        void next_round() noexcept {
            if (phase_ != phase_t::done_k || failed(status_)) return;
            tally_ = {};
            // A refused widening re-checks the entries it held, and the round
            // re-stitches those alone.
            static_cast<void>(health.widen(static_cast<node_id_t>(index_->size())));
            if (failed(reserve_round_(cache_bytes_))) return status_ = status_code_t::out_of_memory_k, void();
            open_health_(phase_t::none_k);
        }

        /**
         *  @brief The round so far: complete once @ref advance answered false, else cancelled.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         *  - @c cancelled_k The round stopped before @ref advance answered false, or was stopped.
         *  - @c out_of_memory_k A health check could not place its islands' adjacency.
         */
        expected<refine_report_t> finish() const noexcept {
            if (!index_) return {{}, status_code_t::uninitialized_k};
            if (failed(status_)) return {tally_, status_};
            bool const done = phase_ == phase_t::done_k && tally_.verdict != refine_verdict_t::stopped_k;
            return {tally_, done ? status_code_t::success_k : status_code_t::cancelled_k};
        }

      private:
        friend ram_index;
        friend reshard_plan_t;

        /** What a @ref add_or_swap_link_ found. */
        enum class taken_t : u8_t { written_k, present_k, refused_k };

        /** Opens the health check following @p phase from the root the last one kept: the opening
         *  one after none, or the closing one after the restitch. */
        void open_health_(phase_t phase) noexcept {
            health.restart_sweep();
            swept_after_ = phase;
            phase_ = phase_t::health_k;
        }

        static void add_(std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).store_add(value, std::memory_order_relaxed);
        }

        /** The rows the current phase tasks over: the round's islands, the entries it gathers into
         *  neighborhoods, or its neighborhoods; none while it draws hubs, since the barrier closing
         *  that phase draws them. */
        iota_view<node_id_t> phase_nodes_() const noexcept {
            if (phase_ == phase_t::heal_k) return {0, static_cast<node_id_t>(health.islands())};
            if (phase_ == phase_t::gather_k) return {0, health.visible};
            if (phase_ == phase_t::restitch_k) return {0, static_cast<node_id_t>(neighborhoods_())};
            return {};
        }

        /** Rows one task of the current phase spans: every island, a neighborhood capacity of
         *  entries gathered, or one neighborhood. */
        std::size_t phase_width_() const noexcept {
            // One heal task: islands sharing an owner row then link in one order.
            if (phase_ == phase_t::heal_k) return max_of<std::size_t>(health.islands(), 1);
            if (phase_ == phase_t::gather_k) return neighborhood_capacity_;
            return 1;
        }

        /** Marks row @p node_id counted as changed, answering whether it was already. */
        bool count_changed_(node_id_t node_id) noexcept {
            return atomic_ref<u1x64_t>(changed_[node_id / health_plan_t::lanes_k])
                .fetch_test_and_set_bit(static_cast<unsigned>(node_id % health_plan_t::lanes_k),
                                        std::memory_order_relaxed);
        }

        /** Takes in a converged health check: the opening one opens the round; the closing one
         *  heals again from what it measured while each heal leaves fewer islands waiting than
         *  the one before it, and otherwise decides the verdict. */
        void converge_() noexcept {
            auto const [reachability, status] = health.finish();
            // An adjacency the audit could not place fails the plan, never reads as no islands.
            if (failed(status)) {
                status_ = status;
                phase_ = phase_t::done_k;
                return;
            }
            std::size_t const islands = reachability.audited_nodes - reachability.reachable_nodes;
            tally_.reachability = reachability;
            if (swept_after_ == phase_t::none_k) return open_round_(reachability);
            // The restitch gave the islands the heal could not link their links, which a heal from
            // this check follows without a distance, as a moved part nothing links into needs.
            bool const stopped = tally_.verdict == refine_verdict_t::stopped_k;
            if (heals_again(islands, islands_waiting_, waiting_before_) && !stopped) return heal_from_(reachability);
            tally_.verdict = concluded_verdict(tally_);
            phase_ = phase_t::done_k;
        }

        /** Folds one task's spend into the count. */
        void add_spent_(traversal_stats_t const& spent) noexcept {
            add_(tally_.traversal_stats.computed_distances, spent.computed_distances);
            add_(tally_.traversal_stats.visited_members, spent.visited_members);
            add_(tally_.traversal_stats.restarts, spent.restarts);
            add_(tally_.traversal_stats.saturated_walks, spent.saturated_walks);
            add_(tally_.traversal_stats.computed_distances_in_refines, spent.computed_distances_in_refines);
            add_(tally_.traversal_stats.computed_distances_in_reverse_refines,
                 spent.computed_distances_in_reverse_refines);
        }

        /** A round's phase after the barrier closing @c phase_: the hubs pack, the entries sort by
         *  neighborhood, or the next phase opens. */
        void advance_round_() noexcept {
            switch (phase_) {
            case phase_t::heal_k: phase_ = phase_t::hubs_k; break;
            case phase_t::hubs_k:
                draw_hubs_();
                phase_ = phase_t::gather_k;
                break;
            case phase_t::gather_k:
                // Neighborhoods over capacity gather again among their own entries.
                order_neighborhoods_();
                if (tally_.widest_neighborhood <= neighborhood_capacity_) phase_ = phase_t::restitch_k;
                break;
            case phase_t::restitch_k: open_health_(phase_t::restitch_k); break;
            default: break;
            }
        }

        /** Skips what is left of the round's phase in hand, as @ref stop describes. */
        void stop_round_() noexcept {
            tally_.verdict = refine_verdict_t::stopped_k;
            switch (phase_) {
            case phase_t::health_k:
                if (swept_after_ == phase_t::none_k) phase_ = phase_t::done_k;
                break;
            case phase_t::heal_k: phase_ = phase_t::done_k; break;
            case phase_t::hubs_k:
            case phase_t::gather_k:
            case phase_t::restitch_k: open_health_(phase_t::restitch_k); break;
            default: break;
            }
        }

        /** Sizes everything a round keeps across its barriers for the audited entries, under
         *  @p cache_bytes of cache a core: one pack of hubs, per entry its neighborhood and its
         *  place in the order by neighborhood, and the starts every split may write. */
        status_t reserve_round_(std::size_t cache_bytes) noexcept {
            std::size_t const visible = health.visible;
            cache_bytes_ = cache_bytes;
            hubs_per_draw_ = visible ? index_->block_columns_(visible, cache_bytes) : 0;
            std::size_t const block_rows = index_->block_rows_(hubs_per_draw_, cache_bytes);
            neighborhood_capacity_ = block_rows ? block_rows : chunk_schedule_t::nodes_in(0);
            std::size_t const hubs = hubs_for_(visible);
            if (failed(changed_.resize(health_plan_t::words_for(visible))) ||
                failed(hub_nodes_.reserve_exactly(hubs)) || failed(hub_vectors_.reserve_exactly(hubs_bytes_(hubs))) ||
                failed(hub_norms_.reserve_exactly(hubs)) || failed(entry_neighborhoods_.resize(visible)) ||
                failed(neighborhood_entries_.resize(visible)) ||
                failed(neighborhood_offsets_.reserve_exactly(split_room(visible, hubs_per_draw_))))
                return status_code_t::out_of_memory_k;
            return status_code_t::success_k;
        }

        /** Hubs one gather of @p entries draws: as many as make a neighborhood half its capacity on
         *  average, at most one pack, and no more than @c entry_neighborhoods_ can number. */
        std::size_t hubs_for_(std::size_t entries) const noexcept {
            return round_hubs(entries, hubs_per_draw_, neighborhood_capacity_);
        }

        /** Hubs the round's first gather over @p entries draws, as @ref round_first_hubs counts. */
        std::size_t first_hubs_(std::size_t entries) const noexcept {
            return round_first_hubs(entries, hubs_per_draw_, neighborhood_capacity_);
        }

        /** Bytes the hubs of the first gather take, one padded row each, then their packs on
         *  a fresh line. */
        std::size_t hubs_bytes_(std::size_t hubs) const noexcept {
            return round_up_to_multiple<std::size_t>(hubs * index_->padded_row_bytes_(), default_alignment_k) +
                   scan_scorer_t::packs_bytes(hubs, index_->dimensions());
        }

        /** Where a task gathers the hubs of a neighborhood it gathers again, past one
         *  neighborhood capacity of entries, and where the packs of either start, past the most
         *  hubs a gather draws. */
        std::size_t gathered_hubs_offset_() const noexcept {
            return round_up_to_multiple<std::size_t>(neighborhood_capacity_ * index_->padded_row_bytes_(),
                                                     default_alignment_k);
        }
        std::size_t gathered_packs_offset_() const noexcept {
            return gathered_hubs_offset_() +
                   round_up_to_multiple<std::size_t>(hubs_for_(health.visible) * index_->padded_row_bytes_(),
                                                     default_alignment_k);
        }

        /** Takes in the opening health check and opens the heal from it. */
        void open_round_(reachability_t const& opening) noexcept {
            tally_.opening = opening;
            tally_.islands_found = opening.audited_nodes - opening.reachable_nodes;
            islands_waiting_ = (std::numeric_limits<std::size_t>::max)();
            heal_from_(opening);
        }

        /** Seeds the draws from the health check @p measured and opens the heal of the islands it
         *  named, counting anew the ones it leaves waiting. */
        void heal_from_(reachability_t const& measured) noexcept {
            round_seed_ = round_seed(measured);
            waiting_before_ = std::exchange(islands_waiting_, 0);
            phase_ = health.visible ? phase_t::heal_k : phase_t::done_k;
        }

        /** Neighborhoods the order by neighborhood holds. */
        std::size_t neighborhoods_() const noexcept {
            return neighborhood_offsets_.size() ? neighborhood_offsets_.size() - 1 : 0;
        }

        /** Entries in neighborhood @p neighborhood. */
        std::size_t neighborhood_width_(std::size_t neighborhood) const noexcept {
            return neighborhood_offsets_[neighborhood + 1] - neighborhood_offsets_[neighborhood];
        }

        /** Entries the gather pass in hand files: every audited one, then those of the
         *  neighborhoods over capacity. */
        std::size_t gathered_entries_() const noexcept {
            if (neighborhood_offsets_.empty()) return health.visible;
            std::size_t entries = 0;
            for (std::size_t neighborhood = 0; neighborhood != neighborhoods_(); ++neighborhood)
                if (neighborhood_width_(neighborhood) > neighborhood_capacity_)
                    entries += neighborhood_width_(neighborhood);
            return entries;
        }

        /** @p count gathered padded rows from @p bytes as the scorer reads them, at
         *  the logical width. */
        vectors_view_t gathered_view_(byte_t const* bytes, std::size_t count) const noexcept {
            vectors_view_t const padded {reinterpret_cast<scalar_t const*>(bytes), count,
                                         index_->vectors_.columns() * nk::dimensions_per_value<scalar_t>()};
            nk::shape_storage_<2> const logical {{count, index_->dimensions()},
                                                 {padded.stride_bytes(0), padded.stride_bytes(1)},
                                                 2};
            return {padded.byte_data(), logical};
        }

        /** Copies the stored vector of @p node_id into padded row @p row of @p rows, answering its
         *  norms. */
        vector_norms_t gather_(node_id_t node_id, byte_t* rows, std::size_t row) const noexcept {
            location_t const location = chunk_schedule_t::locate(node_id);
            std::size_t const row_bytes = index_->padded_row_bytes_();
            std::memcpy(rows + row * row_bytes, index_->vectors_[location], row_bytes);
            return index_->norm_for(location);
        }

        /** @p norms as a scan reads row norms: none when the rows keep none. */
        std::span<vector_norms_t const> row_norms_(std::span<vector_norms_t const> norms) const noexcept {
            return index_->stores_norms() ? norms : std::span<vector_norms_t const> {};
        }

        /** The first @p count of @p context 's thresholds, none of which a distance reaches. */
        static std::span<distance_t> unreached_thresholds_(traversal_context_t& context, std::size_t count) noexcept {
            std::span<distance_t> const thresholds {context.thresholds.data(), count};
            for (distance_t& threshold : thresholds) threshold = farthest_.distance;
            return thresholds;
        }

        /** The first @p count of @p context 's neighborhood-mates, each an entry's nearest
         *  hub so far. */
        static std::span<scored_node_t> nearest_hubs_(traversal_context_t& context, std::size_t count) noexcept {
            std::span<scored_node_t> const nearest {context.neighborhood_mates.data(), count};
            for (scored_node_t& kept : nearest) kept = farthest_;
            return nearest;
        }

        /** Keeps hub @p hub in @p nearest when it lies nearer, the first hub among equals, and
         *  answers a threshold no distance reaches, so every hub's distance is heard. */
        static distance_t keep_nearer_hub_(scored_node_t& nearest, std::size_t hub, distance_t distance) noexcept {
            if (distance < nearest.distance) nearest = {static_cast<node_id_t>(hub), distance};
            return farthest_.distance;
        }

        /** The neighborhood an entry's @p nearest hub gives it; the first for one
         *  no distance reached. */
        static u16_t neighborhood_of_(scored_node_t nearest) noexcept {
            return static_cast<u16_t>(nearest.node_id == node_id_unknown_k ? 0 : nearest.node_id);
        }

        /** Draws the first gather's neighborhood hubs by seeded bijection over the visible entries,
         *  and packs them. */
        void draw_hubs_() noexcept {
            std::size_t const visible = health.visible, hubs = first_hubs_(visible);
            hub_nodes_.clear();
            for (std::size_t rank = 0; rank != hubs; ++rank)
                hub_nodes_.push_back_reserved(static_cast<node_id_t>(seeded_bijection(rank, visible, round_seed_)));
            std::span<node_id_t> const nodes {hub_nodes_.data(), hubs};
            partial_sort(nodes, hubs);
            hub_vectors_.clear();
            hub_vectors_.extend_uninitialized(hubs_bytes_(hubs));
            hub_norms_.clear();
            hub_norms_.extend_uninitialized(hubs);
            for (std::size_t rank = 0; rank != hubs; ++rank)
                hub_norms_[rank] = gather_(nodes[rank], hub_vectors_.data(), rank);
            index_->pack_queries_(hubs_view_(), hubs_packs_(), cache_bytes_);
            neighborhood_offsets_.clear();
        }

        /** The first gather's hubs as the scorer reads them, and where their packs start. */
        vectors_view_t hubs_view_() const noexcept { return gathered_view_(hub_vectors_.data(), hub_nodes_.size()); }
        byte_t const* hubs_packs_() const noexcept {
            return hub_vectors_.data() + hubs_bytes_(hub_nodes_.size()) -
                   scan_scorer_t::packs_bytes(hub_nodes_.size(), index_->dimensions());
        }
        byte_t* hubs_packs_() noexcept {
            return hub_vectors_.data() + hubs_bytes_(hub_nodes_.size()) -
                   scan_scorer_t::packs_bytes(hub_nodes_.size(), index_->dimensions());
        }

        /** Scores the stored entries of @p task against the packed hubs and files each
         *  under its nearest: the row minimum of the task's distances, ties to the lower
         *  hub. Once ordered by neighborhood, a task spans positions in that order instead,
         *  and only the neighborhoods over capacity gather again, each against hubs drawn
         *  among its own entries. */
        void gather_neighborhoods_(task_t const& task, traversal_context_t& context) noexcept {
            traversal_stats_t const before = context.telemetry();
            if (neighborhood_offsets_.empty()) {
                std::size_t const hubs = hub_nodes_.size();
                std::span<distance_t> const thresholds = unreached_thresholds_(context, hubs);
                std::span<scored_node_t> const nearest = nearest_hubs_(context, task.nodes.size());
                auto keep = [&](std::size_t hub, scored_node_t entry) noexcept {
                    return keep_nearer_hub_(nearest[entry.node_id - task.nodes.first], hub, entry.distance);
                };
                dummy_predicate_t admits;
                index_->scan_rows_(task.nodes, hubs_view_(), {hub_norms_.data(), hubs}, hubs_packs_(), thresholds,
                                   admits, keep, context, cache_bytes_);
                for (node_id_t node_id = task.nodes.first; node_id < task.nodes.bound; ++node_id)
                    entry_neighborhoods_[node_id] = neighborhood_of_(nearest[node_id - task.nodes.first]);
            }
            else {
                std::span<node_id_t const> const offsets = to_span(neighborhood_offsets_);
                std::size_t neighborhood = static_cast<std::size_t>(lower_bound(offsets, task.nodes.first + 1) -
                                                                    offsets.data());
                for (--neighborhood; neighborhood + 1 < offsets.size() && offsets[neighborhood] < task.nodes.bound;
                     ++neighborhood)
                    if (neighborhood_width_(neighborhood) > neighborhood_capacity_)
                        gather_run_(neighborhood,
                                    {max_of(offsets[neighborhood], task.nodes.first),
                                     min_of(offsets[neighborhood + 1], task.nodes.bound)},
                                    context);
            }
            add_spent_(context.telemetry() - before);
        }

        /** Files the entries at positions @p run of wide neighborhood @p neighborhood under the
         *  nearest of the hubs that neighborhood draws among its own entries, the same in every
         *  task, its entries and those hubs gathered by ID. */
        void gather_run_(std::size_t neighborhood, iota_view<node_id_t> run, traversal_context_t& context) noexcept {
            node_id_t const first = neighborhood_offsets_[neighborhood];
            std::size_t const entries = neighborhood_width_(neighborhood), hubs = hubs_for_(entries);
            u64_t const seed = mix_bits(round_seed_ ^ first);
            byte_t* const gathered = context.staged_queries.data();
            byte_t* const packs = gathered + gathered_packs_offset_();
            std::span<vector_norms_t> const hub_norms {context.query_norms.data(), hubs};
            for (std::size_t rank = 0; rank != hubs; ++rank)
                hub_norms[rank] = gather_(neighborhood_entries_[first + seeded_bijection(rank, entries, seed)],
                                          gathered + gathered_hubs_offset_(), rank);
            std::span<vector_norms_t> const run_norms {context.neighborhood_norms.data(), run.size()};
            for (std::size_t position = 0; position != run.size(); ++position)
                run_norms[position] = gather_(neighborhood_entries_[run.first + position], gathered, position);
            vectors_view_t const hubs_view = gathered_view_(gathered + gathered_hubs_offset_(), hubs);
            index_->pack_queries_(hubs_view, packs, cache_bytes_);
            std::span<distance_t> const thresholds = unreached_thresholds_(context, hubs);
            std::span<scored_node_t> const nearest = nearest_hubs_(context, run.size());
            auto keep = [&](std::size_t hub, scored_node_t entry) noexcept {
                return keep_nearer_hub_(nearest[entry.node_id], hub, entry.distance);
            };
            index_->scan_gathered_(gathered_view_(gathered, run.size()), 0, row_norms_(run_norms), hubs_view, hub_norms,
                                   packs, thresholds, keep, context, cache_bytes_, 0);
            for (std::size_t position = 0; position != run.size(); ++position)
                entry_neighborhoods_[neighborhood_entries_[run.first + position]] = neighborhood_of_(nearest[position]);
        }

        /** Sorts the entries by neighborhood, as @ref order_by_neighborhood does after a draw
         *  and @ref split_wide_neighborhoods after a gather over the neighborhoods over
         *  capacity, in the starts planning reserved; the tally's widest neighborhood tells
         *  whether the gather runs once more over those alone. */
        void order_neighborhoods_() noexcept {
            node_id_t const visible = health.visible;
            std::span<u16_t const> const tags {entry_neighborhoods_.data(), visible};
            std::span<node_id_t> const entries {neighborhood_entries_.data(), visible};
            std::size_t const neighborhoods = neighborhoods_();
            bool const first = neighborhood_offsets_.empty();
            neighborhood_offsets_.resize_reserved(first ? hub_nodes_.size() + 2 : split_room(visible, hubs_per_draw_));
            std::span<node_id_t> const offsets = to_writable_span(neighborhood_offsets_);
            neighborhood_order_t const order = first ? order_by_neighborhood(tags, hub_nodes_.size(), entries, offsets)
                                                     : split_wide_neighborhoods(tags, entries, offsets, neighborhoods,
                                                                                neighborhood_capacity_, hubs_per_draw_);
            neighborhood_offsets_.resize_reserved(order.neighborhoods + 1);
            tally_.neighborhoods = order.neighborhoods;
            tally_.widest_neighborhood = order.widest;
        }

        /** Re-stitches the neighborhood of @p task: gathers it, scores it against itself, and
         *  re-selects every entry's links among its own and its nearest neighborhood-mates,
         *  keeping only the diverse ones and those @ref keep_reach_ needs. One thread owns the
         *  neighborhood and writes only its entries' rows, each under the row's lock. */
        void restitch_neighborhoods_(task_t const& task, traversal_context_t& context) noexcept {
            traversal_stats_t const before = context.telemetry();
            std::size_t const count = neighborhood_width_(task.position), connectivity = index_->connectivity();
            std::span<node_id_t const> const entries {
                neighborhood_entries_.data() + neighborhood_offsets_[task.position], count};
            byte_t* const gathered = context.staged_queries.data();
            byte_t* const packs = gathered + gathered_packs_offset_();
            std::span<vector_norms_t> const norms {context.neighborhood_norms.data(), count};
            for (std::size_t position = 0; position != count; ++position)
                norms[position] = gather_(entries[position], gathered, position);
            vectors_view_t const neighborhood = gathered_view_(gathered, count);
            index_->pack_queries_(neighborhood, packs, cache_bytes_);

            std::span<scored_node_t> const mates {context.neighborhood_mates.data(), count * connectivity};
            for (scored_node_t& mate : mates) mate = farthest_;
            auto mates_of = [&](std::size_t position) noexcept {
                return mates.subspan(position * connectivity, connectivity);
            };
            auto offer = [&](std::size_t position, scored_node_t mate) noexcept {
                std::span<scored_node_t> const heap = mates_of(position);
                if (further_(heap[0], mate)) replace_farthest(heap, mate, further_);
            };
            // Each pair once, the later entry as the column, so none scores twice or with itself.
            auto keep = [&](std::size_t column, scored_node_t row) noexcept {
                if (column > row.node_id) {
                    offer(row.node_id, scored_node_t {static_cast<node_id_t>(column), row.distance});
                    offer(column, row);
                }
                return farthest_.distance;
            };
            std::span<distance_t> const thresholds = unreached_thresholds_(context, count);
            std::span<vector_norms_t const> const row_norms = row_norms_(norms);
            for (std::size_t row = 0; row < count; row += scan_scorer_t::step_rows_k) {
                std::size_t const rows = min_of<std::size_t>(scan_scorer_t::step_rows_k, count - row);
                index_->scan_gathered_(neighborhood(nk::range(row, row + rows), nk::slice), static_cast<node_id_t>(row),
                                       row_norms.empty() ? row_norms : row_norms.subspan(row, rows), neighborhood,
                                       norms, packs, thresholds, keep, context, cache_bytes_, row);
            }

            results_array_t& results = context.results;
            selection_t& selected = context.selected_neighbors;
            selection_t& live = context.refined_neighbors;
            std::size_t added = 0, dropped = 0, changed = 0, selecting = 0;
            for (std::size_t position = 0; position != count; ++position) {
                node_id_t const node_id = entries[position];
                location_t const location = chunk_schedule_t::locate(node_id);
                results.clear();
                candidates_count_t const live_count = index_->stage_neighbors_(index_->neighbors_for(location), node_id,
                                                                               context);
                std::span<node_id_t const> const live_ids {context.staged_ids.data(), live_count};
                context.computed_distances += index_->add_neighbor_candidates_(
                    index_->vector_for(location), index_->norm_for(location), live_ids, health.visible, context);
                [[maybe_unused]] nk::status_t const sized = live.resize({results.size()});
                for (std::size_t rank = 0; rank != results.size(); ++rank) live[rank] = results[rank];
                for (scored_node_t const mate : mates_of(position)) {
                    if (mate.node_id == node_id_unknown_k) continue;
                    scored_node_t const candidate {entries[mate.node_id], mate.distance};
                    bool listed = false;
                    for (std::size_t rank = 0; rank != live.size() && !listed; ++rank)
                        listed = live[rank].node_id == candidate.node_id;
                    if (!listed) results.insert(scored_node_t {candidate}, results.size() + 1);
                }
                selecting += index_->select_diverse_via_quads(results, node_id, connectivity, selected, context);
                index_->keep_reach_(node_id, live, selected, health);
                std::size_t kept = 0;
                for (std::size_t rank = 0; rank != selected.size(); ++rank)
                    for (node_id_t const link : live_ids) kept += link == selected[rank].node_id;
                added += selected.size() - kept;
                dropped += live_ids.size() - kept;
                if (kept == selected.size() && kept == live_ids.size()) continue;
                static_cast<void>(index_->rewrite_row_(node_id, selected, live_ids));
                changed += !count_changed_(node_id);
            }
            context.computed_distances += selecting;
            context.computed_distances_in_refines += selecting;
            add_(tally_.reselected_nodes, count);
            add_(tally_.changed_nodes, changed);
            add_(tally_.added_links, added);
            add_(tally_.dropped_links, dropped);
            add_spent_(context.telemetry() - before);
        }

        /** Links every unreached or stranded island whose first rank falls in @p task through the
         *  root's component, with no distance: an unreached one's entry from the first of its own
         *  links inside the component that takes it, a stranded one's entry to the entry of the
         *  component linking to it. A component with neither waits for a later round, which its
         *  restitch gives the links to heal through. */
        void heal_(task_t const& task) noexcept {
            auto keeps_none = [](node_id_t) noexcept { return false; };
            std::size_t healed = 0, waiting = 0;
            for (std::size_t rank = task.nodes.first; rank < task.nodes.bound; ++rank) {
                if (!health.island_first(rank)) continue;
                u8_t const roles = health.island_roles(rank);
                node_id_t const entry = health.island_entry(rank);
                if (roles & health_plan_t::unreached_island_k) {
                    // A full owner row may refuse, so the next link inside the component
                    // offers its own.
                    node_id_t owner = node_id_unknown_k;
                    bool linked = false, owned = false;
                    index_->for_each_member_(entry, [&](node_id_t link) noexcept {
                        if (linked || link >= health.visible || !health.in_component(link)) return;
                        owned = true;
                        linked = add_or_swap_link_(link, entry, keeps_none) == taken_t::written_k;
                        if (linked) owner = link;
                    });
                    if (linked) health.record_heal_link(owner, entry), ++healed;
                    waiting += !owned;
                }
                node_id_t const parent = health.island_entry_parent(rank);
                if ((roles & health_plan_t::stranded_island_k) && parent != node_id_unknown_k &&
                    add_or_swap_link_(entry, parent, keeps_none) == taken_t::written_k)
                    health.record_heal_link(entry, parent), ++healed;
                waiting += (roles & health_plan_t::stranded_island_k) && parent == node_id_unknown_k;
            }
            add_(tally_.islands_reconnected, healed);
            add_(islands_waiting_, waiting);
        }

        /**
         *  @brief Links @p member from @p owner 's row under its lock: a free slot first; else a
         *      slot of the root's component.
         *
         *  That slot's resident is the one linked to by the most entries, taken when it stays
         *  linked to by more than @p member, @p keeps does not hold it, and the reach rule releases
         *  it. Adds go first, since an add dead ends no search, while a swap may for a target
         *  outside the round's sample. The in-link counts follow every write.
         */
        template <typename keeps_type_>
        taken_t add_or_swap_link_(node_id_t owner, node_id_t member, keeps_type_&& keeps) noexcept {
            if (owner == member || owner >= health.visible || member >= health.visible) return taken_t::refused_k;
            candidates_count_t const connectivity = static_cast<candidates_count_t>(index_->connectivity());
            neighbors_ref_t const row = index_->neighbors_for(chunk_schedule_t::locate(owner));
            node_id_t const first = row.lock();
            taken_t taken = taken_t::refused_k;
            candidates_count_t free_slot = connectivity, richest = connectivity;
            if (first == owner) {
                row.unlock(member);
                taken = taken_t::written_k;
            }
            else {
                bool present = first == member;
                for (candidates_count_t slot = 1; slot != connectivity && !present; ++slot) {
                    node_id_t const resident = row.load_slot(slot);
                    present = resident == member;
                    if (resident == node_id_unknown_k) {
                        free_slot = slot;
                        break;
                    }
                }
                unsigned richest_links = health.in_links(member);
                for (candidates_count_t slot = 0;
                     !present && free_slot == connectivity && slot != connectivity && health.in_component(owner);
                     ++slot) {
                    node_id_t const resident = slot ? row.load_slot(slot) : first;
                    if (resident >= health.visible || health.in_links(resident) <= richest_links + 1u ||
                        keeps(resident))
                        continue;
                    richest = slot, richest_links = health.in_links(resident);
                }
                node_id_t const displaced = richest == connectivity ? node_id_unknown_k
                                            : richest               ? row.load_slot(richest)
                                                                    : first;
                if (present) taken = taken_t::present_k;
                else if (free_slot != connectivity) row.store_slot(free_slot, member), taken = taken_t::written_k;
                else if (richest != connectivity && health.take_link_under_lock_if_spare(owner, displaced)) {
                    if (richest) row.store_slot(richest, member);
                    static_cast<void>(health.take_in_link_if_spare(displaced));
                    taken = taken_t::written_k;
                }
                row.unlock(taken == taken_t::written_k && richest == 0 ? member : first);
            }
            if (taken != taken_t::written_k) return taken;
            health.count_in_link(member);
            add_(tally_.changed_nodes, !count_changed_(owner));
            return taken;
        }

        /** Whether @p left lies farther than @p right, the larger ID farther among equals. */
        static constexpr auto further_ = [](scored_node_t const& left, scored_node_t const& right) noexcept {
            return left.distance > right.distance || (left.distance == right.distance && left.node_id > right.node_id);
        };

        /** An empty place in a kept heap, farther than any entry. */
        static constexpr scored_node_t farthest_ {node_id_unknown_k, std::numeric_limits<distance_t>::infinity()};

        /** The refined index; null on a moved-from plan. */
        ram_index* index_ = nullptr;

        /** Per entry, whether the round counted it changed already. */
        growing_bitset_t changed_;

        /** The phase whose tasks run now. */
        phase_t phase_ = phase_t::done_k;

        /** The phase the current health check follows, none for the opening one - what decides the
         *  phase after it converges. */
        phase_t swept_after_ = phase_t::none_k;

        /** Island components the last heal could not link, having no link to or from the root's,
         *  and those the heal before it could not, which the repeats must shrink. */
        std::size_t islands_waiting_ = 0;
        std::size_t waiting_before_ = 0;

        /** Counts summed through @c atomic_ref by the tasks, and set by @ref advance. */
        refine_report_t tally_;

        /** The first failure the plan met: an audit whose islands' adjacency could
         *  not be placed. */
        status_t status_ = status_code_t::success_k;

        /** Bytes of cache each core keeps to itself, which sizes the hubs and a
         *  neighborhood's capacity. */
        std::size_t cache_bytes_ = 0;

        /** Hubs one gather draws at most: one pack of the scan's queries. */
        std::size_t hubs_per_draw_ = 0;

        /** The most entries one neighborhood holds: the row block sharing a core's cache with one
         *  pack, and so the most entries one restitch takes at once. */
        std::size_t neighborhood_capacity_ = 0;

        /** The round's hubs, ascending, then their stored vectors, norms and packs. */
        part_bases_t hub_nodes_;
        growing_buffer<byte_t, bytes_allocator_t> hub_vectors_;
        growing_buffer<vector_norms_t, norms_allocator_t> hub_norms_;

        /** Per entry, the neighborhood its nearest hub gave it, among the hubs of
         *  its last gather. */
        growing_buffer<u16_t, neighborhoods_allocator_t> entry_neighborhoods_;

        /** The entries by neighborhood, each one's ascending, and where each neighborhood's run
         *  starts, one past the last closing them. */
        part_bases_t neighborhood_entries_;
        part_bases_t neighborhood_offsets_;

        /** The seed of the round's draws. */
        u64_t round_seed_ = 0;
    };

    /**
     *  @brief Opens one refinement round over the published nodes, planning no entry ahead: the
     *      round gathers every entry into a neighborhood and re-stitches each, as
     *      @ref refine_plan_t describes.
     *
     *  Everything the round keeps across its barriers comes from the index's own allocator, sized
     *  from the published nodes and the cache @p config names, about ten bytes per entry beside the
     *  audit's; the tasks fail only where a context cannot grow to a round's scratch, which
     *  @c refine_plan_t::reserve grows ahead of them.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The audit or the round's columns cannot be placed.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config = {}) noexcept {
        auto [health, status] = plan_health();
        refine_plan_t plan {std::move(health), typename refine_plan_t::part_bases_allocator_t {vectors_allocator_}};
        if (failed(status)) return {std::move(plan), status};
        plan.index_ = this;
        if (failed(plan.reserve_round_(config.cache_bytes))) return {std::move(plan), status_code_t::out_of_memory_k};
        plan.swept_after_ = phase_t::none_k;
        plan.phase_ = phase_t::health_k;
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief Runs one refinement round over every published node on the calling thread: one
     *      @ref plan_refine, its tasks drained in order.
     *
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The calling thread's scratch & telemetry, reused across rounds.
     *
     *  @par Statuses:
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     *  - @c out_of_memory_k The plan's columns cannot be placed, or @p context cannot fit a task.
     */
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(config);
        if (failed(status)) return {{}, status};
        if (status_t const sized = reserve_traversal(context, insert_config_t {non_zero_expansion_factor(config)});
            failed(sized)) {
            [[maybe_unused]] auto const [report, report_status] = plan.finish();
            return {report, sized};
        }
        if (status_t const sized = plan.reserve(context); failed(sized)) {
            [[maybe_unused]] auto const [report, report_status] = plan.finish();
            return {report, sized};
        }
        do
            for (typename refine_plan_t::task_t const task : plan.tasks())
                if (status_t const ran = plan.run(task, context); failed(ran)) {
                    [[maybe_unused]] auto const [report, report_status] = plan.finish();
                    return {report, ran};
                }
        while (plan.advance());
        return plan.finish();
    }

    /**
     *  @brief One reshard in flight out of one source into its targets: the moves, then one round
     *      on every target in turn.
     *
     *  The partition grows the source's rows into one graph-local part per target, breadth first
     *  over the links, one root per part through @ref seeded_bijection under the config's seed,
     *  each part capped at the rows over the targets rounded up; rows no growth reached seed the
     *  smallest part. Scatter tasks copy each row's key, vector, norms and links into its target,
     *  the links inside its part rebased and the links across dropped, one write task per target
     *  files its keys, and the barrier closing the writes publishes every target's rows, then
     *  forgets the source's, so its IDs restart from zero. Every target then runs one round as
     *  @ref refine_plan_t does, which links in what the dropped links left apart, the rounds
     *  summing over the targets; @ref next_round runs another round on every target.
     *
     *  The plan owns its target list and every claim, so a plan dying before the writes published
     *  releases the targets' claims and leaves the source as it was. Tasks of one phase are
     *  disjoint, and @ref advance is the single-caller barrier, as in @ref refine_plan_t.
     */
    struct reshard_plan_t {
        using task_t = typename refine_plan_t::task_t;
        using part_bases_allocator_t = typename refine_plan_t::part_bases_allocator_t;
        using part_bases_t = typename refine_plan_t::part_bases_t;

        /** An unbound plan placed by @p allocator - what a refused plan answers with. */
        explicit reshard_plan_t(refine_plan_t&& rounds, part_bases_allocator_t const& allocator) noexcept
            : rounds_(std::move(rounds)), targets_(allocator), target_claims_(allocator), part_slots_(allocator),
              part_members_(allocator), part_extents_(allocator) {}
        reshard_plan_t(reshard_plan_t&& other) noexcept
            : rounds_(std::move(other.rounds_)), source_(std::exchange(other.source_, nullptr)),
              targets_(std::move(other.targets_)), target_claims_(std::move(other.target_claims_)),
              part_slots_(std::move(other.part_slots_)), part_members_(std::move(other.part_members_)),
              part_extents_(std::move(other.part_extents_)), seed_(other.seed_), moves_(other.moves_),
              refined_targets_(other.refined_targets_), targets_tally_(other.targets_tally_) {}
        reshard_plan_t(reshard_plan_t const&) = delete;
        reshard_plan_t& operator=(reshard_plan_t const&) = delete;

        /** Task @p position of the current phase: the whole partition, a source row scattered, a
         *  target's keys, or a task of the bound target's round; past the end, an empty task. */
        task_t operator[](std::size_t position) const noexcept {
            if (!moving()) return rounds_[position];
            task_t task;
            if (position >= tasks().size()) return task;
            task.position = position;
            if (moves_ == phase_t::partition_k) task.nodes = {0, split_nodes_()};
            else if (moves_ == phase_t::write_k) task.nodes = {0, part_extents_[position].nodes};
            else task.nodes = {static_cast<node_id_t>(position), static_cast<node_id_t>(position + 1)};
            return task;
        }

        /** The current phase's tasks, positionally: one partition, a scatter per source row, a
         *  write per target, then the bound target's round. */
        indexed_range<reshard_plan_t> tasks() const noexcept {
            if (!moving()) return {this, rounds_.tasks().size()};
            if (moves_ == phase_t::partition_k) return {this, 1u};
            if (moves_ == phase_t::write_k) return {this, targets_.size()};
            return {this, std::size_t {split_nodes_()}};
        }

        /**
         *  @brief Runs @p task of the current phase on the calling thread - safe for disjoint tasks
         *      of one phase, and never failing on a bound plan once @ref reserve grew @p context.
         *
         *  The one partition task grows the parts, a scatter task copies one source row into its
         *  target, and a write task files one target's keys, its map's one writer; the rounds run
         *  as @ref refine_plan_t::run does.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no source.
         */
        status_t run(task_t const& task, traversal_context_t& context) noexcept {
            if (!rounds_.index_) return status_code_t::uninitialized_k;
            switch (moves_) {
            case phase_t::partition_k: partition_(); break;
            case phase_t::scatter_k: scatter_(task); break;
            case phase_t::write_k: targets_[task.position]->keyed_.publish(task.nodes); break;
            default: return rounds_.run(task, context);
            }
            return status_code_t::success_k;
        }

        /** The barrier between phases, one caller after every task of the current one ran:
         *  publishes the parts once their keys are written, binds the next target once a round
         *  ended, and answers whether a phase follows. */
        bool advance() noexcept {
            if (!rounds_.index_ || failed(rounds_.status_)) return false;
            if (moves_ == phase_t::partition_k) return moves_ = phase_t::scatter_k, true;
            if (moves_ == phase_t::scatter_k) return moves_ = phase_t::write_k, true;
            if (moves_ == phase_t::write_k) {
                publish_parts_();
                moves_ = phase_t::done_k;
                bind_target_();
                return true;
            }
            if (rounds_.advance()) return true;
            // A plan whose last target's round ended settles nothing more.
            if (failed(rounds_.status_) || refined_targets_ == targets_.size()) return false;
            return settle_target_();
        }

        /** Stops the bound target's round as @ref refine_plan_t::stop does, past the moves. */
        void stop() noexcept {
            if (!moving()) rounds_.stop();
        }

        /** Whether the current phase moves rows or keys - the phases no stop skips. */
        bool moving() const noexcept { return moves_ != phase_t::done_k; }

        /** Whether the bound target is inside a round. */
        bool rounding() const noexcept { return !moving() && rounds_.rounding(); }

        /** The phase the plan is in: a move's, then the bound target's round's. */
        phase_t phase() const noexcept { return moving() ? moves_ : rounds_.phase(); }

        /** Where the round's phase sits among its six; zero in the moves. */
        std::size_t ordinal() const noexcept { return moving() ? 0 : rounds_.ordinal(); }

        /** Units of the round's phase done and in all; none in the moves, which report nothing. */
        std::size_t completed() const noexcept { return moving() ? 0 : rounds_.completed(); }
        std::size_t total() const noexcept { return moving() ? 0 : rounds_.total(); }

        /** The first failure the plan met, which ended it; success otherwise. */
        status_t status() const noexcept { return rounds_.status(); }

        /** Grows @p context to what a round's task touches, as @ref refine_plan_t::reserve does. */
        status_t reserve(traversal_context_t& context) const noexcept { return rounds_.reserve(context); }

        /** Runs another round on every target once every target's ended, from the first; a no-op
         *  in the moves and mid-round. */
        void next_round() noexcept {
            if (moving() || rounds_.phase_ != phase_t::done_k || failed(rounds_.status_)) return;
            refined_targets_ = 0;
            bind_target_();
        }

        /**
         *  @brief The rounds so far, summed over the targets: complete once @ref advance answered
         *      false, else cancelled.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no source.
         *  - @c cancelled_k The moves or a round stopped before @ref advance answered false.
         *  - @c out_of_memory_k A health check could not place its islands' adjacency.
         */
        expected<refine_report_t> finish() const noexcept {
            if (!rounds_.index_) return {{}, status_code_t::uninitialized_k};
            if (moving()) return {{}, status_code_t::cancelled_k};
            return rounds_.finish();
        }

      private:
        friend ram_index;

        /** Rows the source published at planning: one slot each. */
        node_id_t split_nodes_() const noexcept { return static_cast<node_id_t>(part_slots_.size()); }

        /** The most rows one part takes: the source's rows over the targets, rounded up. */
        std::size_t part_capacity_() const noexcept {
            return divide_round_up<std::size_t>(part_slots_.size(), targets_.size());
        }

        /** Calls @p callback with every live link of source row @p node, skipping the self-loop of
         *  a published-empty row; reads plain, since nothing writes the source meanwhile. */
        template <typename callback_type_>
        void for_each_link_(node_id_t node, callback_type_&& callback) const noexcept {
            node_ids_view_t const links = static_cast<ram_index const&>(*source_).neighbors_for(
                chunk_schedule_t::locate(node));
            for (std::size_t slot = 0; slot != links.size() && links[slot] != node_id_unknown_k; ++slot)
                if (links[slot] != node) callback(links[slot]);
        }

        /** Places source node @p node last in @p part, its slot naming the part and position. */
        void enlist_(node_id_t node, std::size_t part) noexcept {
            std::size_t const position = part_extents_[part].nodes++, capacity = part_capacity_();
            part_members_[part * capacity + position] = node;
            part_slots_[node] = static_cast<node_id_t>(part * capacity + position);
        }

        /** Expands the oldest unexpanded member of @p part, enlisting its unplaced neighbors while
         *  the part has room; answers whether the part could expand at all. */
        bool expand_(std::size_t part) noexcept {
            node_id_t& expanded = part_extents_[part].expanded;
            std::size_t const capacity = part_capacity_();
            if (expanded == part_extents_[part].nodes || part_extents_[part].nodes == capacity) return false;
            node_id_t const node_id = part_members_[part * capacity + expanded++];
            node_id_t const split_nodes = split_nodes_();
            for_each_link_(node_id, [&](node_id_t target) noexcept {
                if (target < split_nodes && part_slots_[target] == node_id_unknown_k &&
                    part_extents_[part].nodes != capacity)
                    enlist_(target, part);
            });
            return true;
        }

        /** Grows one part per target from a root the seed picks, breadth first over the source's
         *  links and one expansion per part in turn, each capped at the part capacity; every node
         *  left unplaced then seeds a growth of the smallest part. Every node's slot names its part
         *  and its ID in the target, in the order the growth enlisted it. */
        void partition_() noexcept {
            std::size_t const parts = targets_.size();
            node_id_t const split_nodes = split_nodes_();
            for (node_id_t& slot : part_slots_) slot = node_id_unknown_k;
            for (std::size_t part = 0; part != parts; ++part) part_extents_[part] = {};
            for (std::size_t part = 0; part != parts && part != split_nodes; ++part)
                enlist_(static_cast<node_id_t>(seeded_bijection(part, split_nodes, seed_)), part);
            for (std::size_t expanded = parts; expanded;) {
                expanded = 0;
                for (std::size_t part = 0; part != parts; ++part) expanded += expand_(part);
            }
            for (node_id_t node = 0; node != split_nodes; ++node) {
                if (part_slots_[node] != node_id_unknown_k) continue;
                std::size_t smallest = 0;
                for (std::size_t part = 1; part != parts; ++part)
                    if (part_extents_[part].nodes < part_extents_[smallest].nodes) smallest = part;
                enlist_(node, smallest);
                while (expand_(smallest)) {}
            }
        }

        /** Copies the source nodes of @p task into their targets' claimed rows - keys, vectors,
         *  norms, and the links inside each part rebased through its slots, the links across
         *  dropped, a row left with none becoming the published-empty self-loop. */
        void scatter_(task_t const& task) noexcept {
            ram_index const& source = *source_;
            geometry_t const geometry = source.geometry();
            std::size_t const connectivity = geometry.connectivity;
            std::size_t const row_bytes = padded_row_values(geometry) * sizeof(scalar_t);
            std::size_t const capacity = part_capacity_();
            node_id_t const split_nodes = split_nodes_();
            for (node_id_t node = task.nodes.first; node < task.nodes.bound; ++node) {
                std::size_t const slot = part_slots_[node], part = slot / capacity;
                node_id_t const placed = static_cast<node_id_t>(slot % capacity);
                ram_index& into = *targets_[part];
                location_t const from = chunk_schedule_t::locate(node), to = chunk_schedule_t::locate(placed);
                into.keyed_.assign(placed, source.keyed_[node]);
                std::memcpy(into.vectors_[to], source.vectors_[from], row_bytes);
                if (source.stores_norms()) into.norm_ref_for(to) = source.norms_[from];
                node_id_t* const links = into.neighbors_[to];
                std::size_t kept = 0;
                for_each_link_(node, [&](node_id_t target) noexcept {
                    if (target < split_nodes && part_slots_[target] / capacity == part)
                        links[kept++] = static_cast<node_id_t>(part_slots_[target] % capacity);
                });
                if (!kept) links[kept++] = placed;
                for (std::size_t tail = kept; tail != connectivity; ++tail) links[tail] = node_id_unknown_k;
            }
        }

        /** Publishes every target's nodes, consuming the claim of a full part and returning what
         *  the part left unused, then empties the source. */
        void publish_parts_() noexcept {
            std::size_t const capacity = part_capacity_();
            for (std::size_t part = 0; part != targets_.size(); ++part) {
                ram_index& into = *targets_[part];
                node_id_t const nodes = part_extents_[part].nodes;
                target_claims_[part].consume();
                if (capacity != nodes) into.release_capacity(capacity - nodes);
                if (!nodes) continue;
                node_id_t dispensed = 0;
                atomic_ref<node_id_t>(into.next_dispensed_id_)
                    .compare_exchange_strong(dispensed, nodes, std::memory_order_release, std::memory_order_relaxed);
                atomic_ref<node_id_t>(into.last_visible_id_).store(nodes, std::memory_order_release);
            }
            std::exchange(source_, nullptr)->forget_(split_nodes_());
        }

        /** Turns the rounds to the next target: its health check over the rows its part received,
         *  their changed marks cleared, and its round from an empty count. */
        void bind_target_() noexcept {
            std::size_t const target = refined_targets_;
            rounds_.index_ = targets_[target];
            rounds_.health.rebind(*rounds_.index_, part_extents_[target].nodes);
            // Within what planning reserved for a full part.
            [[maybe_unused]] status_t const unchanged = rounds_.changed_.resize(
                health_plan_t::words_for(rounds_.health.visible));
            rounds_.tally_ = {};
            rounds_.swept_after_ = phase_t::none_k;
            rounds_.phase_ = phase_t::health_k;
        }

        /** Folds an ended round into the reshard's, then binds the next target and answers true,
         *  or leaves the sums in the rounds once the last target's round ended. */
        bool settle_target_() noexcept {
            if (refined_targets_) targets_tally_ += rounds_.tally_;
            else targets_tally_ = rounds_.tally_;
            if (++refined_targets_ == targets_.size()) return rounds_.tally_ = targets_tally_, false;
            bind_target_();
            return true;
        }

        /** One round's columns over a full part, rebound to every target in turn. */
        refine_plan_t rounds_;

        /** The index the reshard empties once its rows publish in the targets; null otherwise. */
        ram_index* source_ = nullptr;

        /** The targets, one part each, copied at planning. */
        growing_buffer<ram_index*,
                       typename std::allocator_traits<part_bases_allocator_t>::template rebind_alloc<ram_index*>>
            targets_;

        /** One claim of the part capacity per target, its rows released unless the writes
         *  published them. */
        growing_buffer<capacity_claim_t, part_bases_allocator_t> target_claims_;

        /** Per source row: its part times the part capacity, plus its row in that target. */
        part_bases_t part_slots_;

        /** Per part, a run of the part capacity: its source nodes in target order. */
        part_bases_t part_members_;

        /** One part's rows, and how many of them its growth expanded. */
        struct part_extent_t {
            node_id_t nodes = 0;
            node_id_t expanded = 0;
        };

        /** Per part, its extent. */
        growing_buffer<part_extent_t, part_bases_allocator_t> part_extents_;

        /** What picks the parts' roots. */
        u64_t seed_ = 0;

        /** The phase of the moves still pending; done once the parts published. */
        phase_t moves_ = phase_t::done_k;

        /** Targets whose rounds ended so far; the bound one is the next. */
        std::size_t refined_targets_ = 0;

        /** The reports of the targets' rounds ended so far, folded. */
        refine_report_t targets_tally_ {};
    };

    /**
     *  @brief Opens a reshard of @p sources into @p targets: this index, the one source, moves into
     *      one graph-local part per target, then every target runs one round, as
     *      @ref reshard_plan_t describes; the source ends empty.
     *
     *  Every fallible step happens here: the target list copied, a full part's rows claimed in
     *  every target, and one audit and one round's columns over a full part, everything the plan
     *  keeps on this index's own allocator. The tasks cannot fail on contexts
     *  @c reshard_plan_t::reserve grew past the plan's expansion factor. Nothing may insert into
     *  this index or a target meanwhile, nor search this one; the sharded index closes the
     *  dispensers, and a direct caller keeps the others out itself.
     *
     *  @param[in] sources This index alone: a RAM mono-index moves one source into its targets.
     *  @param[in] targets The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] config The rounds' knobs, zeros meaning "derive from connectivity", and the seed
     *      picking the parts' roots, so equal seeds over equal graphs split alike.
     *  @param[in] rows The rows this index holds once the moves planned ahead of the split land,
     *      as a gather into it before its cut plans them; the rows it holds now otherwise.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k No source, no target, or a null or repeated index in either list.
     *  - @c unsupported_k Several sources and several targets.
     *  - @c config_mismatch_k Sources other than this index alone, fewer than two targets, a
     *    non-empty one, this index itself, another metric or another geometry.
     *  - @c contended_k An insertion in flight on this index or on a target.
     *  - @c overflow_k The parts pass the 32-bit ID space.
     */
    expected<reshard_plan_t> plan_reshard(std::span<ram_index* const> sources, std::span<ram_index* const> targets,
                                          reshard_config_t config, std::size_t rows) noexcept {
        using bitset_allocator_t = typename health_plan_t::bitset_allocator_t;
        using part_bases_allocator_t = typename refine_plan_t::part_bases_allocator_t;
        reshard_plan_t plan {refine_plan_t {health_plan_t {bitset_allocator_t {vectors_allocator_}},
                                            part_bases_allocator_t {vectors_allocator_}},
                             part_bases_allocator_t {vectors_allocator_}};
        std::size_t const parts = targets.size();
        if (sources.empty() || targets.empty()) return {std::move(plan), status_code_t::invalid_argument_k};
        for (std::span<ram_index* const> const listed : {sources, targets})
            if (status_t const refused = listed_refusal(listed); failed(refused)) return {std::move(plan), refused};
        if (sources.size() > 1 && parts > 1) return {std::move(plan), status_code_t::unsupported_k};
        if (sources.front() != this || parts < 2) return {std::move(plan), status_code_t::config_mismatch_k};
        for (ram_index const* into : targets) {
            if (into == this || into->kernel_ != kernel_ || into->size() || into->geometry() != geometry_)
                return {std::move(plan), status_code_t::config_mismatch_k};
            if (!into->settled_()) return {std::move(plan), status_code_t::contended_k};
        }
        if (!settled_()) return {std::move(plan), status_code_t::contended_k};

        std::size_t const capacity = divide_round_up<std::size_t>(rows, parts);
        if (rows >= node_id_unknown_k || parts * capacity > static_cast<std::size_t>(node_id_unknown_k))
            return {std::move(plan), status_code_t::overflow_k};
        if (failed(plan.targets_.reserve_exactly(parts)) || failed(plan.target_claims_.reserve_exactly(parts)) ||
            failed(plan.part_extents_.resize(parts)) || failed(plan.part_slots_.resize(rows)) ||
            failed(plan.part_members_.resize(parts * capacity)))
            return {std::move(plan), status_code_t::out_of_memory_k};
        for (ram_index* into : targets) plan.targets_.push_back_reserved(into);
        // One audit over a full part sweeps every target in turn, rebound as its turn comes.
        auto [health, health_status] = plan_audit_(static_cast<node_id_t>(capacity));
        if (failed(health_status)) return {std::move(plan), health_status};
        plan.rounds_.health = std::move(health);
        // A full part's rows in every target, claimed from its empty watermark, die with the plan
        // unless the writes publish them.
        for (ram_index* into : targets) {
            auto [claimed, claim_status] = into->claim(capacity);
            if (failed(claim_status)) return {std::move(plan), claim_status};
            bool const based = claimed.base() == 0;
            plan.target_claims_.push_back_reserved(std::move(claimed));
            if (!based) return {std::move(plan), status_code_t::contended_k};
        }

        // One round's columns over a full part serve every target in turn.
        plan.rounds_.index_ = targets[0];
        if (failed(plan.rounds_.reserve_round_(config.refine.cache_bytes)))
            return {std::move(plan), status_code_t::out_of_memory_k};
        plan.source_ = this;
        plan.seed_ = config.seed;
        plan.moves_ = phase_t::partition_k;
        return {std::move(plan), status_code_t::success_k};
    }

    /** @ref plan_reshard of the rows this index holds now. */
    expected<reshard_plan_t> plan_reshard(std::span<ram_index* const> sources, std::span<ram_index* const> targets,
                                          reshard_config_t config = {}) noexcept {
        return plan_reshard(sources, targets, config, size());
    }

#pragma endregion Refinement

#pragma region Search

    /** Optional tuning knobs for @c search(); zeros mean "derive from connectivity". */
    using search_config_t = dense::search_config_t;

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

        /** Hint keys no shard held a node under. */
        std::size_t missing_hints = 0;
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
     *  @c immutable_k - or stronger - the caller marks no mutators and the traversal runs
     *  atomics-free. @ref dense_ram_index_concurrency
     *
     *  @param[in] query_vector The query, in the index's own dtype & dimensionality.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *
     *  @par Statuses:
     *  - @c unsupported_k @p config names neither a walk nor a scan.
     *  - @c config_mismatch_k The query's width differs from the index's dimensions.
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     */
    expected<search_result_t> search(vector_view_t query_vector, search_config_t config,
                                     traversal_context_t& context) noexcept {
        if (config.traversal == traversal_t::scan_k)
            return scan_if_(query_vector, dummy_predicate_t {}, context, config, static_cast<node_id_t>(size()));
        return search(query_vector, dummy_fold_t {}, config, context);
    }

    /**
     *  @brief Finds the published nodes a lightning indexer scores highest for one query row - the
     *      @c search contract, with every distance 1 − the node's indexer score.
     *
     *  @param[in] query_vector The row the walk navigates by, conventionally Σₕ wₕ·qₕ.
     *  @param[in] heads The indexer heads of this one query row.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the highest score first.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index does not walk by dot.
     */
    expected<search_result_t> search(vector_view_t query_vector, indexer_heads<scalar_t> const& heads,
                                     search_config_t config, traversal_context_t& context) noexcept
        requires a_floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_code_t::config_mismatch_k};
        return walk_(query_vector, dummy_fold_t {}, indexer_rescoring<quads_kit_t> {heads}, context, config,
                     static_cast<node_id_t>(size()));
    }

    /**
     *  @brief Walks exactly like @c search while @p fold reads every quad the walk scores - the
     *      entry point for verbs that reduce a walk to something other than its nearest matches.
     *
     *  @param[in] query_vector The query, in the index's own dtype & dimensionality.
     *  @param[in,out] fold The reader of every scored quad - see @c fold.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The calling thread's scratch & telemetry: the nearest published
     *      candidates end up in @c context.results, ascending by distance.
     *
     *  @par Statuses:
     *  - @c unsupported_k @p config names a traversal other than a walk.
     *  - @c config_mismatch_k The query's width differs from the index's dimensions.
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     */
    template <typename fold_type_>
        requires a_fold<std::remove_cvref_t<fold_type_>, dot_result_t>
    expected<search_result_t> search(vector_view_t query_vector, fold_type_&& fold, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return walk_(query_vector, fold, dummy_rescoring_t {}, context, config, static_cast<node_id_t>(size()));
    }

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector that satisfy
     *      @p predicate - the filtered sibling of @c search, sharing its whole contract.
     *
     *  Non-matching nodes still score and route the walk - filtering navigation itself would strand
     *  it in disconnected islands - but only matching nodes enter the reported candidates, whose
     *  radius governs termination once it holds @c expansion_factor entries; until then the
     *  ordinary navigation radius bounds the frontier. Predicates come in two shapes, detected by
     *  signature: per-node @c admission_t(node_id_t), or batched - the scored quad and an
     *  admissions buffer, pushing what it admits. Each node is tested at most once per query, so
     *  the exact visited set is the predicate's memoization.
     *
     *  @param[in] query_vector The query, in the index's own dtype & dimensionality.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in] predicate The admission predicate, called with published node IDs only.
     *  @param[in,out] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *
     *  @par Statuses:
     *  - @c unsupported_k @p config names neither a walk nor a scan.
     *  - @c config_mismatch_k The query's width differs from the index's dimensions.
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t query_vector, search_config_t config, predicate_type_&& predicate,
                                        traversal_context_t& context) noexcept {
        if (config.traversal == traversal_t::scan_k) {
            return scan_if_(query_vector, predicate, context, config, static_cast<node_id_t>(size()));
        }
        return walk_if_(query_vector, predicate, context, config, static_cast<node_id_t>(size()));
    }

    /**
     *  @brief Graph products of one left-hand row: every pair the walk scores, node IDs beside native
     *      dots, written until the spans fill.
     *
     *  @param[in] left The left-hand row, in the index's own dtype & dimensionality.
     *  @param[out] node_ids The node ID of each kept pair, in walk order.
     *  @param[out] products The dot of each kept pair, parallel to @p node_ids.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The calling thread's scratch & telemetry.
     *  @return The pairs scored in @c count - above the spans' width when they were truncated - or
     *      @c config_mismatch_k unless the index walks by dot.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index does not walk by dot, or @p products and @p node_ids
     *    differ in length.
     *  - @c unsupported_k @p config names a traversal other than a walk.
     *  - @c config_mismatch_k The row's width differs from the index's dimensions.
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     */
    expected<search_result_t> multiply(vector_view_t left, std::span<node_id_t> node_ids,
                                       std::span<dot_result_t> products, search_config_t config,
                                       traversal_context_t& context) noexcept {
        if (kernel_ != nk_kernel_dot_k || products.size() != node_ids.size())
            return {{}, status_code_t::config_mismatch_k};
        multiply_fold<node_id_t, dot_result_t> fold {node_ids, products};
        auto [walked, status] = search(left, fold, config, context);
        walked.count = fold.scored;
        return {walked, status};
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
     *      <tt>void(std::span&lt;node_id_t const&gt;, std::span&lt;vector_view_t&gt;)</tt>, one
     *      row per ID. A node without a row refuses with @c shape_mismatch_k, since a missing
     *      row is a wrong softmax.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys, packed and
     *      divided by their partitions.
     *  @param[out] carry Every query row's softmax state, which merges this walk with other branches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] context The calling thread's scratch & telemetry.
     *  @return The query rows attended in @c count.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index does not walk by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, nk::matrix_span<f32_t> outputs,
                                     std::span<softmax_state_t> carry, search_config_t config,
                                     traversal_context_t& context) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_code_t::config_mismatch_k};
        std::size_t const rows_count = queries.extent(0), columns = queries.extent(1);
        if (!head_count || rows_count % head_count) return {{}, status_code_t::shape_mismatch_k};
        if (outputs.extent(0) != rows_count || outputs.extent(1) != columns || carry.size() != rows_count)
            return {{}, status_code_t::config_mismatch_k};
        if (!outputs.is_contiguous()) return {{}, status_code_t::unsupported_layout_k};
        status_t status = status_code_t::success_k;
        search_result_t attended;
        for (std::size_t first = 0; first != rows_count; first += head_count) {
            attention_rescoring<quads_kit_t, elementwise_kit_t, value_lookup_type_> const rescoring {
                queries(nk::range(first, first + head_count), nk::slice),
                value_lookup,
                score_scale,
                outputs(nk::range(first, first + head_count), nk::slice),
                carry.subspan(first, head_count),
                status};
            rescoring.clear();
            auto const [walked, walked_status] = walk_(rescoring.queries.row(0).as_vector(), dummy_fold_t {}, rescoring,
                                                       context, config, static_cast<node_id_t>(size()));
            if (failed(walked_status)) return {attended, walked_status};
            if (failed(status)) return {attended, status};
            rescoring.finish();
            attended.visited_members += walked.visited_members;
            attended.computed_distances += walked.computed_distances;
            attended.restarts += walked.restarts;
            attended.saturated_walks += walked.saturated_walks;
        }
        attended.count = rows_count;
        return {attended, status_code_t::success_k};
    }

    /**
     *  @brief The compile-time form of @c search() & @c search_if(): one whole-traversal
     *      instantiation per @p synchronization_ contract, per fold type and per rescoring type.
     *
     *  A marked contract thus holds no atomics, and @c dummy_fold_t & @c dummy_rescoring_t inline
     *  away. The walk ends against @p bound once it fills - @c context.results itself for every
     *  walk but a filtered one.
     *
     *  @par Statuses:
     *  - @c unsupported_k @p config names a traversal other than a walk.
     *  - @c config_mismatch_k The query's width differs from the index's dimensions.
     *  - @c overflow_k A queue capacity of @p context wraps when rounded.
     */
    template <synchronization_t synchronization_, typename fold_type_, typename rescoring_type_>
        requires a_fold<std::remove_cvref_t<fold_type_>, dot_result_t>
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
                                               first_invisible_id, {});
    }

#pragma endregion Search

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the contract it runs under, and the runs
     *      of the first chunk's rows it hands out to any number of threads.
     *
     *  Vectors and sentinel-padded neighbor rows are what a snapshot holds, and the norms when the
     *  columns hold one for them. A checkpointing reader, not a mutator: concurrent searches keep
     *  running on every mode. Under @c concurrent_k every row is copied under its slot-zero lock,
     *  as writers edit it, and @ref finish probes for settlement across the whole plan, since a
     *  later insert back-linking itself edits an earlier row; the marked modes read plain, and a
     *  violated certification surfaces as @c contended_k rather than a silently torn snapshot.
     */
    struct [[nodiscard]] save_plan_t {

        /** Where the rows go, vetted against the watermark at planning. */
        columns_t columns {};

        /** The contract the caller marked. */
        synchronization_t synchronization = synchronization_t::concurrent_k;

        /** Rows under the watermark at planning. */
        node_id_t visible = 0;

        /** The allocation counter at planning - the growth sensor the probe re-reads. */
        node_id_t allocated = 0;

        /** Rows whose vectors and neighbors arrived, counted through @c atomic_ref. */
        std::size_t covered = 0;

        save_plan_t() noexcept = default;

        /** The index being snapshotted into @p columns under @p synchronization; only
         *  @ref plan_save calls it. */
        save_plan_t(ram_index& index, columns_t columns, synchronization_t synchronization) noexcept
            : columns(columns), synchronization(synchronization), index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), synchronization(other.synchronization), visible(other.visible),
              allocated(other.allocated), covered(other.covered), index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        // The count is this plan's own: a copy would count into a twin `finish` never reads.
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing - no rows claimed, no lock held between tasks - so a dying plan
         *  leaves the index as it found it. */
        ~save_plan_t() noexcept = default;

        /** One task of the plan: one run of the rows under the watermark, inside one chunk. */
        struct task_t {

            /** The task's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_nodes;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_nodes;
        };

        /** Task @p position of the plan; an empty one past the last. */
        task_t operator[](std::size_t position) const noexcept {
            task_t task;
            if (position >= tasks().size()) return task;
            // Every chunk holds a whole number of the first chunk's rows, so these runs tile each.
            std::size_t const run_rows = chunk_schedule_t::nodes_in(0);
            std::size_t const first = position * run_rows;
            std::size_t const bound = min_of(first + run_rows, static_cast<std::size_t>(visible));
            task.position = position;
            task.index_nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            task.source_nodes = task.index_nodes;
            return task;
        }

        /** The tasks the save spans - one per run it reaches; none once spent. */
        indexed_range<save_plan_t> tasks() const noexcept {
            std::size_t const count = index_ ? divide_round_up<std::size_t>(visible, chunk_schedule_t::nodes_in(0))
                                             : 0u;
            return {this, count};
        }

        /**
         *  @brief Copies task @p task 's rows into the plan's own columns: the vectors and their
         *      norms straight from chunk memory, since neither changes once published, and the
         *      neighbors folded on their way into the column.
         *
         *  Every ID past the watermark folds to the sentinel.
         *
         *  @param[in] task The task to copy, as @ref tasks hands them out.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The task holds no rows, or its rows fall outside the columns.
         */
        status_t save(task_t const& task) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            iota_view<node_id_t> const rows = task.index_nodes;
            if (rows.empty()) return status_code_t::config_mismatch_k;
            geometry_t const geometry = index_->geometry();
            std::size_t const connectivity = geometry.connectivity;
            std::size_t const step_slots = rows.size() * connectivity;
            // Chunk rows are padded to whole kit tasks, while checkpoints keep logical rows.
            std::size_t const row_bytes = stored_row_values(geometry) * sizeof(scalar_t);
            std::size_t const first = task.source_nodes.first;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, first, rows.size(), row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, first, rows.size(),
                                                          connectivity * sizeof(node_id_t));
            bytes_span_t const norms_rows = flat_rows(columns.norms, first, rows.size(), sizeof(vector_norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_code_t::config_mismatch_k;

            location_t const location = chunk_schedule_t::locate(rows.first);
            for (std::size_t row = 0; row != rows.size(); ++row)
                std::memcpy(vectors_rows.data() + row * row_bytes,
                            index_->vectors_[location_t {location.chunk, location.offset + row}], row_bytes);
            if (index_->stores_norms() && !norms_rows.empty())
                std::memcpy(norms_rows.data(), &index_->norms_[location], norms_rows.size());

            // The column may sit at any byte of a caller's buffer, so each ID is stored as bytes.
            byte_t* const saved_neighbors = neighbors_rows.data();
            auto const save_neighbor = [saved_neighbors, visible = visible](std::size_t word,
                                                                            node_id_t neighbor) noexcept {
                node_id_t const folded = neighbor < visible ? neighbor : node_id_unknown_k;
                std::memcpy(saved_neighbors + word * sizeof(node_id_t), &folded, sizeof(node_id_t));
            };
            if (synchronization == synchronization_t::concurrent_k)
                for (std::size_t row = 0; row != rows.size(); ++row) {
                    neighbors_ref_t const neighbors = index_->neighbors_for(
                        chunk_schedule_t::locate(static_cast<node_id_t>(rows.first + row)));
                    // A refinement relocks published rows and moves no counter `finish` re-reads.
                    node_id_t const first_neighbor = neighbors.lock();
                    save_neighbor(row * connectivity, first_neighbor);
                    for (candidates_count_t slot = 1; slot != neighbors.capacity(); ++slot)
                        save_neighbor(row * connectivity + slot, neighbors.load_slot(slot));
                    neighbors.unlock(first_neighbor);
                }
            else {
                node_id_t const* const source = index_->neighbors_[location];
                for (std::size_t word = 0; word != step_slots; ++word) save_neighbor(word, source[word]);
            }
            atomic_ref<std::size_t>(covered).store_add(rows.size(), std::memory_order_release);
            return status_code_t::success_k;
        }

        /**
         *  @brief Pronounces the snapshot complete: every chunk's vectors and neighbors arrived,
         *      and under @c concurrent_k the counters rested across the whole plan. A success
         *      spends the plan; a refusal leaves it whole, so the missing tasks can still run.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The tasks covered other than every row.
         *  - @c contended_k Under @c concurrent_k, a counter moved during the plan.
         */
        status_t finish() noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != nodes)
                return status_code_t::config_mismatch_k;
            if (synchronization == synchronization_t::concurrent_k) {
                // A row copy's lock acquired any back-link's unlock, so its dispense shows below.
                // Closed holds no batch: a refine moves no counter, a load the watermark.
                node_id_t const dispensed_now =
                    atomic_ref<node_id_t const>(index_->next_dispensed_id_).load(std::memory_order_relaxed);
                bool const settled =
                    atomic_ref<node_id_t const>(index_->last_visible_id_).load(std::memory_order_relaxed) == visible &&
                    (dispensed_now == visible || dispensed_now == node_id_unknown_k) &&
                    atomic_ref<node_id_t const>(index_->last_allocated_id_).load(std::memory_order_relaxed) ==
                        allocated;
                if (!settled) return status_code_t::contended_k;
            }
            index_ = nullptr;
            return status_code_t::success_k;
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
     *  watermark are refused here, so no task ever writes past them.
     *
     *  @param[out] columns Where the rows go, sized from @ref saved_shape.
     *  @param[in] synchronization The contract the caller marks.
     *
     *  @par Statuses:
     *  - @c contended_k A dispensed batch has yet to publish.
     *  - @c config_mismatch_k A column is too short for the rows, or norms are given to an index
     *    that stores none.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization) noexcept {
        save_plan_t plan {*this, columns, synchronization};
        plan.visible = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        node_id_t const dispensed = atomic_ref<node_id_t const>(next_dispensed_id_).load(std::memory_order_acquire);
        plan.allocated = atomic_ref<node_id_t const>(last_allocated_id_).load(std::memory_order_acquire);
        if (dispensed != plan.visible && dispensed != node_id_unknown_k) return {{}, status_code_t::contended_k};
        std::size_t const rows = static_cast<std::size_t>(plan.visible);
        if (columns.vectors.size() < rows * stored_row_values(geometry_) * sizeof(scalar_t) ||
            columns.neighbors.size() < rows * geometry_.connectivity * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(vector_norms_t))))
            return {{}, status_code_t::config_mismatch_k};
        return {std::move(plan), status_code_t::success_k};
    }

    /**
     *  @brief Serializes every row under the watermark into @p columns on the calling thread, the
     *      one-shot over @ref plan_save for a caller with no threads to fan the tasks out over.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A column is too short for the rows, or norms are given to an index
     *    that stores none.
     *  - @c contended_k A dispensed batch has yet to publish, or under @c concurrent_k a counter
     *    moved during the save.
     */
    status_t save(columns_t columns, synchronization_t synchronization) noexcept {
        auto [plan, status] = plan_save(columns, synchronization);
        if (failed(status)) return status;
        for (typename save_plan_t::task_t const task : plan.tasks())
            if (status_t const saved = plan.save(task); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows claimed after the watermark and the runs they span,
     *      which any number of threads fill, each run filing its keys for @ref finish to publish.
     *
     *  Demands sole ownership of the index. A task copies out of the caller's columns or out of
     *  another index's chunks, straight into chunk rows the plan covered unfilled, and files its
     *  keys before counting its rows. A source's IDs are local to the rows it delivers, so every
     *  run shifts them onto the appended range and folds every other ID onto the sentinel; norms go
     *  beside the vectors, or re-derive from them where the source holds none. Repeated plans
     *  append, and one plan takes several parts when the caller slices its tasks between them. The
     *  plan owns the claim its rows ride in: one that dies before @c finish refills its rows as a
     *  growth leaves them, forgets their keys and releases them, so the next plan appends at the
     *  same base and the rows it half-filled are never published. Nothing a task writes is visible
     *  before @c finish, so a caller holding plans on several indexes publishes every one or none.
     */
    struct [[nodiscard]] load_plan_t {

        /** The first appended ID - the row the claim starts at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows per run: a power of two from the first chunk's rows up to a slab's, so runs tile
         *  every chunk. */
        std::size_t run_rows = 0;

        /** Rows whose vectors, neighbors and keys arrived, counted through @c atomic_ref. */
        std::size_t covered = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base in runs of @p run_rows and the claim its rows ride in -
         *  empty for a plan of no rows; only @ref plan_load calls it. */
        load_plan_t(ram_index& index, node_id_t base, std::size_t run_rows, capacity_claim_t&& claim) noexcept
            : base(base), rows(claim.entries()), run_rows(run_rows), index_(&index), claim_(std::move(claim)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : base(other.base), rows(std::exchange(other.rows, 0)), run_rows(other.run_rows), covered(other.covered),
              index_(std::exchange(other.index_, nullptr)), claim_(std::move(other.claim_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        // The count is this plan's own: a copy would count into a twin `finish` never reads.
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** An unfinished plan's rows go back to a growth's fill and its keys out of the map before
         *  the claim's own death releases them. */
        ~load_plan_t() noexcept {
            if (!index_ || !rows) return;
            index_->refill_unpublished_({base, rows_through_chunk_of_(base + rows - 1u)});
            index_->keyed_.forget_from(base);
        }

        /** One task of the plan: one run of the appended nodes. */
        struct task_t {

            /** The task's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** Where the rows go in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_nodes;

            /** Where they come from in the caller's columns or source index, from its first row. */
            iota_view<node_id_t> source_nodes;
        };

        /** Task @p position of the plan; an empty one past the last. */
        task_t operator[](std::size_t position) const noexcept {
            task_t task;
            if (position >= tasks().size()) return task;
            std::size_t const run = run_of_(base, run_rows) + position;
            u64_t const first = max_of<u64_t>(run_start_(run, run_rows), base);
            u64_t const bound = min_of<u64_t>(run_start_(run + 1, run_rows), u64_t {base} + rows);
            task.position = position;
            task.index_nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            task.source_nodes = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            return task;
        }

        /** The tasks the append spans - one per run it reaches; none once spent. */
        indexed_range<load_plan_t> tasks() const noexcept {
            std::size_t const count = index_ && rows ? run_of_(base + rows - 1, run_rows) - run_of_(base, run_rows) + 1u
                                                     : 0u;
            return {this, count};
        }

        /**
         *  @brief Reads task @p task 's rows out of @p columns straight into chunk memory: the
         *      vectors with their norms, read or derived as they arrive, then the neighbors and the
         *      keys, which the task files itself.
         *
         *  Rows narrower than the chunk's land at the tail of their run and spread forward. Every
         *  ID inside @p source_range is folded onto the appended range and every other onto the
         *  sentinel. A sharded plan drives it one part's slice at a time, its task naming the rows
         *  in that part's row and ID spaces. The task ending the plan fills the rest of its chunk.
         *
         *  @param[in] task The task to fill; its source nodes address @p columns.
         *  @param[in] columns The rows' columns, one part's under a sharded plan: bytes, or column
         *      sources that read a part's file past the page cache.
         *  @param[in] source_range The ID space those rows speak.
         *  @param[in] key_of Row @c r of the task's source nodes takes the key <tt>key_of(r)</tt>.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The task holds no rows, its rows fall outside @p columns, or
         *    they are wider than the index's.
         *  - @c config_mismatch_k @p source_range rebases past the plan's rows.
         */
        template <typename columns_type_, typename key_of_type_>
        status_t load(task_t const& task, columns_type_ const& columns, iota_view<node_id_t> source_range,
                      key_of_type_&& key_of) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            iota_view<node_id_t> const range = task.index_nodes;
            if (range.empty()) return status_code_t::config_mismatch_k;
            geometry_t const geometry = index_->geometry();
            std::size_t const connectivity = geometry.connectivity;
            std::size_t const column_connectivity = columns.connectivity ? columns.connectivity : connectivity;
            if (column_connectivity > connectivity) return status_code_t::config_mismatch_k;
            // The part's rows go at this task's own index rows, so an ID's shift follows from
            // the task rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - task.source_nodes.first + source_range.first);
            // A rebased ID past this plan would name whichever row a later insert fills.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_code_t::config_mismatch_k;

            // Checkpoints keep logical rows, while chunk rows are padded to whole kit tasks.
            std::size_t const count = range.size(), first = task.source_nodes.first;
            std::size_t const stored_bytes = stored_row_values(geometry) * sizeof(scalar_t);
            std::size_t const padded_bytes = padded_row_values(geometry) * sizeof(scalar_t);
            location_t const location = chunk_schedule_t::locate(range.first);
            bytes_span_t const vectors = bytes_of(index_->vectors_[location], count * padded_row_values(geometry));
            if (status_t const read = copy_entries(columns.vectors, first, count, stored_bytes,
                                                   vectors.last(count * stored_bytes));
                failed(read))
                return read;
            expand_entries_forward(vectors, count, stored_bytes, padded_bytes, byte_t {0});
            if (index_->stores_norms()) {
                if (!columns.norms.empty()) {
                    if (status_t const read = copy_entries(columns.norms, first, count, sizeof(vector_norms_t),
                                                           bytes_of(&index_->norms_[location], count));
                        failed(read))
                        return read;
                }
                else
                    for (std::size_t offset_in_chunk = 0; offset_in_chunk != count; ++offset_in_chunk) {
                        location_t const row {location.chunk, location.offset + offset_in_chunk};
                        index_->norm_ref_for(row) = index_->norms(index_->vector_for(row));
                    }
            }

            std::span<node_id_t> const copied {index_->neighbors_[location], count * connectivity};
            bytes_span_t const neighbor_rows = bytes_of(copied.data(), copied.size());
            std::size_t const column_row_bytes = column_connectivity * sizeof(node_id_t);
            if (status_t const read = copy_entries(columns.neighbors, first, count, column_row_bytes,
                                                   neighbor_rows.last(count * column_row_bytes));
                failed(read))
                return read;
            expand_entries_forward(neighbor_rows, count, column_row_bytes, connectivity * sizeof(node_id_t),
                                   byte_t {0xFF});
            rebase_neighbors(copied, source_range, shift, node_id_unknown_k);
            compact_neighbors(copied, connectivity, range.first, node_id_unknown_k);
            for (node_id_t node = range.first; node != range.bound; ++node)
                index_->keyed_.assign(node, key_of(static_cast<node_id_t>(first + node - range.first)));
            settle_run_(range);
            return status_code_t::success_k;
        }

        /**
         *  @brief The same copy out of @p source 's own chunks rather than columns, as a reshard
         *      drives it, one source shard's slice at a time, its task naming the rows in that
         *      shard's row and ID spaces.
         *
         *  Both indexes keep padded rows of one geometry, so every run moves whole with its norms
         *  and zero tails, and the neighbors rebase in this index's own chunk. The copy reads
         *  @p source without locks, so nothing may insert into, refine or clear it while the task
         *  runs.
         *
         *  @param[in] task The task to fill; its source nodes address @p source.
         *  @param[in] source The index the rows come from, of this one's geometry and norms.
         *  @param[in] source_range The ID space the source's rows speak.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The task is empty or uneven, @p source differs in geometry or
         *    norms, or the task reads past its published rows.
         *  - @c config_mismatch_k @p source_range rebases past the plan's rows.
         */
        status_t load(task_t const& task, ram_index const& source, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            iota_view<node_id_t> const range = task.index_nodes;
            geometry_t const geometry = index_->geometry();
            if (range.empty() || range.size() != task.source_nodes.size() || source.geometry() != geometry ||
                source.stores_norms() != index_->stores_norms() || task.source_nodes.bound > source.size())
                return status_code_t::config_mismatch_k;
            node_id_t const shift = range.first - task.source_nodes.first + source_range.first;
            if (shift + source_range.size() > base + rows) return status_code_t::config_mismatch_k;
            std::size_t const connectivity = geometry.connectivity;
            std::size_t const row_bytes = padded_row_values(geometry) * sizeof(scalar_t);

            // The task sits in one chunk here, while the source's chunks start elsewhere.
            for (node_id_t writing = range.first, reading = task.source_nodes.first; writing != range.bound;) {
                location_t const into = chunk_schedule_t::locate(writing), from = chunk_schedule_t::locate(reading);
                node_id_t const run = static_cast<node_id_t>(
                    min_of<std::size_t>(range.bound - writing, chunk_schedule_t::nodes_in(from.chunk) - from.offset));
                std::memcpy(index_->vectors_[into], source.vectors_[from], run * row_bytes);
                if (index_->stores_norms())
                    std::memcpy(&index_->norms_[into], &source.norms_[from], run * sizeof(vector_norms_t));
                std::memcpy(index_->neighbors_[into], source.neighbors_[from], run * connectivity * sizeof(node_id_t));
                for (node_id_t offset = 0; offset != run; ++offset)
                    index_->keyed_.assign(writing + offset, source.keyed_[reading + offset]);
                writing += run;
                reading += run;
            }

            std::span<node_id_t> const copied {index_->neighbors_[chunk_schedule_t::locate(range.first)],
                                               range.size() * connectivity};
            rebase_neighbors(copied, source_range, shift, node_id_unknown_k);
            compact_neighbors(copied, connectivity, range.first, node_id_unknown_k);
            settle_run_(range);
            return status_code_t::success_k;
        }

        /**
         *  @brief Publishes the appended nodes once every run's vectors, neighbors and keys
         *      arrived: the claim consumed first, then identity, then the watermark last with a
         *      release, which holds the keys every run filed.
         *
         *  The identity releases too, and moves only an open dispenser: a closed one stays closed
         *  for its closer to reopen past the rows. A success spends the plan; a refusal leaves it
         *  whole, so the missing tasks can still run.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The tasks covered other than every row.
         */
        status_t finish() noexcept {
            if (!index_) return status_code_t::uninitialized_k;
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != rows)
                return status_code_t::config_mismatch_k;
            if (rows) {
                claim_.consume();
                node_id_t dispensed = base;
                atomic_ref<node_id_t>(index_->next_dispensed_id_)
                    .compare_exchange_strong(dispensed, static_cast<node_id_t>(base + rows), std::memory_order_release,
                                             std::memory_order_relaxed);
                atomic_ref<node_id_t>(index_->last_visible_id_)
                    .store(static_cast<node_id_t>(base + rows), std::memory_order_release);
            }
            index_ = nullptr;
            return status_code_t::success_k;
        }

      private:

        /** What every run does once its rows arrived: files their keys, which the map takes from
         *  any number of runs at once, fills the rest of the chunk when the run ends the plan, and
         *  counts the rows. */
        void settle_run_(iota_view<node_id_t> range) noexcept {
            index_->keyed_.publish(range);
            if (range.bound == static_cast<std::size_t>(base) + rows)
                index_->refill_unpublished_({range.bound, rows_through_chunk_of_(range.bound - 1u)});
            atomic_ref<std::size_t>(covered).store_add(range.size(), std::memory_order_release);
        }

        /** The index appended to; null in the empty plan a refused @ref plan_load hands back,
         *  and once @c finish published. */
        ram_index* index_ = nullptr;

        /** The rows this plan appends, released by their own death unless consumed. */
        capacity_claim_t claim_;
    };

    /**
     *  @brief Opens a load of @p rows after the watermark, sourced per task either from
     *      columns or from another index.
     *
     *  It covers their chunks unfilled, then claims them through the ordinary capacity protocol, so
     *  a preceding @c reserve is consumed rather than duplicated and chunks grow only on a genuine
     *  deficit. Runs hold the power of two nearest the rows a request of @p request_bytes holds of
     *  the widest row, at least the first chunk's rows and at most a slab's. Refuses an insert
     *  mid-flight and rows past the ID space. The sharded index closes the dispenser first, so no
     *  insert arrives before @c finish; a direct caller keeps insertions out itself.
     *
     *  @param[in] rows Rows to append.
     *  @param[in] request_bytes Bytes the drive under the source takes in one request; zero for a
     *      source in memory, which runs the first chunk's rows at a time.
     *
     *  @par Statuses:
     *  - @c contended_k A dispensed batch has yet to publish, or another claim won first.
     *  - @c overflow_k The rows pass the ID space, the reserved ceiling or the chunk directory.
     *  - @c config_mismatch_k or @c unknown_kind_k A growth meets a malformed geometry or kernel.
     */
    expected<load_plan_t> plan_load(std::size_t rows, std::size_t request_bytes = 0) noexcept {
        node_id_t const visible = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        std::size_t const widest_row_bytes = max_of(padded_row_values(geometry_) * sizeof(scalar_t),
                                                    geometry_.connectivity * sizeof(node_id_t));
        std::size_t const request_rows = widest_row_bytes ? request_bytes / widest_row_bytes : 0;
        std::size_t const below = std::bit_floor(request_rows);
        std::size_t const nearest_rows = request_rows - below > 2 * below - request_rows ? 2 * below : below;
        std::size_t const run_rows = min_of(max_of(nearest_rows, chunk_schedule_t::nodes_in(0)),
                                            chunk_schedule_t::nodes_in(chunk_schedule_t::doubling_chunks_k));
        if (!rows) return {load_plan_t {*this, visible, run_rows, {}}, status_code_t::success_k};
        node_id_t const dispensed = atomic_ref<node_id_t>(next_dispensed_id_).load(std::memory_order_acquire);
        if (dispensed != visible && dispensed != node_id_unknown_k) return {{}, status_code_t::contended_k};
        if (rows > max_node_count_k - visible) return {{}, status_code_t::overflow_k};
        if (status_t const vetted = vet_growth_(); failed(vetted)) return {{}, vetted};
        std::size_t const loaded_rows = static_cast<std::size_t>(visible) + rows;
        if (loaded_rows > atomic_ref<std::size_t const>(ceiling_nodes_).load(std::memory_order_relaxed))
            return {{}, status_code_t::overflow_k};

        // The tasks write every byte of their rows and fill the rest of the last chunk, so the
        // claim's growth finds these chunks published and leaves them unfilled.
        status_t covered = vectors_.cover_uninitialized(loaded_rows, padded_row_values(geometry_));
        if (succeeded(covered) && stores_norms()) covered = norms_.cover_uninitialized(loaded_rows);
        if (succeeded(covered)) covered = neighbors_.cover_uninitialized(loaded_rows, geometry_.connectivity);
        auto [capacity_claim, status] = succeeded(covered) ? claim(rows) : expected<capacity_claim_t> {{}, covered};
        if (succeeded(status) && capacity_claim.base() == visible)
            return {load_plan_t {*this, visible, run_rows, std::move(capacity_claim)}, status_code_t::success_k};
        // A refused plan leaves what it covered as a growth would, and a claim that won elsewhere
        // dies, releasing its rows.
        refill_unpublished_({visible, rows_through_chunk_of_(loaded_rows - 1u)});
        return {{}, failed(status) ? status : status_code_t::contended_k};
    }

    /**
     *  @brief Appends the whole source on the calling thread under @p keys, one per row - the
     *      one-shot over @ref plan_load for a caller with no threads to fan the tasks out over;
     *      columns shorter than @p rows are refused before a row is claimed.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A column or @p keys is shorter than @p rows, its neighbor rows are
     *    wider than the index's, or norms are given to an index that stores none.
     *  - @c contended_k A dispensed batch has yet to publish, or another claim won first.
     *  - @c overflow_k The rows pass the ID space, the reserved ceiling or the chunk directory.
     *  - @c config_mismatch_k @p source_range rebases past the loaded rows.
     */
    status_t load(columns_view_t columns, keys_view_t keys, std::size_t rows,
                  iota_view<node_id_t> source_range) noexcept {
        std::size_t const column_connectivity = columns.connectivity ? columns.connectivity : geometry_.connectivity;
        if (keys.size() < rows || columns.vectors.size() < rows * stored_row_values(geometry_) * sizeof(scalar_t) ||
            column_connectivity > geometry_.connectivity ||
            columns.neighbors.size() < rows * column_connectivity * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(vector_norms_t))))
            return status_code_t::config_mismatch_k;
        auto [plan, status] = plan_load(rows);
        if (failed(status)) return status;
        for (typename load_plan_t::task_t const task : plan.tasks())
            if (status_t const loaded = plan.load(task, columns, source_range,
                                                  [&](node_id_t row) noexcept { return keys[row]; });
                failed(loaded))
                return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

  private:

#pragma region Storage

    /** Trampoline for @c dense::norms with hardened types - zeroed moments, without
     *  touching the vector, when the rows keep none. */
    inline vector_norms_t norms(vector_view_t vec) const noexcept {
        return stores_norms() ? dense::norms<quads_kit_t>(vec) : vector_norms_t {};
    }

    /** Trampoline for @c dense::distances with hardened types. */
    inline quad<distance_t> distances(vector_view_t query, vector_norms_t query_norms, quad<vector_view_t> candidates,
                                      quad<vector_norms_t> candidates_norms) const noexcept {
        return dense::distances<quads_kit_t>(kernel_, query, query_norms, candidates, candidates_norms);
    }

    /** Trampoline for @c dense::scores with hardened types - the walk's scoring,
     *  whose dots a fold reads. */
    inline scores_result<dot_result_t> scores(vector_view_t query, vector_norms_t query_norms,
                                              quad<vector_view_t> candidates,
                                              quad<vector_norms_t> candidates_norms) const noexcept {
        return dense::scores<quads_kit_t>(kernel_, query, query_norms, candidates, candidates_norms);
    }

    /** As @c scores, streaming @p next_candidates in through the prefetcher. */
    inline scores_result<dot_result_t> scores(vector_view_t query, vector_norms_t query_norms,
                                              quad<vector_view_t> candidates, quad<vector_norms_t> candidates_norms,
                                              quad<vector_view_t> next_candidates) const noexcept {
        return dense::scores<quads_kit_t>( //
            kernel_, query, query_norms, candidates, candidates_norms, next_candidates, prefetcher_);
    }

    /** Scores @p query against four consecutive entries of the cached @p vectors & @p norms
     *  arrays starting at @p start; the gather cursor sticks at @p count - 1, so short tails
     *  repeat the last entry and the kernel always streams four valid lanes. */
    inline quad<distance_t> distances_via_quads(                   //
        vector_view_t query, vector_norms_t query_norms,           //
        vector_view_t const* vectors, vector_norms_t const* norms, //
        candidates_count_t start, candidates_count_t count) const noexcept {
        quad<vector_view_t> lane_vectors;
        quad<vector_norms_t> lane_norms;
        for (candidates_count_t lane = 0, source = start; lane != lane_vectors.size(); ++lane) {
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
        inline candidates_count_t capacity() const noexcept { return static_cast<candidates_count_t>(row_.size()); }

        /** Relaxed atomic read of slot @p index; slot 0 belongs to @c peek_first / @c lock. */
        inline node_id_t load_slot(candidates_count_t index) const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return atomic_ref<node_id_t>(row_[index]).load(std::memory_order_relaxed);
            else return row_[index];
        }

        /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current
         *  lock holder. */
        inline void store_slot(candidates_count_t index, node_id_t value) const noexcept {
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
            candidates_count_t live = 1;
            while (live != capacity() && row_[live] != node_id_unknown_k) ++live;
            for (candidates_count_t slot = live; slot != capacity(); ++slot)
                if (row_[slot] != node_id_unknown_k) return false;
            for (candidates_count_t slot = 1; slot != live; ++slot)
                for (candidates_count_t earlier = 0; earlier != slot; ++earlier)
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
    inline quad<vector_view_t> vectors_quad_of(neighbors_type_ const& neighbors, candidates_count_t start,
                                               node_id_t first_invisible_id,
                                               quad<vector_view_t> lane_vectors) const noexcept {
        if constexpr (prefetcher_t::prefetches_k) {
            candidates_count_t const slots_limit = min_of<candidates_count_t>(start + lane_vectors.size(),
                                                                              neighbors.capacity());
            candidates_count_t lane = 0;
            for (candidates_count_t slot = start; slot < slots_limit; ++slot) {
                node_id_t const neighbor_id = neighbors.load_slot(slot);
                if (neighbor_id == node_id_unknown_k) break;
                if (neighbor_id >= first_invisible_id) continue;
                lane_vectors[lane++] = vector_for(chunk_schedule_t::locate(neighbor_id));
            }
        }
        return lane_vectors;
    }

    /** Streams the neighbor row at @p location toward L2, for a walk holding it a hop ahead. */
    inline void prefetch_neighbors_(location_t const& location) const noexcept {
        prefetcher_.prefetch(neighbors_[location], neighbors_.columns() * sizeof(node_id_t));
    }

    /** Streams the vector row and norms at @p location toward L2, for a scoring draws ahead. */
    inline void prefetch_vector_(location_t const& location) const noexcept {
        prefetcher_.prefetch(vectors_[location], vectors_.columns() * sizeof(scalar_t));
        if (stores_norms()) prefetcher_.prefetch(&norms_[location], sizeof(vector_norms_t));
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
        if (status_t const vetted = vet_growth_(); failed(vetted)) return {0, vetted};

        std::size_t const ceiling_rows = atomic_ref<std::size_t const>(ceiling_nodes_).load(std::memory_order_relaxed);
        std::size_t added_capacity = 0;
        while (added_capacity < minimal_capacity) {
            std::size_t const hosted_rows = chunk_schedule_t::nodes_before(chunks_count_);
            if (chunks_count_ == max_chunks_k || hosted_rows >= ceiling_rows)
                return {added_capacity, status_code_t::overflow_k};

            std::size_t const covered_rows = chunk_schedule_t::nodes_before(chunks_count_ + 1u);
            // Norms exist only when the finalize reads them, which a dot does on compensated kits
            // alone. The all-ones neighbors fill is load-bearing: a sentinel leading slot reads as
            // unpublished, and sentinel tails terminate short rows, so no counts are stored.
            static_assert(node_id_unknown_k == static_cast<node_id_t>(~node_id_t {0}));
            status_t covered = vectors_.cover(covered_rows, padded_row_values(geometry_));
            if (succeeded(covered) && stores_norms()) covered = norms_.cover(covered_rows);
            if (succeeded(covered)) covered = neighbors_.cover(covered_rows, geometry_.connectivity, 0xFF);
            if (succeeded(covered)) covered = keyed_.cover(covered_rows);
            if (failed(covered)) return {added_capacity, covered};

            // A chunk crossing the ceiling hosts only the rows below it.
            added_capacity += min_of(chunk_schedule_t::nodes_in(chunks_count_), ceiling_rows - hosted_rows);
            ++chunks_count_;
            // The all-ones ID is the sentinel: account one row less in the final chunk, so no
            // claim can dispense it and the dispenser never wraps.
            if (chunks_count_ == max_chunks_k) added_capacity -= 1;
        }
        return {added_capacity, status_code_t::success_k};
    }

    /** Refuses what every growth refuses, before any chunk is allocated: constructors allocate
     *  nothing, so the first growth or load is where a kernel the kits cannot finalize, or a
     *  malformed geometry, surfaces. */
    status_t vet_growth_() const noexcept {
        if (!kernel_supported<quads_kit_t>(kernel_)) return status_code_t::unknown_kind_k;
        // Sub-byte rows fill whole storage values: a kernel's tail loads would drop a ragged tail.
        if (!geometry_.dimensions || geometry_.dimensions % nk::dimensions_per_value<scalar_t>() ||
            !std::has_single_bit(static_cast<u32_t>(geometry_.connectivity)))
            return status_code_t::config_mismatch_k;
        return status_code_t::success_k;
    }

    /** Refills the rows of @p nodes each column's published chunks hold as a growth leaves them:
     *  zero vectors and norms, and sentinel neighbors. */
    void refill_unpublished_(iota_view<std::size_t> nodes) noexcept {
        std::size_t const vector_bytes = vectors_.columns() * sizeof(scalar_t);
        std::size_t const neighbor_bytes = neighbors_.columns() * sizeof(node_id_t);
        std::size_t const vectors_bound = min_of(nodes.bound, vectors_.capacity());
        std::size_t const norms_bound = stores_norms() ? min_of(nodes.bound, norms_.capacity()) : 0;
        std::size_t const neighbors_bound = min_of(nodes.bound, neighbors_.capacity());
        for (std::size_t node = nodes.first; node < nodes.bound;) {
            location_t const location = chunk_schedule_t::locate(node);
            std::size_t const run = min_of(nodes.bound - node,
                                           chunk_schedule_t::nodes_in(location.chunk) - location.offset);
            if (node < vectors_bound) std::memset(static_cast<void*>(vectors_[location]), 0, run * vector_bytes);
            if (node < norms_bound) std::memset(static_cast<void*>(&norms_[location]), 0, run * sizeof(vector_norms_t));
            if (node < neighbors_bound) std::memset(neighbors_[location], 0xFF, run * neighbor_bytes);
            node += run;
        }
    }

    /** One past the last row of the chunk holding @p node. */
    static std::size_t rows_through_chunk_of_(std::size_t node) noexcept {
        return chunk_schedule_t::nodes_before(chunk_schedule_t::locate(node).chunk + 1u);
    }

    /** The run holding row @p node when runs hold up to @p run_rows rows: they double from the
     *  first chunk's rows up to @p run_rows, then stay, so each lies inside one chunk and
     *  together they tile every chunk. */
    static std::size_t run_of_(std::size_t node, std::size_t run_rows) noexcept {
        u64_t const first_rows = chunk_schedule_t::nodes_in(0);
        u64_t const doubling_rows = 2 * u64_t {run_rows} - first_rows;
        std::size_t const doubling_runs =
            static_cast<std::size_t>(std::bit_width(run_rows) - std::bit_width(first_rows)) + 1u;
        if (node < doubling_rows)
            return static_cast<std::size_t>(std::bit_width(node + first_rows) - std::bit_width(first_rows));
        return doubling_runs + static_cast<std::size_t>((node - doubling_rows) / run_rows);
    }

    /** The first row of run @p run under @ref run_of_ 's schedule, in 64 bits, as the run past
     *  the last may pass the ID space. */
    static u64_t run_start_(std::size_t run, std::size_t run_rows) noexcept {
        u64_t const first_rows = chunk_schedule_t::nodes_in(0);
        std::size_t const doubling_runs =
            static_cast<std::size_t>(std::bit_width(run_rows) - std::bit_width(first_rows)) + 1u;
        if (run <= doubling_runs) return (first_rows << run) - first_rows;
        return 2 * u64_t {run_rows} - first_rows + u64_t {run - doubling_runs} * run_rows;
    }

    /**
     *  @brief Claims storage capacity for @p count new nodes, growing chunks when needed.
     *  @return The first row of the claim - only meaningful on @c success_k, and rarely needed:
     *      identity is dispensed separately.
     *
     *  Spins while other threads are mid-growth, covering all interleavings of the capacity state
     *  machine. Claims @b storage only, never identity: node IDs are dispensed per batch from
     *  @c next_dispensed_id_ by @c dispense_. On success, enough sentinel-filled rows exist for
     *  @p count dispenses, and the caller must roll forward to wiring & publishing them all -
     *  claimed capacity cannot be released once IDs are dispensed against it, or the
     *  visibility watermark would stall forever. @ref dense_ram_index_concurrency
     */
    expected<node_id_t> claim_capacity(std::size_t count) noexcept {
        // Rejected before the counter dance, so it neither grows the index in vain nor overflows
        // the signed capacity arithmetic below.
        if (count > atomic_ref<std::size_t const>(ceiling_nodes_).load(std::memory_order_relaxed))
            return {node_id_unknown_k, status_code_t::overflow_k};
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
                // Only the deficit grows, so a claim the ceiling's truncated chunk fits succeeds.
                auto [added_capacity,
                      chunk_status] = create_next_chunk(static_cast<std::size_t>(-new_remaining_capacity));
                if (chunk_status != status_code_t::success_k) {
                    // Revert the claim, releasing any partially added chunks along the way.
                    atomic_ref<std::ptrdiff_t>(remaining_capacity_)
                        .store_add(count + added_capacity, std::memory_order_release);
                    return {node_id_unknown_k, chunk_status};
                }
                // The take releases too: a claimer that fit on the counter before the growth but whose rows this
                // take pushes into the new chunk reaches its pointer through this word alone.
                node_id_t const first_id =
                    atomic_ref<node_id_t>(last_allocated_id_).fetch_add(count, std::memory_order_acq_rel);
                atomic_ref<std::ptrdiff_t>(remaining_capacity_).store_add(added_capacity, std::memory_order_release);
                return {first_id, status_code_t::success_k};
            }
            // We may be lucky and the space is already reserved for the whole batch.
            else if (new_remaining_capacity >= 0) {
                return {atomic_ref<node_id_t>(last_allocated_id_).fetch_add(count, std::memory_order_acquire),
                        status_code_t::success_k};
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
     *  @param[in,out] candidates Search results, ascending by distance to the query; their ties
     *      reorder in place on the hashed ring around @p query_id.
     *  @param[in] query_id The row being linked.
     *  @param[in] limit Ceiling on the kept neighbors - typically the connectivity.
     *  @param[out] out Receives the kept scored nodes, diverse picks first; its live shape is the fill.
     *  @param[in,out] context Scratch for the selection, plus the telemetry counters.
     *  @return The number of distance computations performed, for the caller's telemetry.
     */
    std::size_t select_diverse(results_array_t& candidates, node_id_t query_id, std::size_t limit, selection_t& out,
                               traversal_context_t& context) const noexcept {
        std::size_t const computed_distances = select_diverse_via_quads(candidates, query_id, limit, out, context);
        fill_selection(to_span(candidates), out, limit);
        return computed_distances;
    }

    /** Appends the nearest rejected candidates of the ascending @p candidates to the diverse prefix
     *  in @p out until @p limit or the candidates run out, past the query's other duplicates once
     *  the prefix keeps one. The prefix is a subsequence of them, so one merge-walk tells a kept
     *  candidate from a rejected one. */
    static void fill_selection(std::span<scored_node_t const> candidates, selection_t& out,
                               std::size_t limit) noexcept {
        std::size_t const diverse_count = out.size();
        if (diverse_count == limit || nk::failed(out.resize({limit}))) return;
        bool kept_zero = false;
        for (std::size_t index = 0; index != diverse_count; ++index) kept_zero |= out[index].distance == 0;
        std::size_t filled_count = diverse_count;
        std::size_t matched = 0;
        for (scored_node_t const candidate : candidates) {
            if (filled_count == limit) break;
            if (matched != diverse_count && out[matched].node_id == candidate.node_id) ++matched;
            else if (!(kept_zero && candidate.distance == 0)) out[filled_count++] = candidate;
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] nk::status_t const shrunk = out.resize({filled_count});
    }

    /** The diverse prefix of @c select_diverse: pair distances computed on the fly with the 4-way
     *  kernels, the greedy scan deciding as they arrive, at most one zero-distance candidate kept.
     *  The kept prefix's views & norms ride the staging surface, free during any selection. */
    std::size_t select_diverse_via_quads(results_array_t& candidates, node_id_t query_id, std::size_t limit,
                                         selection_t& out, traversal_context_t& context) const noexcept {
        if (nk::failed(out.resize({limit}))) return 0;
        // Equal distances order on the hashed ring around the row, so a duplicate class links in a
        // ring, not a star.
        scored_node_t* const candidates_data = candidates.data();
        for (std::size_t index = 1; index < candidates.size(); ++index) {
            scored_node_t const candidate = candidates_data[index];
            u32_t const ring = ring_offset(candidate.node_id, query_id);
            std::size_t position = index;
            for (; position && candidates_data[position - 1].distance == candidate.distance &&
                   ring_offset(candidates_data[position - 1].node_id, query_id) > ring;
                 --position)
                candidates_data[position] = candidates_data[position - 1];
            candidates_data[position] = candidate;
        }
        vector_view_t* const kept_vectors = context.staged_vectors.data();
        vector_norms_t* const kept_norms = context.staged_norms.data();
        candidates_count_t kept_count = 0;
        bool kept_zero = false;
        std::size_t computed_distances = 0;
        for (std::size_t index = 0; index != candidates.size() && kept_count != limit; ++index) {
            scored_node_t const candidate = candidates_data[index];
            // The candidate a quad of draws ahead streams in while this one is judged.
            if (index + quad_lanes_k < candidates.size())
                prefetch_vector_(chunk_schedule_t::locate(candidates_data[index + quad_lanes_k].node_id));
            // A second duplicate of the row would only repeat the first one's links.
            if (kept_zero && candidate.distance == 0) continue;
            location_t const candidate_location = chunk_schedule_t::locate(candidate.node_id);
            vector_view_t const candidate_vector = vector_for(candidate_location);
            vector_norms_t const candidate_norms = norm_for(candidate_location);
            // 4-way batches over the kept neighbors' cached views; short tails repeat the last
            // one. A batch may score past the first violation, still cheaper than four serial.
            candidates_count_t kept = 0;
            while (kept != kept_count) {
                quad<distance_t> const kept_distances = distances_via_quads( //
                    candidate_vector, candidate_norms, kept_vectors, kept_norms, kept, kept_count);
                candidates_count_t const lanes_count = min_of<candidates_count_t>(kept_count - kept, 4u);
                computed_distances += lanes_count;
                candidates_count_t losses = 0;
                for (candidates_count_t lane = 0; lane != lanes_count; ++lane)
                    losses += kept_distances[lane] < candidate.distance;
                if (losses) break; // Some kept neighbor is closer than the query - not diverse.
                kept += lanes_count;
            }
            // Surviving every batch is the diversity verdict itself - no flag to thread through.
            if (kept != kept_count) continue;
            kept_zero |= candidate.distance == 0;
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
     *      4-way sweeps feeding the sorted re-selection candidates with real node IDs, the
     *      appended node entering on its reused symmetric score.
     *  @return The distances computed, for the caller's telemetry.
     */
    std::size_t stage_scores_via_quads(vector_view_t neighbor_vector, vector_norms_t neighbor_norms,
                                       node_id_t new_node_id, distance_t appended_node_score,
                                       traversal_context_t& context) const noexcept {
        results_array_t& refining_candidates = context.refining_candidates;
        candidates_count_t const count = static_cast<candidates_count_t>(context.staged_ids.size());
        candidates_count_t const candidates_limit = count + 1;
        refining_candidates.insert(scored_node_t(new_node_id, appended_node_score), candidates_limit);
        std::size_t computed = 0;
        for (candidates_count_t scored = 0; scored != count;) {
            quad<distance_t> const staged_distances = distances_via_quads( //
                neighbor_vector, neighbor_norms, context.staged_vectors.data(), context.staged_norms.data(), scored,
                count);
            candidates_count_t const lanes_count = min_of<candidates_count_t>(count - scored, 4u);
            for (candidates_count_t lane = 0; lane != lanes_count; ++lane)
                refining_candidates.insert(scored_node_t(context.staged_ids[scored + lane], staged_distances[lane]),
                                           candidates_limit);
            scored += lanes_count;
            computed += lanes_count;
        }
        return computed;
    }

    /**
     *  @brief Back-links the node: for every selected neighbor, locks that neighbor's slots via the
     *      slot-0 sentinel exchange and inserts the node - appending when there is room, otherwise
     *      re-selecting the diverse best among the existing neighbors plus the appended node.
     *
     *  A neighbor already holding the node is left as it stands, so a refinement keeping an
     *  existing neighbor never duplicates it.
     *
     *  Locking discipline: at most one neighbor lock is held at any moment, and never while
     *  spinning on the watermark - callers invoke this before their watermark turn, so every lock
     *  is released by the time the turn's spin begins. All slot writes happen under the lock, and
     *  the unlocking release-store of slot 0 is what republishes the edited neighbors.
     *
     *  @param[in] new_node_id The node whose neighbors are back-linked to it.
     *  @param[in] selected The node's chosen neighbors with their recorded query distances -
     *      those score the appended node for free, since the metrics are symmetric.
     *  @param[in,out] context Scratch for the re-selection, plus the telemetry counters.
     */
    void back_link_neighbors(node_id_t new_node_id, selection_t const& selected,
                             traversal_context_t& context) noexcept {
        results_array_t& refining_candidates = context.refining_candidates;
        std::size_t const node_connectivity = connectivity();

        for (candidates_count_t selected_index = 0; selected_index != selected.size(); ++selected_index) {
            node_id_t const neighbor_id = selected[selected_index].node_id;
            location_t const neighbor_location = chunk_schedule_t::locate(neighbor_id);
            neighbors_ref_t const neighbors = neighbors_for(neighbor_location);

            // Lock by exchanging the sentinel in; the displaced slot-0 value is ours to restore.
            node_id_t const first_slot_value = neighbors.lock();

            // The next row's lock word heads here with write intent while this row is edited.
            if (selected_index + 1 != selected.size())
                prefetcher_.promote(neighbors_[chunk_schedule_t::locate(selected[selected_index + 1].node_id)],
                                    sizeof(node_id_t));

            // No neighbors is encoded as a self-loop: the node becomes the only neighbor.
            if (first_slot_value == neighbor_id) {
                neighbors.unlock(new_node_id);
                continue;
            }

            // One scan of the live prefix finds both its extent and a node already wired.
            candidates_count_t const slots_count = neighbors.capacity();
            candidates_count_t existing_count = 1;
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
            // neighbors and the appended node, scored against the neighbor itself, under the lock.
            vector_view_t const neighbor_vector = vector_for(neighbor_location);
            vector_norms_t const neighbor_norms = norm_for(neighbor_location);

            // Staged at their own slot positions, the lock-displaced slot-0 value first. The
            // appended node stays out: its symmetric score was already recorded at selection time.
            if (nk::failed(context.staged_ids.resize({slots_count}))) {
                neighbors.unlock(first_slot_value);
                ++context.saturated_walks;
                continue;
            }
            // Every staged row streams in before the first quad scores, so their misses overlap.
            location_t const first_location = chunk_schedule_t::locate(first_slot_value);
            prefetch_vector_(first_location);
            context.staged_ids[0] = first_slot_value;
            context.staged_vectors[0] = vector_for(first_location);
            context.staged_norms[0] = norm_for(first_location);
            for (candidates_count_t slot = 1; slot != slots_count; ++slot) {
                node_id_t const existing_id = neighbors.load_slot(slot);
                location_t const existing_location = chunk_schedule_t::locate(existing_id);
                prefetch_vector_(existing_location);
                context.staged_ids[slot] = existing_id;
                context.staged_vectors[slot] = vector_for(existing_location);
                context.staged_norms[slot] = norm_for(existing_location);
            }

            // The candidates are every neighbor plus the appended node; that "+ 1" lives in the
            // sorted candidates alone, never in an operand shape.
            refining_candidates.clear();
            selection_t& refined = context.refined_neighbors;
            std::size_t refine_distances = 0;
            refine_distances = stage_scores_via_quads(neighbor_vector, neighbor_norms, new_node_id,
                                                      selected[selected_index].distance, context);
            refine_distances += select_diverse_via_quads(refining_candidates, neighbor_id, node_connectivity, refined,
                                                         context);
            context.computed_distances += refine_distances;
            context.computed_distances_in_reverse_refines += refine_distances;

            // Rewrite the tail slots, sentinel-terminate any shrink, and unlock with the new head.
            // A non-empty refined fill always holds: the heuristic never rejects the nearest candidate.
            candidates_count_t const refined_count = static_cast<candidates_count_t>(refined.size());
            for (candidates_count_t slot = 1; slot < refined_count; ++slot)
                neighbors.store_slot(slot, refined[slot].node_id);
            for (candidates_count_t slot = refined_count; slot < slots_count; ++slot)
                neighbors.store_slot(slot, node_id_unknown_k);
            neighbors.unlock(refined[0].node_id);
        }
    }

    /** Rows claimed by batches and loads, published or not - what a shard set measures its bound
     *  against. */
    std::size_t claimed_() const noexcept {
        return atomic_ref<node_id_t const>(last_allocated_id_).load(std::memory_order_acquire);
    }

    /** Dispenses one range of IDs for every row @p claim holds with one bounded add, and
     *  consumes the claim; a batch publishes that range whole. Every ID it hands out must be
     *  published, so nothing after it may fail. A closed dispenser answers @c contended_k and
     *  leaves the claim whole. */
    expected<iota_view<node_id_t>> dispense_(capacity_claim_t&& claim) noexcept {
        node_id_t const rows = static_cast<node_id_t>(claim.entries());
        // The `acq_rel` order chains the hosting chunks' growth through the all-RMW dispenser to this thread.
        // Claimed capacity keeps an open dispenser under the sentinel; only a closed one refuses.
        atomic_ref<node_id_t> const dispenser {next_dispensed_id_};
        node_id_t const first = dispenser.fetch_add_if_at_most(rows, node_id_unknown_k, std::memory_order_acq_rel);
        if (first == node_id_unknown_k) return {{}, status_code_t::contended_k};
        claim.consume();
        return {{first, static_cast<node_id_t>(first + rows)}, status_code_t::success_k};
    }

    /** Closes the dispenser at the watermark: one compare-exchange that succeeds only while every
     *  dispensed ID is published, after which every dispense refuses until @c reopen_. */
    status_t close_() noexcept {
        node_id_t visible = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        return atomic_ref<node_id_t>(next_dispensed_id_)
                       .compare_exchange_strong(visible, node_id_unknown_k, std::memory_order_acq_rel,
                                                std::memory_order_relaxed)
                   ? status_code_t::success_k
                   : status_code_t::contended_k;
    }

    /** Reopens a closed dispenser at the watermark, past every row a load published meanwhile; an
     *  open one stays as it is. */
    void reopen_() noexcept {
        node_id_t closed = node_id_unknown_k;
        atomic_ref<node_id_t>(next_dispensed_id_)
            .compare_exchange_strong(closed, atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire),
                                     std::memory_order_acq_rel, std::memory_order_relaxed);
    }

    /**
     *  @brief Wires the dispensed row @p new_node_id: stores row @p new_row of @p new_rows and its
     *      norms, walks the graph, writes the neighbors and unlocks slot zero, then back-links.
     *      Moves no watermark, so the row's range publishes whole through @c publish_.
     *
     *  The walk admits the rows below its watermark snapshot and the rows dispensed before it began
     *  whose slot zero shows them published. The row is already dispensed, so a refused walk wires a
     *  thinner node rather than abandoning an ID nothing will ever publish.
     */
    void wire_(node_id_t new_node_id, rows_view_t new_rows, std::size_t new_row, traversal_context_t& context,
               insert_config_t config) noexcept {
        location_t const new_location = chunk_schedule_t::locate(new_node_id);

        // The scratch structures persist across rows and calls - reset them for this one.
        context.results.clear();
        context.refining_candidates.clear();
        context.frontier.clear();
        context.visited.clear();

        // Norms derive from the owned copy - the exact bytes future distance computations stream -
        // and storing it is the import, a byte move when the scalars agree. An `f64` row widens
        // through the head of the staged queries, which an insertion walks without.
        vector_span_t const owned_vector = vector_span_for(new_location);
        nk::vector_span<f32_t> const widened {reinterpret_cast<f32_t*>(context.staged_queries.data()),
                                              geometry_.dimensions};
        import_row<elementwise_kit_t, rmscast_kit_t>(new_rows, new_row, kernel_, owned_vector, widened);
        vector_norms_t const new_norms = norms(owned_vector);
        if (stores_norms()) norm_ref_for(new_location) = new_norms;

        // A short walk waits out the entries dispensed ahead: wired at once, each would meet
        // node zero alone.
        std::size_t const expansion_factor = non_zero_expansion_factor(config);
        std::size_t const wanted = min_of<std::size_t>(geometry_.connectivity, expansion_factor);
        for (bool waited = false;; waited = true) {
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            node_id_t const first_undispensed_id =
                atomic_ref<node_id_t>(next_dispensed_id_).load(std::memory_order_acquire);
            dispensed_admission_t const admission {this, first_invisible_id, first_undispensed_id, new_node_id};
            if (failed(search_around(vector_view_t(owned_vector), new_norms, admission, expansion_factor, context,
                                     context.results, dummy_fold_t {}, {}, single_zero_rescoring_t {}))) {
                ++context.saturated_walks;
                break;
            }
            if (waited || first_invisible_id == new_node_id || context.results.size() >= wanted) break;
            for (node_id_t ahead = first_invisible_id; ahead != new_node_id; ++ahead) {
                neighbors_ref_t const neighbors = neighbors_for(chunk_schedule_t::locate(ahead));
                while (neighbors.peek_first() == node_id_unknown_k) neighbors.wait_while_locked();
            }
        }

        // No reader can arrive before slot 0 turns non-sentinel, so relaxed stores into slots
        // 1.. need no lock and the untouched tail keeps its sentinel termination.
        selection_t& selected = context.selected_neighbors;
        std::size_t const refine_distances = select_diverse(context.results, new_node_id, geometry_.connectivity,
                                                            selected, context);
        context.computed_distances += refine_distances;
        context.computed_distances_in_refines += refine_distances;
        neighbors_ref_t const new_neighbors = neighbors_for(new_location);
        candidates_count_t const selected_count = static_cast<candidates_count_t>(selected.size());
        for (candidates_count_t slot = 1; slot < selected_count; ++slot)
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
     *  watermark past them, and with them their keys @ref publish_keys_ filed. */
    void publish_(iota_view<node_id_t> range) noexcept {
        atomic_ref<node_id_t>(last_visible_id_).store(range.bound, std::memory_order_release);
    }

    /** Stores the key of @p node, which only its claimer writes, ahead of its range's turn. */
    void assign_key_(node_id_t node, key_t const& key) noexcept { keyed_.assign(node, key); }

    /** Files every row of @p range under its key, the map's one writer: the holder of the range's
     *  turn, before @ref publish_. Plain stores and one release byte per row, no
     *  read-modify-write. */
    void publish_keys_(iota_view<node_id_t> range) noexcept { keyed_.publish(range); }

#pragma endregion Insertion

#pragma region Refinement

    /** An audit over the IDs below @p visible, its bitsets on the index's own allocator. */
    expected<health_plan_t> plan_audit_(node_id_t visible) const noexcept {
        // An empty graph is audited, not refused: no words to sweep, no tasks, zero reached.
        health_plan_t plan {typename health_plan_t::bitset_allocator_t {vectors_allocator_}};
        if (failed(plan.widen(visible))) return {std::move(plan), status_code_t::out_of_memory_k};
        plan.rebind(*this, visible);
        return {std::move(plan), status_code_t::success_k};
    }

    /** Whether no insertion is in flight: every dispensed ID is published, or the dispenser is
     *  closed, which it only ever is at the watermark. */
    bool settled_() const noexcept {
        node_id_t const dispensed = atomic_ref<node_id_t const>(next_dispensed_id_).load(std::memory_order_acquire);
        return dispensed == node_id_unknown_k || dispensed == size();
    }

    /** Forgets every row below @p visible - what a split's source becomes once its rows published
     *  elsewhere: the watermark drops to zero, an open dispenser with it, the key map empties, the
     *  neighbor rows refill with the sentinel, and their storage returns to the free rows, so the
     *  IDs restart from zero and a closed dispenser reopens there. No claim may start meanwhile; a
     *  walk or lookup still reaching the index reads storage nothing frees, and its caller's epoch
     *  discards what it read. */
    void forget_(node_id_t visible) noexcept {
        atomic_ref<node_id_t>(last_visible_id_).store(0, std::memory_order_release);
        keyed_.forget();
        node_id_t dispensed = visible;
        atomic_ref<node_id_t>(next_dispensed_id_)
            .compare_exchange_strong(dispensed, 0, std::memory_order_release, std::memory_order_relaxed);
        std::size_t const row_bytes = connectivity() * sizeof(node_id_t);
        for (node_id_t row = 0; row != visible;) {
            location_t const location = chunk_schedule_t::locate(row);
            node_id_t const run = static_cast<node_id_t>(
                min_of<std::size_t>(visible - row, chunk_schedule_t::nodes_in(location.chunk) - location.offset));
            std::memset(neighbors_[location], 0xFF, run * row_bytes);
            row += run;
        }
        if (visible) release_capacity(visible);
    }

    /**
     *  @brief Stages the live prefix of @p neighbors into @c context.staged_ids, waiting out a
     *      lock holder like any traversal does; a published-empty self-loop stages nothing.
     *  @return The staged count - the tensor's live shape.
     */
    candidates_count_t stage_neighbors_(neighbors_ref_t neighbors, node_id_t node_id,
                                        traversal_context_t& context) const noexcept {
        ids_tensor_t& staged_ids = context.staged_ids;
        candidates_count_t const slots_count = neighbors.capacity();
        if (nk::failed(staged_ids.resize({slots_count}))) {
            ++context.saturated_walks;
            return 0;
        }
        node_id_t first_value = neighbors.peek_first();
        while (first_value == node_id_unknown_k && node_id < size()) {
            neighbors.wait_while_locked();
            first_value = neighbors.peek_first();
        }
        candidates_count_t staged_count = 0;
        // A forgotten row stays the sentinel and stages nothing; the caller's epoch voids the walk.
        if (first_value != node_id && first_value != node_id_unknown_k) {
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

    /** Adds the published @p row members the walk's candidates lack, scored against the node, past
     *  the walk's breadth: its nearer finds evict the far members, the long links among them, which
     *  must still compete. @return The distances computed. */
    std::size_t add_neighbor_candidates_(vector_view_t own_vector, vector_norms_t own_norms,
                                         std::span<node_id_t const> row, node_id_t first_invisible_id,
                                         traversal_context_t& context) const noexcept {
        results_array_t& results = context.results;
        quad<node_id_t> lane_ids;
        quad<vector_view_t> lane_vectors;
        quad<vector_norms_t> lane_norms;
        candidates_count_t gathered = 0;
        std::size_t computed = 0;
        auto score_gathered = [&]() noexcept {
            // Pad the tail lanes with lane zero; the padded outputs are never consumed.
            for (candidates_count_t lane = gathered; lane != lane_ids.size(); ++lane)
                lane_vectors[lane] = lane_vectors[0], lane_norms[lane] = lane_norms[0];
            quad<distance_t> const lane_distances = distances(own_vector, own_norms, lane_vectors, lane_norms);
            for (candidates_count_t lane = 0; lane != gathered; ++lane)
                results.insert(scored_node_t(lane_ids[lane], lane_distances[lane]), results.size() + 1);
            computed += gathered;
            gathered = 0;
        };
        for (node_id_t const member : row) {
            bool listed = member >= first_invisible_id;
            for (std::size_t position = 0; position != results.size() && !listed; ++position)
                listed = results[position].node_id == member;
            if (listed) continue;
            location_t const location = chunk_schedule_t::locate(member);
            lane_ids[gathered] = member;
            lane_vectors[gathered] = vector_for(location);
            lane_norms[gathered] = norm_for(location);
            if (++gathered == lane_ids.size()) score_gathered();
        }
        if (gathered) score_gathered();
        return computed;
    }

    /**
     *  @brief Writes @p selected into @p node_id 's row under its lock, sentinel terminated and
     *      republished through slot 0.
     *
     *  Neighbors are distinct, so a kept neighbor matching none of the @p live members read is a
     *  member the row gained, the one change a pass counts.
     *
     *  @return Whether the row gained a member.
     */
    bool rewrite_row_(node_id_t node_id, selection_t const& selected, std::span<node_id_t const> live) noexcept {
        neighbors_ref_t const own_neighbors = neighbors_for(chunk_schedule_t::locate(node_id));
        candidates_count_t const selected_count = static_cast<candidates_count_t>(selected.size());
        candidates_count_t matches = 0;
        for (candidates_count_t selected_index = 0; selected_index != selected_count; ++selected_index)
            for (node_id_t const live_id : live) matches += live_id == selected[selected_index].node_id;
        bool const changed = matches != selected_count;
        candidates_count_t const slots_count = own_neighbors.capacity();
        [[maybe_unused]] node_id_t const displaced = own_neighbors.lock();
        for (candidates_count_t slot = 1; slot < selected_count; ++slot)
            own_neighbors.store_slot(slot, selected[slot].node_id);
        for (candidates_count_t slot = max_of<candidates_count_t>(selected_count, 1); slot < slots_count; ++slot)
            own_neighbors.store_slot(slot, node_id_unknown_k);
        own_neighbors.unlock(selected_count ? selected[0].node_id : node_id);
        return changed;
    }

    /** Calls @p callback with every published member of @p node_id 's row, read without its lock,
     *  for the phases no row is written in; a published-empty row has none. */
    template <typename callback_type_>
    void for_each_member_(node_id_t node_id, callback_type_&& callback) noexcept {
        neighbors_ref_t const row = neighbors_for(chunk_schedule_t::locate(node_id));
        node_id_t const first = row.peek_first();
        if (first == node_id || first == node_id_unknown_k) return;
        callback(first);
        for (candidates_count_t slot = 1; slot != row.capacity(); ++slot) {
            node_id_t const member = row.load_slot(slot);
            if (member == node_id_unknown_k) return;
            callback(member);
        }
    }

    /**
     *  @brief Puts back what @p selected dropped of the @p live members that reachability needs.
     *
     *  That is a member whose last shallower in-link @p node_id is, and, when no pick stands one
     *  depth to root closer to a landmark, the row's nearest live member that does. A short row
     *  grows by each; a full one gives up its farthest pick that is kept by neither, a live member
     *  only once another shallower in-link keeps it.
     *
     *  By induction over the depths every row reached both ways stays so throughout the merge: a
     *  row at a depth keeps one in-link from the depth before, and one out-link to a lower reverse
     *  depth, and the landmarks at depth zero never move.
     */
    void keep_reach_(node_id_t node_id, selection_t const& live, selection_t& selected,
                     health_plan_t& health) noexcept {
        constexpr u8_t unknown_depth_k = health_plan_t::unknown_depth_k;
        u8_t const depth = health.depth_from_root(node_id), depth_to_root = health.depth_to_root(node_id);
        // The saturated depth counts no shallower in-links, so its children are never released.
        u8_t const child_depth = depth < unknown_depth_k - 1 ? static_cast<u8_t>(depth + 1) : unknown_depth_k;
        auto position_in = [&](selection_t const& among, node_id_t member) noexcept {
            std::size_t position = 0;
            while (position != among.size() && among[position].node_id != member) ++position;
            return position;
        };
        auto releasable = [&](node_id_t member) noexcept {
            return child_depth == unknown_depth_k || health.depth_from_root(member) != child_depth ||
                   position_in(live, member) == live.size() || health.take_shallower_in_link_if_spare(member);
        };
        // The kept members gather at the tail, ahead of which the farthest releasable
        // pick gives way.
        std::size_t kept = 0;
        auto keep = [&](scored_node_t member) noexcept {
            // A member the selection holds stays where it ranks; moving it would unsort the picks
            // the farthest of which give way.
            if (position_in(selected, member.node_id) != selected.size()) return;
            if (selected.size() < connectivity()) {
                [[maybe_unused]] nk::status_t const grown = selected.resize({selected.size() + 1});
                selected[selected.size() - 1] = member;
                ++kept;
                return;
            }
            std::size_t const tail = selected.size() - kept;
            std::size_t position = tail;
            while (position && !releasable(selected[position - 1].node_id)) --position;
            if (!position--) return;
            selected[position] = member;
            std::swap(selected[position], selected[tail - 1]);
            ++kept;
        };
        if (child_depth != unknown_depth_k)
            for (std::size_t position = 0; position != live.size(); ++position) {
                scored_node_t const member = live[position];
                if (health.depth_from_root(member.node_id) != child_depth ||
                    position_in(selected, member.node_id) != selected.size())
                    continue;
                if (!health.take_shallower_in_link_if_spare(member.node_id)) keep(member);
            }
        if (depth_to_root != unknown_depth_k && depth_to_root != 0) {
            std::size_t standing = 0;
            while (standing != selected.size() && health.depth_to_root(selected[standing].node_id) >= depth_to_root)
                ++standing;
            scored_node_t way_to_root = standing != selected.size()
                                            ? selected[standing]
                                            : scored_node_t {node_id_unknown_k,
                                                             std::numeric_limits<distance_t>::infinity()};
            for (std::size_t position = 0; position != live.size() && standing == selected.size(); ++position)
                if (health.depth_to_root(live[position].node_id) < depth_to_root &&
                    live[position].distance < way_to_root.distance)
                    way_to_root = live[position];
            if (way_to_root.node_id != node_id_unknown_k) keep(way_to_root);
        }
    }

#pragma endregion Refinement

#pragma region Search

    /** The body of @c search_typed below the watermark snapshot @p first_invisible_id the caller took:
     *  the index's own, or one consistent across shards. The @p hinted nodes take landmark slots
     *  ahead of the random draws. */
    template <synchronization_t synchronization_, typename fold_type_, typename rescoring_type_>
    expected<search_result_t> search_below_(vector_view_t query_vector, traversal_context_t& context,
                                            search_config_t config, fold_type_&& fold, results_array_t const& bound,
                                            rescoring_type_ const& rescoring, node_id_t first_invisible_id,
                                            std::span<node_id_t const> hinted) noexcept {
        if (config.traversal != traversal_t::walk_k) return {{}, status_code_t::unsupported_k};
        // Every distance kernel takes its width from the geometry, so a narrower query would be over-read.
        if (query_vector.size() != geometry_.dimensions) return {{}, status_code_t::config_mismatch_k};
        std::size_t const expansion_factor = non_zero_expansion_factor(config);

        // The same reservations `insert` makes, which also size the buffer `search_if` admits into.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, connectivity()));
            failed(sized))
            return {{}, sized};
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // An empty snapshot has nothing to walk.
        if (first_invisible_id == 0) return {{}, status_code_t::success_k};

        // Snapshot the context's running telemetry, to report this query's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        vector_norms_t const query_norms = norms(query_vector);
        status_t const walked = search_around<synchronization_>(
            query_vector, query_norms, snapshot_admission_t {first_invisible_id}, expansion_factor, context, bound,
            fold, hinted, rescoring);

        // A refused walk still ranked what it reached; the counter says the ranking is short
        // because scratch ran out, not because the graph had nothing better.
        if (failed(walked)) ++context.saturated_walks;
        search_result_t result;
        result.count = bound.size();
        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.restarts = context.restarts - restarts_before;
        result.saturated_walks = context.saturated_walks - saturated_walks_before;
        return {result, status_code_t::success_k};
    }

    /** The one dispatch of a walk under @p fold and @p rescoring onto @c search_below_, by the
     *  contract @p config names, below the watermark snapshot @p first_invisible_id, bounded by
     *  @c context.results and seeded by @p hinted ahead of its random draws. */
    template <typename fold_type_, typename rescoring_type_>
    expected<search_result_t> walk_(vector_view_t query_vector, fold_type_&& fold, rescoring_type_ const& rescoring,
                                    traversal_context_t& context, search_config_t config, node_id_t first_invisible_id,
                                    std::span<node_id_t const> hinted = {}) noexcept {
        if (config.synchronization != synchronization_t::concurrent_k)
            return search_below_<synchronization_t::immutable_k>(query_vector, context, config, fold, context.results,
                                                                 rescoring, first_invisible_id, hinted);
        return search_below_<synchronization_t::concurrent_k>(query_vector, context, config, fold, context.results,
                                                              rescoring, first_invisible_id, hinted);
    }

    /** The filtered walk behind @c search_if, below the watermark snapshot @p first_invisible_id,
     *  seeded by @p hinted. */
    template <typename predicate_type_>
    expected<search_result_t> walk_if_(vector_view_t query_vector, predicate_type_& predicate,
                                       traversal_context_t& context, search_config_t config,
                                       node_id_t first_invisible_id, std::span<node_id_t const> hinted = {}) noexcept {
        // Matches admit into the re-selection buffer, idle during any search and reserved beside the
        // the walk's candidates; it stops the walk, then swaps into `context.results`.
        context.refining_candidates.clear();
        predicate_fold<predicate_type_, results_array_t> fold {predicate, context.refining_candidates,
                                                               non_zero_expansion_factor(config)};
        auto const [result, status] = config.synchronization != synchronization_t::concurrent_k
                                          ? search_below_<synchronization_t::immutable_k>(
                                                query_vector, context, config, fold, context.refining_candidates,
                                                dummy_rescoring_t {}, first_invisible_id, hinted)
                                          : search_below_<synchronization_t::concurrent_k>(
                                                query_vector, context, config, fold, context.refining_candidates,
                                                dummy_rescoring_t {}, first_invisible_id, hinted);
        std::swap(context.results, context.refining_candidates);
        return {result, status};
    }

    /** The exact sibling of @c walk_if_: one query against every row below @p first_invisible_id,
     *  the nearest breadth of rows that @p admits lets in kept in @c context.results. */
    template <typename admits_type_>
    expected<search_result_t> scan_if_(vector_view_t query_vector, admits_type_&& admits, traversal_context_t& context,
                                       search_config_t config, node_id_t first_invisible_id) noexcept {
        if (query_vector.size() != geometry_.dimensions) return {{}, status_code_t::config_mismatch_k};
        std::size_t const expansion_factor = non_zero_expansion_factor(config);
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, connectivity()));
            failed(sized))
            return {{}, sized};
        context.results.clear();
        if (first_invisible_id == 0) return {{}, status_code_t::success_k};

        std::size_t const computed_distances_before = context.computed_distances;
        auto keep = [&](std::size_t, scored_node_t candidate) noexcept {
            [[maybe_unused]] bool const kept = context.results.insert(std::move(candidate), expansion_factor);
            return context.results.size() == expansion_factor ? context.results.top().distance
                                                              : (std::numeric_limits<distance_t>::max)();
        };
        // The query scores where it lies, so the staged queries hold only its packs.
        if (failed(context.reserve_queries(scan_scorer_t::packs_bytes(1, geometry_.dimensions))))
            return {{}, status_code_t::out_of_memory_k};
        vectors_view_t const queries {query_vector.data(), 1, query_vector.size()};
        if (status_t const scanned = scan_queries_(queries, context.staged_queries.data(), admits, keep, context,
                                                   config.cache_bytes, first_invisible_id);
            failed(scanned))
            return {{}, scanned};
        search_result_t result;
        result.count = context.results.size();
        result.computed_distances = context.computed_distances - computed_distances_before;
        return {result, status_code_t::success_k};
    }

    /** Rows @b [first,first+count) of @p rows in the stored scalar: viewed when they already are,
     *  else imported into @p context 's staged queries, sized for them beforehand. */
    vectors_view_t import_rows_(rows_view_t rows, std::size_t first, std::size_t count,
                                traversal_context_t& context) const noexcept {
        assert(context.staged_queries.numel() >= staged_rows_bytes(vector_dtype_k, count, rows.columns) &&
               "The staged queries are sized before rows import into them");
        return import_staged_rows<elementwise_kit_t, rmscast_kit_t, scalar_t>(rows, first, count, kernel_,
                                                                              context.staged_queries.data());
    }

    /** Queries one exact-scan group takes: an even share of @p queries_count over @p threads, so
     *  every thread passes over the rows once, in whole tasks of the scorer's columns. */
    static constexpr std::size_t scan_width(std::size_t queries_count, std::size_t threads) noexcept {
        std::size_t const share = divide_round_up(queries_count, max_of<std::size_t>(threads, 1));
        return min_of(round_up_to_multiple(share, scan_scorer_t::step_columns_k), queries_count);
    }

    /** Bytes of staged queries one exact-scan group of @p queries imports and packs into, which
     *  @c scan_ reserves; whatever a caller keeps past them survives the scan. */
    std::size_t scan_staged_bytes(std::size_t queries) const noexcept {
        return staged_rows_bytes(vector_dtype_k, queries, dimensions()) +
               scan_scorer_t::packs_bytes(queries, dimensions());
    }

    /** The exact dots of @p query against the published @p nodes into @p dots, four nodes a call
     *  through the walk's kit, as a fold reads them. */
    void dots_(vector_view_t query, std::span<node_id_t const> nodes, std::span<dot_result_t> dots) const noexcept {
        vector_norms_t const query_norms = norms(query);
        for (std::size_t first = 0; first < nodes.size(); first += quad_lanes_k) {
            std::size_t const lanes = min_of<std::size_t>(quad_lanes_k, nodes.size() - first);
            quad<vector_view_t> lane_rows;
            quad<vector_norms_t> lane_norms;
            for (std::size_t lane = 0; lane != quad_lanes_k; ++lane) {
                location_t const location = chunk_schedule_t::locate(nodes[first + min_of(lane, lanes - 1)]);
                lane_rows[lane] = vector_for(location);
                lane_norms[lane] = norm_for(location);
            }
            scores_result<dot_result_t> const scored = scores(query, query_norms, lane_rows, lane_norms);
            for (std::size_t lane = 0; lane != lanes; ++lane) dots[first + lane] = scored.dots[lane];
        }
    }

    /**
     *  @brief The exact sibling of @c walk_: scores every row below @p first_invisible_id against
     *      each query of @p group, keeping each row that beats a query's worst match through
     *      @p keep.
     *
     *  The group imports into @p context 's staged queries, as cheap beside a scan as its packs,
     *  which go past it, and @p config names the cache each block of rows stays in while every
     *  block of queries scores it.
     */
    template <typename admits_type_, typename keep_type_>
    status_t scan_(rows_view_t group, admits_type_&& admits, keep_type_&& keep, traversal_context_t& context,
                   search_config_t config, node_id_t first_invisible_id) noexcept {
        if (!first_invisible_id) return status_code_t::success_k;
        // Every kernel takes its width from the geometry, so a narrower group would be over-read.
        if (group.columns != dimensions()) return status_code_t::config_mismatch_k;
        if (failed(context.reserve_queries(scan_staged_bytes(group.rows)))) return status_code_t::out_of_memory_k;
        byte_t* const packs = context.staged_queries.data() +
                              staged_rows_bytes(vector_dtype_k, group.rows, group.columns);
        return scan_queries_(import_rows_(group, 0, group.rows, context), packs, admits, keep, context,
                             config.cache_bytes, first_invisible_id);
    }

    /**
     *  @brief Scores every row below @p first_invisible_id against every one of @p queries, the
     *      rows outer, so each block of them stays in @p cache_bytes while every block of queries
     *      meets it, each block of queries packed once into @p packs.
     *
     *  @p admits sees a row only once it beats some query's worst match, at most once per block of
     *  queries. Zero @p cache_bytes stream each chunk whole against one block of every query.
     */
    template <typename admits_type_, typename keep_type_>
    status_t scan_queries_(vectors_view_t queries, byte_t* packs, admits_type_& admits, keep_type_& keep,
                           traversal_context_t& context, std::size_t cache_bytes,
                           node_id_t first_invisible_id) noexcept {
        std::size_t const columns = queries.extent(0);
        if (failed(context.reserve_scan(columns, scan_scorer_t::step_rows_k, scan_scorer_t::dots_columns_k)))
            return status_code_t::out_of_memory_k;
        std::span<vector_norms_t> const query_norms {context.query_norms.data(), columns};
        std::span<distance_t> const thresholds {context.thresholds.data(), columns};
        for (std::size_t column = 0; column != columns; ++column)
            query_norms[column] = norms(queries.row(column).as_vector()),
            thresholds[column] = (std::numeric_limits<distance_t>::max)();
        pack_queries_(queries, packs, cache_bytes);
        scan_rows_({0, first_invisible_id}, queries, query_norms, packs, thresholds, admits, keep, context,
                   cache_bytes);
        return status_code_t::success_k;
    }

    /** Bytes a stored row takes in the chunks, padded to whole kit tasks. */
    std::size_t padded_row_bytes_() const noexcept { return padded_row_values(geometry_) * sizeof(scalar_t); }

    /** Queries one packed block of @p columns holds under @p cache_bytes. */
    std::size_t block_columns_(std::size_t columns, std::size_t cache_bytes) const noexcept {
        return scan_scorer_t::block_columns(columns, cache_bytes, padded_row_bytes_());
    }

    /** Stored rows one block streams past a packed block of @p columns queries under
     *  @p cache_bytes, sharing the cache with it and never below a task; zero with no cache named,
     *  where each chunk streams whole. */
    std::size_t block_rows_(std::size_t columns, std::size_t cache_bytes) const noexcept {
        if (!cache_bytes) return 0;
        std::size_t const row_bytes = padded_row_bytes_();
        std::size_t const block_bytes = block_columns_(columns, cache_bytes) * row_bytes;
        return max_of((cache_bytes - min_of(cache_bytes, block_bytes)) / row_bytes, scan_scorer_t::step_rows_k);
    }

    /** Packs @p queries a block at a time into @p packs, once for every thread that scans against
     *  them. */
    void pack_queries_(vectors_view_t queries, byte_t* packs, std::size_t cache_bytes) const noexcept {
        std::size_t const columns = queries.extent(0);
        std::size_t const block_columns = block_columns_(columns, cache_bytes);
        std::size_t const pack_stride = scan_scorer_t::packs_bytes(block_columns, dimensions());
        for (std::size_t first_column = 0, block = 0; first_column < columns; first_column += block_columns, ++block)
            scan_scorer_t::pack(
                queries(nk::range(first_column, min_of(first_column + block_columns, columns)), nk::slice),
                packs + block * pack_stride);
    }

    /**
     *  @brief Scores the stored @p rows against every packed block of @p queries, the rows
     *      the outer, so each block of them stays in @p cache_bytes while every block of
     *      queries meets it.
     *
     *  The @p query_norms and @p thresholds are the caller's, so threads scanning other rows may
     *  hand in copies of one set; the context's task buffers must fit one task against a block.
     */
    template <typename admits_type_, typename keep_type_>
    void scan_rows_(iota_view<node_id_t> rows, vectors_view_t queries, std::span<vector_norms_t const> query_norms,
                    byte_t const* packs, std::span<distance_t> thresholds, admits_type_& admits, keep_type_& keep,
                    traversal_context_t& context, std::size_t cache_bytes) noexcept {
        std::size_t const columns = queries.extent(0), dimensions = this->dimensions();
        std::size_t const block_columns = block_columns_(columns, cache_bytes);
        std::size_t const blocks = divide_round_up(columns, block_columns);
        std::size_t const pack_stride = scan_scorer_t::packs_bytes(block_columns, dimensions);
        std::size_t const block_rows = block_rows_(columns, cache_bytes);
        for (std::size_t first = rows.first; first < rows.bound;) {
            location_t const chunk_start = chunk_schedule_t::locate(static_cast<node_id_t>(first));
            std::size_t const chunk = chunk_start.chunk;
            std::size_t const chunk_rows = min_of(chunk_schedule_t::nodes_in(chunk) - chunk_start.offset,
                                                  rows.bound - first);
            std::size_t const stride_rows = block_rows ? block_rows : chunk_rows;
            for (std::size_t block_first = 0; block_first < chunk_rows; block_first += stride_rows) {
                std::size_t const rows_count = min_of(stride_rows, chunk_rows - block_first);
                location_t const location {chunk, chunk_start.offset + block_first};
                // Chunk rows are padded to whole kit tasks, while the queries score at
                // the logical width.
                vectors_view_t const padded {vectors_[location], rows_count,
                                             vectors_.columns() * nk::dimensions_per_value<scalar_t>()};
                nk::shape_storage_<2> const logical {{rows_count, dimensions},
                                                     {padded.stride_bytes(0), padded.stride_bytes(1)},
                                                     2};
                vectors_view_t const stored {padded.byte_data(), logical};
                std::span<vector_norms_t const> const row_norms =
                    stores_norms() ? std::span<vector_norms_t const> {&norms_[location], rows_count}
                                   : std::span<vector_norms_t const> {};
                for (std::size_t block = 0; block != blocks; ++block) {
                    std::size_t const first_column = block * block_columns;
                    std::size_t const width = min_of(block_columns, columns - first_column);
                    auto keep_column = [&](std::size_t column, scored_node_t candidate) noexcept {
                        return keep(first_column + column, candidate);
                    };
                    scan_scorer_t const scorer {kernel_,
                                                queries(nk::range(first_column, first_column + width), nk::slice),
                                                query_norms.subspan(first_column, width), packs + block * pack_stride,
                                                context.step_dots.span()};
                    scan_block(scorer, stored, static_cast<node_id_t>(first + block_first), row_norms,
                               thresholds.subspan(first_column, width), context.step_distances.span(), admits,
                               keep_column);
                }
            }
            first += chunk_rows;
        }
        context.computed_distances += rows.size() * columns;
    }

    /**
     *  @brief Scores gathered @p rows, the first being node @p first, against the packed blocks of
     *      @p queries from column @p first_column on, as @ref scan_rows_ scores stored rows.
     *
     *  The rows already sit in one block that shares the cache with a pack, gathered by the caller;
     *  the context's task buffers must fit one task against a block.
     */
    template <typename keep_type_>
    void scan_gathered_(vectors_view_t rows, node_id_t first, std::span<vector_norms_t const> row_norms,
                        vectors_view_t queries, std::span<vector_norms_t const> query_norms, byte_t const* packs,
                        std::span<distance_t> thresholds, keep_type_& keep, traversal_context_t& context,
                        std::size_t cache_bytes, std::size_t first_column) noexcept {
        std::size_t const columns = queries.extent(0);
        std::size_t const block_columns = block_columns_(columns, cache_bytes);
        std::size_t const pack_stride = scan_scorer_t::packs_bytes(block_columns, dimensions());
        dummy_predicate_t admits;
        for (std::size_t block = first_column / block_columns; block * block_columns < columns; ++block) {
            std::size_t const block_first = block * block_columns;
            std::size_t const block_bound = min_of(block_first + block_columns, columns);
            // A quads scorer reads its queries where they lie, so it may start mid-block; a
            // pack starts whole.
            std::size_t const start = scan_scorer_t::dots_columns_k ? block_first : max_of(block_first, first_column);
            auto keep_column = [&](std::size_t column, scored_node_t candidate) noexcept {
                return keep(start + column, candidate);
            };
            scan_scorer_t const scorer {kernel_, queries(nk::range(start, block_bound), nk::slice),
                                        query_norms.subspan(start, block_bound - start), packs + block * pack_stride,
                                        context.step_dots.span()};
            scan_block(scorer, rows, first, row_norms, thresholds.subspan(start, block_bound - start),
                       context.step_distances.span(), admits, keep_column);
            context.computed_distances += rows.extent(0) * (block_bound - start);
        }
    }

    /** What a search walks: the IDs below its watermark snapshot, published for every reader. */
    struct snapshot_admission_t {

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** Whether the walk may locate @p node_id. */
        bool admits(node_id_t node_id) const noexcept { return node_id < first_invisible_id; }

        /** Landmarks come from the IDs from this one on. */
        node_id_t sampling_first() const noexcept { return 0; }

        /** Landmarks come from the IDs below this bound, the snapshot. */
        node_id_t sampling_bound() const noexcept { return first_invisible_id; }
    };

    /** What an insertion walks: the IDs below its snapshot, and past it every row dispensed before
     *  the walk began whose slot zero shows it published. The acquire of the dispenser held those
     *  rows' chunks to the walk, so batches wired side by side link to each other rather than
     *  forming separate islands. */
    struct dispensed_admission_t {

        /** The index the walk reads. */
        ram_index* index = nullptr;

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** The dispenser as the walk began: no row at or past it is reachable. */
        node_id_t first_undispensed_id = 0;

        /** The walking row. */
        node_id_t node_id = 0;

        /** Whether the walk may locate @p candidate_id; the acquire of its slot zero holds its
         *  payload. */
        bool admits(node_id_t candidate_id) const noexcept {
            if (candidate_id < first_invisible_id) return true;
            if (candidate_id >= first_undispensed_id) return false;
            return index->neighbors_for(chunk_schedule_t::locate(candidate_id)).peek_first() != node_id_unknown_k;
        }

        /** Landmarks come from the IDs from this one on. */
        node_id_t sampling_first() const noexcept { return 0; }

        /** Landmarks come from the IDs below this bound, every one dispensed ahead of the walking row, which
         *  rows wired in dispensed order have mostly published. */
        node_id_t sampling_bound() const noexcept { return node_id; }
    };

    /**
     *  @brief Samples one 4-way batch of distinct, unvisited, published landmarks, scores them
     *      against the @p query, and queues them into the frontier & the result candidates.
     *  @return @c advanced_k, @c exhausted_k once every published node is already visited, or
     *      @c sated_k once the fold asks the walk to end - the last two end the walk rather than
     *      failing it. A failed status means scratch could not grow.
     *
     *  Uniform draws that collide with @c visited fall back to cyclic linear probing within the
     *  range @p admission sets, so every call makes strict progress without retry budgets. Only
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

        quad<node_id_t> landmarks_ids;
        quad<vector_view_t> landmarks_vectors;
        quad<vector_norms_t> landmarks_norms;
        candidates_count_t landmarks_in_batch = 0;

        if (status_t const seen = visited.reserve(visited.size() + landmarks_ids.size()); failed(seen))
            return {walk_step_t::exhausted_k, seen};
        // `visited` may also hold IDs no draw names, like an insertion's later siblings, so the size
        // check is conservative: it can only under-sample, never let the probe loop spin.
        node_id_t const sampling_first = admission.sampling_first(), sampling_bound = admission.sampling_bound();
        node_id_t unpublished_draws = 0;
        while (landmarks_in_batch != landmarks_ids.size() && visited.size() < sampling_bound - sampling_first) {
            // Caller-staged seeds go first, each taken once and skipped when unpublished or
            // already visited; past them, draw uniformly and probe forward cyclically.
            node_id_t landmark_node_id;
            if (!seeds.empty()) {
                landmark_node_id = seeds.front();
                seeds = seeds.subspan(1);
                if (!admission.admits(landmark_node_id) || visited.contains(landmark_node_id)) continue;
            }
            else {
                landmark_node_id = sampling_first +
                                   uniform_below<node_id_t> {sampling_bound - sampling_first}(context.random_generator);
                while (visited.contains(landmark_node_id))
                    landmark_node_id = landmark_node_id + 1 == sampling_bound ? sampling_first : landmark_node_id + 1;
            }

            location_t const landmark_location = chunk_schedule_t::locate(landmark_node_id);
            // Published nodes may be transiently locked for re-linking - re-roll those. Under a
            // mutation-free contract every sampled node is published & unlocked - no probe needed.
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (neighbors_for(landmark_location).peek_first() == node_id_unknown_k) {
                    if (landmark_node_id >= admission.first_invisible_id) {
                        if (++unpublished_draws == sampling_bound - admission.first_invisible_id) break;
                    }
                    // Below the snapshot only a lock holds the sentinel, unless a window forgot the
                    // row since.
                    else if (landmark_node_id >=
                             atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_relaxed))
                        break;
                    continue;
                }

            [[maybe_unused]] membership_t const membership = visited.insert(landmark_node_id);
            prefetch_vector_(landmark_location);
            landmarks_ids[landmarks_in_batch] = landmark_node_id;
            landmarks_vectors[landmarks_in_batch] = vector_for(landmark_location);
            landmarks_norms[landmarks_in_batch] = norm_for(landmark_location);
            ++landmarks_in_batch;
        }
        if (!landmarks_in_batch) return {walk_step_t::exhausted_k, status_code_t::success_k};

        // Pad the tail lanes with lane zero; the padded outputs are never consumed.
        for (candidates_count_t lane = landmarks_in_batch; lane != landmarks_ids.size(); ++lane) {
            landmarks_ids[lane] = landmarks_ids[0];
            landmarks_vectors[lane] = landmarks_vectors[0];
            landmarks_norms[lane] = landmarks_norms[0];
        }
        scores_result<dot_result_t> landmark_scores = scores(query, query_norms, landmarks_vectors, landmarks_norms);
        rescoring(landmarks_ids, landmarks_vectors, landmarks_norms, landmark_scores, landmarks_in_batch);
        context.computed_distances += landmarks_in_batch;
        quad<scored_node_t> landmarks_scored;
        for (candidates_count_t lane = 0; lane != landmarks_in_batch; ++lane)
            landmarks_scored[lane] = scored_node_t(landmarks_ids[lane], landmark_scores.distances[lane]);
        std::span const scored = std::span(landmarks_scored).first(landmarks_in_batch);
        // Landmarks fold right here, in lane order: they are already visited, so the walk never
        // re-scores them and this is their one chance to be absorbed.
        if (fold.absorb(scored, std::span<dot_result_t const>(landmark_scores.dots).first(landmarks_in_batch)) ==
            walk_demand_t::sated_k)
            return {walk_step_t::sated_k, status_code_t::success_k};
        settle_fold(fold);
        network_sort(scored, closer_scored_node_t {});
        std::span<scored_node_t> admitted = scored;
        if constexpr (std::is_same_v<rescoring_type_, single_zero_rescoring_t>)
            admitted = single_zero_rescoring_t::admitted(scored, results);
        results.insert(admitted, expansion_factor);
        for (std::size_t lane = 0; lane != admitted.size(); ++lane)
            if (status_t const queued = frontier.insert(scored_node_t(admitted[lane])); failed(queued))
                return {walk_step_t::advanced_k, queued};
        return {walk_step_t::advanced_k, status_code_t::success_k};
    }

    /**
     *  @brief Greedy best-first NSW walk from random published landmarks toward the @p query.
     *
     *  Read-only on the graph: takes no locks, writes nothing, retries transiently locked nodes.
     *  A refused re-table ends the walk early with the ranking it has and says so in the status,
     *  which the callers count as a saturated walk rather than a failure.
     *
     *  @param[in] query The vector the walk heads toward.
     *  @param[in] query_norms The norms of @p query, where the metric takes them.
     *  @param[in] admission What the walk may locate and draw landmarks from: the IDs below the
     *      caller's watermark snapshot, and for an insertion the published rows dispensed before it.
     *  @param[in] expansion_factor How many candidates the walk keeps - its stopping radius.
     *  @param[in,out] context Scratch & telemetry; must arrive pre-reserved & pre-cleared, and ends
     *      with up to @p expansion_factor nearest published candidates in @c context.results.
     *  @param[in] bound The candidates whose fill ends restarts and whose radius bounds the walk
     *      once full:
     *      @c context.results itself, or a filtered walk's admitted candidates.
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
        // A sated fold or a refused re-table may have left rows held from the last walk.
        context.ahead.clear();
        std::size_t const node_connectivity = connectivity();
        assert(results.capacity() >= expansion_factor && "Candidates short of the expansion never tightens the radius");

        // As many landmarks as one row holds, a stochastic multi-start drawn in the scoring
        // kernels' 4-way batches, so the last batch may round past the connectivity.
        candidates_count_t const landmark_batches = divide_round_up<candidates_count_t>(node_connectivity,
                                                                                        quad_lanes_k);
        for (candidates_count_t landmarks_batch = 0; landmarks_batch != landmark_batches; ++landmarks_batch) {
            auto const [seeding, seed_status] = sample_landmarks<synchronization_>(
                query, query_norms, admission, expansion_factor, context, seeds, fold, rescoring);
            if (failed(seed_status)) return seed_status;
            if (seeding == walk_step_t::sated_k) return status_code_t::success_k;
            if (seeding == walk_step_t::exhausted_k) break; // Everything published is already visited.
        }

        // Greedily walk the frontier nearest-first, keeping the best `expansion_factor` in `results`.
        while (true) {
            // A drained frontier before full result candidates means the walk exhausted everything
            // reachable from its seeds, so re-seed and continue. Every successful batch strictly
            // grows `visited` and `results`, so restarts are self-limiting without a budget.
            if (frontier.empty() && context.ahead.empty()) {
                // Re-seed until the bound fills - a filtered walk's matches may hide in regions the
                // drained frontier never reached.
                if (bound.size() >= expansion_factor) break;
                auto const [seeding, seed_status] = sample_landmarks<synchronization_>(
                    query, query_norms, admission, expansion_factor, context, seeds, fold, rescoring);
                if (failed(seed_status)) return seed_status;
                if (seeding == walk_step_t::sated_k) return status_code_t::success_k;
                if (seeding == walk_step_t::exhausted_k) break; // Nothing fresh left to restart from.
                ++context.restarts;
                continue;
            }

            // Peek - don't pop yet: the chosen node's neighbors may be locked, in which case
            // we retry the selection rather than dropping it from the frontier.
            // Rows held ahead come first: popped a hop early, so their reads are already in flight.
            frontier_queue_t& source = context.ahead.empty() ? frontier : context.ahead;
            scored_node_t const candidate = source.top();
            // Only a full bound has a radius, so a filtered walk routes on until its matches fill.
            bool const bound_full = bound.size() == expansion_factor;
            // Past the full bound's radius nothing deeper improves the answer, and the rows held
            // ahead are past it too, since it only tightened after they were popped.
            if (bound_full && candidate.distance > bound.top().distance) {
                if (&source == &frontier) break;
                context.ahead.clear();
                continue;
            }
            // Every other entry past that radius is never popped either: shed them when the coming
            // expansion, at most one node's neighbors, would not fit.
            if (bound_full && frontier.capacity() - frontier.size() < node_connectivity)
                frontier.erase_below(bound.top());

            location_t const candidate_location = chunk_schedule_t::locate(candidate.node_id);
            neighbors_ref<synchronization_> const candidate_neighbors = neighbors_for<synchronization_>( //
                candidate_location);

            // Slot 0 holds either the lock sentinel or the first neighbor's ID; a locked node is
            // waited out rather than dropped, since dropping it would silently degrade recall.
            node_id_t const first_neighbor_id = candidate_neighbors.peek_first();
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (first_neighbor_id == node_id_unknown_k) {
                    // Below the snapshot only a lock holds the sentinel, unless a window forgot the
                    // row since.
                    if (candidate.node_id < admission.first_invisible_id &&
                        candidate.node_id >=
                            atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_relaxed))
                        break;
                    candidate_neighbors.wait_while_locked();
                    continue;
                }

            // The node is live and readable - commit to it and drop it from the frontier.
            source.pop();
            ++context.visited_members;
            // The row's unvisited vectors stream in before its first quad, so the misses overlap.
            if constexpr (prefetcher_t::prefetches_k)
                for (candidates_count_t slot = 0; slot != candidate_neighbors.capacity(); ++slot) {
                    node_id_t const neighbor_id = slot ? candidate_neighbors.load_slot(slot) : first_neighbor_id;
                    if (neighbor_id == node_id_unknown_k) break;
                    if (admission.admits(neighbor_id) && !visited.contains(neighbor_id))
                        prefetch_vector_(chunk_schedule_t::locate(neighbor_id));
                }
            // Idle threads give the walk their share of the memory queues: keep `walk_width - 1`
            // rows popped ahead inside the radius, their rows streaming toward L2.
            while (context.ahead.size() + 1 < context.walk_width && !frontier.empty() &&
                   !(bound_full && frontier.top().distance > bound.top().distance)) {
                prefetch_neighbors_(chunk_schedule_t::locate(frontier.top().node_id));
                context.ahead.insert_reserved(scored_node_t(frontier.top()));
                frontier.pop();
            }

            // Slot 0 comes from the acquiring load above - never re-read it, a writer may have
            // locked the node since. The remaining slots are relaxed reads of a row a back-link may
            // rewrite meanwhile, so they mix old and new neighbors and may name one node twice.
            candidates_count_t const slots_count = candidate_neighbors.capacity();
            candidates_count_t neighbor_slot = 0;
            bool list_ended = false;
            while (!list_ended && neighbor_slot != slots_count) {

                quad<node_id_t> gathered_ids;
                quad<vector_view_t> gathered_vectors;
                quad<vector_norms_t> gathered_norms;
                candidates_count_t gathered_count = 0;
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
                    // A row rewritten mid-read may repeat a node inside one quad.
                    bool repeated = false;
                    for (candidates_count_t lane = 0; lane != gathered_count; ++lane)
                        repeated |= gathered_ids[lane] == neighbor_id;
                    if (repeated) continue;
                    location_t const neighbor_location = chunk_schedule_t::locate(neighbor_id);
                    gathered_ids[gathered_count] = neighbor_id;
                    gathered_vectors[gathered_count] = vector_for(neighbor_location);
                    gathered_norms[gathered_count] = norm_for(neighbor_location);
                    ++gathered_count;
                }
                if (!gathered_count) continue;

                // Pad the tail lanes with lane zero; the padded outputs are never consumed.
                for (candidates_count_t lane = gathered_count; lane != gathered_ids.size(); ++lane) {
                    gathered_ids[lane] = gathered_ids[0];
                    gathered_vectors[lane] = gathered_vectors[0];
                    gathered_norms[lane] = gathered_norms[0];
                }
                scores_result<dot_result_t> gathered_scores = scores( //
                    query, query_norms, gathered_vectors, gathered_norms,
                    // A row's last quad streams in the first quad of the row held next.
                    (!list_ended && neighbor_slot != slots_count) || context.ahead.empty()
                        ? vectors_quad_of(candidate_neighbors, neighbor_slot, admission.first_invisible_id,
                                          gathered_vectors)
                        : vectors_quad_of(
                              neighbors_for<synchronization_>(chunk_schedule_t::locate(context.ahead.top().node_id)), 0,
                              admission.first_invisible_id, gathered_vectors));
                rescoring(gathered_ids, gathered_vectors, gathered_norms, gathered_scores, gathered_count);
                context.computed_distances += gathered_count;
                quad<scored_node_t> scored_batch;
                for (candidates_count_t lane = 0; lane != gathered_count; ++lane)
                    scored_batch[lane] = scored_node_t(gathered_ids[lane], gathered_scores.distances[lane]);
                std::span const scored = std::span(scored_batch).first(gathered_count);
                // Folding is independent of navigation: a fold reads every scored quad in lane order,
                // including what the candidates reject, and never decides where the walk routes.
                if (fold.absorb(scored, std::span<dot_result_t const>(gathered_scores.dots).first(gathered_count)) ==
                    walk_demand_t::sated_k)
                    return status_code_t::success_k;
                network_sort(scored, closer_scored_node_t {});
                std::span<scored_node_t> admitted = scored;
                if constexpr (std::is_same_v<rescoring_type_, single_zero_rescoring_t>)
                    admitted = single_zero_rescoring_t::admitted(scored, results);

                // One merge folds the ascending quad into the candidates; only its accepted prefix
                // enters the frontier. The merge precedes the marks, so a refusal keeps the quad.
                std::size_t const accepted_count = results.insert(admitted, expansion_factor);

                // Grow the visited set ahead of the marks - `set()` itself never rehashes.
                // Full quads - the hot case - mark all four lanes in one interleaved probe sweep.
                if (status_t const seen = visited.reserve(visited.size() + gathered_count); failed(seen)) return seen;
                if (gathered_count == gathered_ids.size()) {
                    [[maybe_unused]] u64_t const fresh = visited.insert(gathered_ids);
                }
                else
                    for (candidates_count_t lane = 0; lane != gathered_count; ++lane) {
                        [[maybe_unused]] membership_t const membership = visited.insert(gathered_ids[lane]);
                    }

                for (candidates_count_t lane = 0; lane != accepted_count; ++lane)
                    if (status_t const queued = frontier.insert(scored_node_t(admitted[lane])); failed(queued))
                        return queued;
            }
            settle_fold(fold);
        }
        return status_code_t::success_k;
    }

#pragma endregion Search

    /** One row per vector, padded to whole kit tasks. @ref dense_ram_index_malloc */
    chunked_matrix<scalar_t, chunk_schedule_t, max_node_count_k, vectors_allocator_type_> vectors_;

    /** One entry per vector's pre-computed norms, covered only when @ref stores_norms. */
    chunked_column<vector_norms_t, chunk_schedule_t, max_node_count_k, vectors_allocator_type_> norms_;

    /** One row per vector's fixed-capacity neighbors, sentinel-filled. */
    chunked_matrix<node_id_t, chunk_schedule_t, max_node_count_k, neighbors_allocator_type_> neighbors_;

    /** Every node's key, and every key's nodes; the turn holder files a range's keys right before
     *  its watermark store. */
    keyed_t keyed_;

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

    /** The ID dispenser: the first identifier of the next batch's range, or the sentinel while a
     *  refine or a load holds it closed. One compare-exchange per batch moves it past every row of
     *  that batch, so a batch's IDs are contiguous. @ref dense_ram_index_concurrency */
    alignas(default_alignment_k) node_id_t next_dispensed_id_ {0};

    /** The visibility watermark: every ID strictly below it is published, its vector, norms &
     *  neighbors readable - the safe bound for random landmark sampling. Advanced strictly in
     *  dispensed order by each batch taking its turn. @ref dense_ram_index_concurrency */
    alignas(default_alignment_k) node_id_t last_visible_id_ {0};

    /** Fixed at construction - kept off the watermark's line with the other cold members. */
    alignas(default_alignment_k) geometry_t geometry_;

    /** Rows the index may ever host, lowered only by a reservation before the index is shared. */
    std::size_t ceiling_nodes_ = max_node_count_k;

    /** The distance kind every walk ranks by; only the distance epilogue and the norms chunks
     *  read it. */
    nk_kernel_kind_t kernel_;

    /** Streams the rows a walk or a scan reaches next toward the core. */
    prefetcher_t prefetcher_;

    /** Chunks every column covers. Deliberately non-atomic: only the elected grower mutates it,
     *  and successive growers are ordered by the acquire/release pair on @c remaining_capacity_,
     *  which no release lifts while one is inside. @c create_next_chunk */
    candidates_count_t chunks_count_ = 0;
};

/**
 *  @brief External reader's cursor over the watermark of one @c ram_index:
 *      random access & ID-order iteration over nodes, plus neighbor-following through a privately
 *      staged copy of the current node's neighbors.
 *
 *  Strictly read-only under every contract, so readers never stall writers. Under @c concurrent_k
 *  the staging reads neighbor words through relaxed atomics, tolerating the same stale-but-valid
 *  mix the traversal does; exact neighbors come from the @c immutable_k contract, which reads plain
 *  memory. The cursor captures the geometry once at construction, starts past the end, and
 *  @c seek(0) enters the graph.
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
    using candidates_count_t = typename index_t::candidates_count_t;
    using node_ids_view_t = typename index_t::node_ids_view_t;
    using location_t = typename index_t::location_t;

    using ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<node_id_t>;
    using ids_tensor_t = nk::tensor<node_id_t, ids_allocator_t, 1>;

    /** The watermark ordering behind this cursor's snapshots: acquiring while writers may be
     *  live, plain when the caller marked a mutation-free environment. */
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
    candidates_count_t staged_count_ = 0;

    /** This cursor's watermark snapshot: IDs strictly below it are readable. */
    node_id_t visible_count_;

    /** The current node ID, or the watermark snapshot when past the end. */
    node_id_t current_;

    /** The index's connectivity, captured once at construction. */
    candidates_count_t connectivity_;

    /** Binds the cursor & captures its snapshots; allocates nothing - call @c reserve(). */
    explicit ram_cursor(index_t const& index) noexcept
        : index_(index), visible_count_(atomic_ref<node_id_t const>(index.last_visible_id_).load(watermark_order_k)),
          current_(visible_count_), connectivity_(static_cast<candidates_count_t>(index.connectivity())) {}

    ram_cursor(ram_cursor const&) = delete;
    ram_cursor& operator=(ram_cursor const&) = delete;
    ram_cursor(ram_cursor&&) = default;
    ram_cursor& operator=(ram_cursor&&) = delete;

    /** Sizes the staging for the index's connectivity through @p allocator - the index's
     *  traversals-side family by default; call once before any @c seek. */
    status_t reserve(allocator_t allocator = {}) noexcept {
        if (staged_neighbors_.size() >= connectivity_) return status_code_t::success_k;
        auto staged = ids_tensor_t::zeros({connectivity_}, ids_allocator_t(allocator));
        if (!staged) return status_code_t::out_of_memory_k;
        staged_neighbors_ = std::move(staged.value);
        return status_code_t::success_k;
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

    /**
     *  @brief Positions on @p node & stages a copy of its neighbors, the cursor staying where it
     *      was on any refusal. Read-only under every contract.
     *
     *  @par Statuses:
     *  - @c malformed_k @p node is at or past the snapshot.
     *  - @c uninitialized_k The staging is not reserved yet.
     *  - @c contended_k The index forgot @p node since the snapshot: a clear or a split emptied it.
     */
    status_t seek(node_id_t node) noexcept {
        if (node >= visible_count_) return status_code_t::malformed_k;
        if (staged_neighbors_.size() < connectivity_) return status_code_t::uninitialized_k;
        location_t const location = index_t::chunk_schedule_t::locate(node);
        node_ids_view_t const row = index_.neighbors_for(location);
        // Slot 0 is the lock word: acquire it exactly once and never re-read it, so a fresh lock
        // sentinel cannot slip into the staging.
        node_id_t first_value;
        if constexpr (synchronization_ == synchronization_t::concurrent_k) {
            first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            while (first_value == node_id_unknown_k && node < index_.size()) {
                typename index_t::wait_t {}(row.data(), node_id_unknown_k, candidates_count_t(0), fu::wait_capped_t {});
                first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            }
            // Forgetting drops the watermark before it refills the rows with the lock sentinel.
            if (first_value == node_id_unknown_k) return status_code_t::contended_k;
        }
        else first_value = row[0];
        current_ = node;
        location_ = location;
        staged_neighbors_[0] = first_value;
        // Relaxed atomics make a live re-linker's stale-but-valid mix defined; marked
        // contracts read plain memory.
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            for (candidates_count_t slot = 1; slot != connectivity_; ++slot)
                staged_neighbors_[slot] = atomic_ref<node_id_t const>(row[slot]).load(std::memory_order_relaxed);
        else
            for (candidates_count_t slot = 1; slot != connectivity_; ++slot) staged_neighbors_[slot] = row[slot];
        // A self-loop is "published, zero neighbors"; the all-ones sentinel terminates short rows.
        if (staged_neighbors_[0] == node) { staged_count_ = 0; }
        else {
            candidates_count_t live = 0;
            while (live != connectivity_ && staged_neighbors_[live] != node_id_unknown_k) ++live;
            staged_count_ = live;
        }
        return status_code_t::success_k;
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

    /**
     *  Follows the staged neighbor at @p slot.
     *
     *  @par Statuses:
     *  - @c malformed_k @p slot is beyond the live prefix.
     *  - @c malformed_k The neighbor is at or past this cursor's snapshot.
     */
    status_t follow(candidates_count_t slot) noexcept {
        if (slot >= staged_count_) return status_code_t::malformed_k;
        return seek(staged_neighbors_[slot]);
    }
};

} // namespace unum::usearch::dense
