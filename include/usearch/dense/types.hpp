/**
 *  @file include/usearch/dense/types.hpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief The dense tier's vocabulary, shared by its RAM, VRAM & NVMe indexes.
 */
#pragma once

#include <cstring> // `std::memcmp` - keys compare by their bytes, whatever their type

#include <limits> // `std::numeric_limits` - the reducer's hit ceiling & unscored rows
#include <span>   // `std::span` - the reducer's rankings & outputs

#include <numkong/tensor.hpp> // `nk::matrix_view` - the indexer heads

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the reducer's staging
#include "usearch/utilities/ram_sorts.hpp"   // `radix_group`, `partial_sort` - the reducer's grouping & selection

namespace unum::usearch::dense {
using namespace unum::usearch::utilities;

/** Capability cascade for every scalar, shared by all its metrics: each capability group - the
 *  CPU's in @c ram_metric.hpp, NVIDIA's in @c vram_nvidia_metric.cuh - specializes it under a
 *  constraint on @p caps_. Reading order is the priority order; the first row whose @c available
 *  flag is true wins, else the fallback does. */
template <nk::numeric_dtype scalar_, nk_capability_t caps_>
struct quads_chain;

/** Dense node identifier within one sub-index - 32 bits, matching the 4-billion ceiling. */
using node_id_t = u32_t;

/** Distance representation shared by all metrics, regardless of the stored dtype. */
using distance_t = f32_t;

/** Reserved all-ones value, never a valid node ID within a sub-index. Doubles as the
 *  lock & empty-slot sentinel in neighbor lists of every index design. */
inline constexpr node_id_t node_id_unknown_k = 0xFFFFFFFFu;

/** The NumKong dtype a column of @p dtype stores, or @c nk_dtype_unknown_k where NumKong has
 *  none. */
constexpr nk_dtype_t nk_dtype_of(dtype_t dtype) noexcept {
    switch (dtype) {
    case dtype_t::f64_k: return nk_f64_k;
    case dtype_t::f32_k: return nk_f32_k;
    case dtype_t::bf16_k: return nk_bf16_k;
    case dtype_t::f16_k: return nk_f16_k;
    case dtype_t::e5m2_k: return nk_e5m2_k;
    case dtype_t::e4m3_k: return nk_e4m3_k;
    case dtype_t::e3m2_k: return nk_e3m2_k;
    case dtype_t::e2m3_k: return nk_e2m3_k;
    case dtype_t::i64_k: return nk_i64_k;
    case dtype_t::i32_k: return nk_i32_k;
    case dtype_t::i16_k: return nk_i16_k;
    case dtype_t::i8_k: return nk_i8_k;
    case dtype_t::i4x2_k: return nk_i4_k;
    case dtype_t::u64_k: return nk_u64_k;
    case dtype_t::u32_k: return nk_u32_k;
    case dtype_t::u16_k: return nk_u16_k;
    case dtype_t::u8_k: return nk_u8_k;
    case dtype_t::u4x2_k: return nk_u4_k;
    case dtype_t::u1x8_k: return nk_u1_k;
    default: return nk_dtype_unknown_k;
    }
}

/** The column dtype storing NumKong's @p dtype, or @c dtype_t::unknown_k where SafeGraphs has no
 *  token. */
constexpr dtype_t dtype_of(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_f64_k: return dtype_t::f64_k;
    case nk_f32_k: return dtype_t::f32_k;
    case nk_bf16_k: return dtype_t::bf16_k;
    case nk_f16_k: return dtype_t::f16_k;
    case nk_e5m2_k: return dtype_t::e5m2_k;
    case nk_e4m3_k: return dtype_t::e4m3_k;
    case nk_e3m2_k: return dtype_t::e3m2_k;
    case nk_e2m3_k: return dtype_t::e2m3_k;
    case nk_i64_k: return dtype_t::i64_k;
    case nk_i32_k: return dtype_t::i32_k;
    case nk_i16_k: return dtype_t::i16_k;
    case nk_i8_k: return dtype_t::i8_k;
    case nk_i4_k: return dtype_t::i4x2_k;
    case nk_u64_k: return dtype_t::u64_k;
    case nk_u32_k: return dtype_t::u32_k;
    case nk_u16_k: return dtype_t::u16_k;
    case nk_u8_k: return dtype_t::u8_k;
    case nk_u4_k: return dtype_t::u4x2_k;
    case nk_u1_k: return dtype_t::u1x8_k;
    default: return dtype_t::unknown_k;
    }
}

/** One traversal candidate: a node ID paired with its distance to the query. */
struct scored_node_t {

    /** Identifier of the scored node within its sub-index. */
    node_id_t node_id;

    /** Distance to the query, in the instantiated kernel's metric. */
    distance_t distance;
};

/** Orders scored nodes nearest-first; the comparator of the sorted results pool. */
struct closer_scored_node_t {
    inline bool operator()(scored_node_t first, scored_node_t second) const noexcept {
        return first.distance < second.distance;
    }
};

/** Orders scored nodes furthest-first; the comparator of the max-heap frontier. */
struct further_scored_node_t {
    inline bool operator()(scored_node_t first, scored_node_t second) const noexcept {
        return first.distance > second.distance;
    }
};

/** Outcome of one reachability audit, the same on every dense layer and residency. */
struct health_result_t {

    /** Nodes published when the audit started - the population it judges; later insertions are not counted. */
    std::size_t audited_nodes = 0;

    /** Of those, the nodes forward-reachable from the landmark seeds - the searcher's view. */
    std::size_t reachable_nodes = 0;

    /** Level-synchronous sweeps to convergence - a diameter proxy. */
    std::size_t sweeps = 0;
};

/** Outcome of one refinement or heal pass: every count means the same on every dense layer and residency. */
struct [[nodiscard]] refine_result_t {

    /** Nodes re-selected: the planned range, then the islands of every round. */
    std::size_t visited_nodes = 0;

    /** Of those, the nodes whose neighbors differ as a set from before - never more than @c visited_nodes. */
    std::size_t changed_nodes = 0;

    /** Nodes the first audit left unreached, counted once. */
    std::size_t islands_found = 0;

    /** Links actually written from a reached node to an island. */
    std::size_t islands_reconnected = 0;

    /** Nodes the final audit reached. */
    std::size_t reachable_nodes = 0;

    /** Audit levels swept across the pass. */
    std::size_t sweeps = 0;

    /** Frontier candidates expanded across all re-selection walks. */
    std::size_t visited_members = 0;

    /** Total distance computations, with padding lanes of 4-way kernels excluded. */
    std::size_t computed_distances = 0;

    /** Subset of @c computed_distances spent in the forward diversity selection. */
    std::size_t computed_distances_in_refines = 0;

    /** Subset of @c computed_distances spent re-selecting a full neighbor set while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;

    /** Starvation restarts across the walks - a graph-connectivity health signal. */
    std::size_t restarts = 0;

    /** Walks a scratch refusal ended early; their rows re-selected from a shorter pool. */
    std::size_t saturated_walks = 0;
};

/** The tail slots of a full row a heal may give up to islands, which spread over them by ID so
 *  islands sharing a row land apart: the farther half, keeping the nearer half that carries search
 *  and slot 0, and nothing in a single-neighbor row. */
template <typename size_type_>
constexpr size_type_ heal_spread(size_type_ connectivity) noexcept {
    return connectivity / 2;
}

/** The online-softmax carry of one attended row, in natural-log units - the C++ face of
 *  @c usearch_dense_softmax_state_t. */
struct softmax_state_t {

    /** Largest logit seen so far. */
    f32_t maximum_score;

    /** Sum of exponentials taken against that maximum. */
    f32_t partition;
};

/** Fixed 32-byte descriptor fronting every serialized dense index payload - volatile
 *  in-RAM snapshots, GPU-resident uploads, and persisted on-NVMe layouts alike. It describes
 *  the graph & its frozen geometry, never the residency: what sections follow and how they
 *  are laid out is the per-storage contract versioned by @c version_k. */
struct index_header_t {

    /** The expected @c magic contents: "usearch3" read as a little-endian word. */
    inline static constexpr u64_t magic_k = usearch_magic_k;

    /** The layout revision this implementation writes and understands. */
    inline static constexpr u32_t version_k = 1;

    /** Identifies USearch dense index payloads; must equal @c magic_k. */
    u64_t magic;

    /** Layout revision of the sections that follow. */
    u32_t version;

    /** The stored @c nk_dtype_t; must match the loading instantiation. */
    u32_t dtype;

    /** The @c nk_kernel_kind_t metric; must match the loading index's kernel. */
    u32_t kernel;

    /** Scalar dimensions per vector - one half of the frozen geometry. */
    u32_t dimensions;

    /** Neighbor IDs per node - the other half of the frozen geometry. */
    u32_t connectivity;

    /** Published nodes in the payload; the chunk layout re-derives from it. */
    u32_t node_count;
};
static_assert(sizeof(index_header_t) == 32 && std::is_trivially_copyable_v<index_header_t>);

/** Whether @p scalar_type_ stores real numbers, as indexer scoring needs of its keys. */
template <typename scalar_type_>
concept floating_scalar = !nk::is_integral_dtype<scalar_type_>();

/** A lightning indexer, as DeepSeek V4 and GLM 5 pick sparse-attention keys: a node scores `Σₕ wₕ·ReLU(qₕ·k)`
 *  against these heads, and its distance is `1 − score`, while the walk navigates by the query rows,
 *  conventionally `Σₕ wₕ·qₕ`. */
template <typename scalar_type_>
struct indexer_heads {

    /** Every query's heads back to back, @b [queries·heads,dimensions], in the index's own dtype. */
    nk::matrix_view<scalar_type_> heads;

    /** The signed head weights with the model's scale folded in, @b [queries,heads]. */
    nk::matrix_view<f32_t> weights;

    /** The heads of query row @p query alone. */
    indexer_heads row(std::size_t query) const noexcept {
        std::size_t const heads_count = weights.extent(1);
        return {heads(nk::range(query * heads_count, (query + 1) * heads_count), nk::slice),
                weights(nk::range(query, query + 1), nk::slice)};
    }
};

/** The walks' default: no heads, so every node scores as the kit scored it. */
struct dummy_heads_t {
    dummy_heads_t row(std::size_t) const noexcept { return {}; }
};

/** Warps per thread block across every kernel of the VRAM design - four schedulers per
 *  multiprocessor on every modern GPU, one lane group each. */
inline constexpr u32_t vram_warps_per_block_k = 4;

/** Resident blocks per multiprocessor - shared by the launch grids and the kernels' launch
 *  bounds, so the register allocator sizes to it; small-pool search kernels fit four. */
inline constexpr u32_t vram_resident_blocks_per_multiprocessor_k = 3;

/** Rows of the per-warp shared candidate cache; also bounds the diversity-selection pool prefix
 *  and the staging arrays. */
inline constexpr u32_t vram_cache_rows_k = 64;

/** Ceiling on concurrently in-flight candidate-vector gathers per lane - the memory-parallelism
 *  window over the @b candidate axis; the NVIDIA index also caps the lanes sharing one row at it,
 *  so a wide row takes more steps rather than more reductions. */
inline constexpr u32_t vram_max_gathers_in_flight_k = 8;

/** IDs the first insert wave over an empty shard admits at once; every later wave doubles, as wide
 *  as the rows already below it. */
inline constexpr u32_t vram_admission_ramp_floor_k = 32;

/** Slots of the widest register pool on the walks' rung ladder, @c vram_sorted_pool<16,warp_t>:
 *  as many ranks per lane. A walk expanding past that pool's width - this times the warp's lanes,
 *  512 on 32-lane warps - takes the arena frontier instead, so no expansion or count is capped by
 *  it. */
inline constexpr u32_t vram_widest_pool_slots_k = 16;

/** Device-resident telemetry counters, one block per driving context, zeroed at each submission
 *  and read after its drain. */
struct vram_telemetry_block_t {
    u64_t hops, scored, lock_attempts, lock_skips, reselections, restarts, locked_drops;

    /** Scored candidates split by storage residency; zero unless the view names a resident
     *  range. */
    u64_t local_scored, remote_scored;

    /** Frontier tail entries shed past their slice, and visited-set claims out of probes. */
    u64_t tail_drops, visited_saturations;

    /** Nodes whose neighbors a refinement merge landed in a different sequence than they held. */
    u64_t changed_nodes;
};

/** Why @p keys and @p scores cannot hold @p rows rankings side by side with contiguous rows, or
 *  success. */
template <typename key_type_, typename score_type_>
status_t rankings_refusal(std::size_t rows, nk::matrix_span<key_type_> keys,
                          nk::matrix_span<score_type_> scores) noexcept {
    if (keys.extent(0) != rows || scores.extent(0) != rows || scores.extent(1) != keys.extent(1))
        return status_t::config_mismatch_k;
    if (keys.stride_bytes(1) != sizeof(key_type_) || scores.stride_bytes(1) != sizeof(score_type_))
        return status_t::unsupported_layout_k;
    return status_t::success_k;
}

/** Why @p span cannot take @p rows copied rows of @p columns values each, packed within a row, or
 *  success; rows may stride past @p columns, and wider rows keep their tails. */
template <typename value_type_>
status_t rows_refusal(std::size_t rows, nk::matrix_span<value_type_> span, std::size_t columns) noexcept {
    if (span.extent(0) != rows || (rows && span.extent(1) < columns)) return status_t::config_mismatch_k;
    if (rows && span.stride_bytes(1) != sizeof(value_type_)) return status_t::unsupported_layout_k;
    return status_t::success_k;
}

/** Bytes @p count logical values of @p dtype span packed back to back. */
inline std::size_t packed_bytes(nk_dtype_t dtype, std::size_t count) noexcept {
    return divide_round_up<std::size_t>(nk_dtype_bits(dtype) * count, 8);
}

/** One strided matrix's rank-2 shape: NumKong counts the last extent in logical dimensions, so a
 *  packed row is as wide as @p columns; a zero @p stride means packed rows. */
template <typename scalar_type_>
nk::shape_storage_<2> rows_shape_of(std::size_t rows, std::size_t stride, std::size_t columns) noexcept {
    std::size_t const row_values = divide_round_up<std::size_t>(columns, nk::dimensions_per_value<scalar_type_>());
    std::ptrdiff_t const row_bytes = static_cast<std::ptrdiff_t>(stride ? stride : row_values * sizeof(scalar_type_));
    return {{rows, columns}, {row_bytes, static_cast<std::ptrdiff_t>(sizeof(scalar_type_))}, 2};
}

/** A scalar rows may hold: a NumKong scalar, or a builtin float the engines keep as their own. */
template <typename scalar_type_>
concept row_scalar = nk::numeric_dtype<scalar_type_> || std::is_same_v<scalar_type_, f32_t> ||
                     std::is_same_v<scalar_type_, f64_t>;

/** The NumKong dtype rows of @p scalar_type_ hold. */
template <row_scalar scalar_type_>
constexpr nk_dtype_t dtype_of() noexcept {
    if constexpr (std::is_same_v<scalar_type_, f32_t>) return nk_f32_k;
    else if constexpr (std::is_same_v<scalar_type_, f64_t>) return nk_f64_k;
    else return scalar_type_::dtype();
}

/**
 *  @brief Rows in any scalar NumKong casts, @b [rows,columns] in logical values, @c stride bytes
 *      apart; a zero stride repeats the first row.
 *
 *  Every verb takes its input rows this way and imports them into its arm's scalar per thread, so
 *  a typed view converts and passes untouched when it already holds that scalar. Sub-byte rows
 *  fill whole bytes, as NumKong casts them and sub-byte indexes store them.
 */
struct rows_view_t {
    byte_t const* data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t stride = 0;
    nk_dtype_t dtype = nk_dtype_unknown_k;

    rows_view_t() = default;
    rows_view_t(byte_t const* data, std::size_t rows, std::size_t columns, std::size_t stride,
                nk_dtype_t dtype) noexcept
        : data(data), rows(rows), columns(columns), stride(stride), dtype(dtype) {}
    template <row_scalar scalar_type_>
    rows_view_t(nk::matrix_view<scalar_type_> typed) noexcept
        : data(reinterpret_cast<byte_t const*>(typed.byte_data())), rows(typed.extent(0)), columns(typed.extent(1)),
          stride(static_cast<std::size_t>(typed.stride_bytes(0))), dtype(dtype_of<scalar_type_>()) {}
    template <row_scalar scalar_type_>
    rows_view_t(nk::matrix_span<scalar_type_> typed) noexcept : rows_view_t(nk::matrix_view<scalar_type_>(typed)) {}

    byte_t const* row(std::size_t row) const noexcept { return data + row * stride; }

    /** Rows @b [first,first+count). */
    rows_view_t slice(std::size_t first, std::size_t count) const noexcept {
        return {row(first), count, columns, stride, dtype};
    }

    /** Whether the rows can be borrowed as @p scalar_type_ in place: that scalar, not repeated. */
    template <row_scalar scalar_type_>
    bool holds() const noexcept {
        return dtype == dtype_of<scalar_type_>() && stride;
    }

    /** The rows as @p scalar_type_, which they must @ref holds. */
    template <row_scalar scalar_type_>
    nk::matrix_view<scalar_type_> as() const noexcept {
        return {reinterpret_cast<char const*>(data), rows_shape_of<scalar_type_>(rows, stride, columns)};
    }
};

/** Rows a verb writes out in the caller's scalar, the writable twin of @c rows_view_t: packed
 *  within a row, filling whole bytes, @c stride bytes apart. */
struct rows_span_t {
    byte_t* data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t stride = 0;
    nk_dtype_t dtype = nk_dtype_unknown_k;

    rows_span_t() = default;
    rows_span_t(byte_t* data, std::size_t rows, std::size_t columns, std::size_t stride, nk_dtype_t dtype) noexcept
        : data(data), rows(rows), columns(columns), stride(stride), dtype(dtype) {}
    template <row_scalar scalar_type_>
    rows_span_t(nk::matrix_span<scalar_type_> typed) noexcept
        : data(reinterpret_cast<byte_t*>(typed.byte_data())), rows(typed.extent(0)), columns(typed.extent(1)),
          stride(static_cast<std::size_t>(typed.stride_bytes(0))), dtype(dtype_of<scalar_type_>()) {}

    byte_t* row(std::size_t row) const noexcept { return data + row * stride; }

    /** Rows @b [first,first+count). */
    rows_span_t slice(std::size_t first, std::size_t count) const noexcept {
        return {row(first), count, columns, stride, dtype};
    }

    /** Whether the rows already are @p scalar_type_, so a verb may write them in place. */
    template <row_scalar scalar_type_>
    bool holds() const noexcept {
        return dtype == dtype_of<scalar_type_>();
    }

    /** The rows as @p scalar_type_, which they must @ref holds. */
    template <row_scalar scalar_type_>
    nk::matrix_span<scalar_type_> as() const noexcept {
        return {reinterpret_cast<char*>(data), rows_shape_of<scalar_type_>(rows, stride, columns)};
    }
};

/** One ranked hit while @c maxsim_reducer folds a query, then one document's summed maxima. */
template <typename key_type_>
struct maxsim_entry {

    /** The document the ranked token belongs to. */
    key_type_ key;

    /** The query row that ranked it, counted within its query. */
    u32_t row;

    /** Its dot against that row, then the document's summed per-row maxima once the query folds. */
    f64_t score;
};

/**
 *  @brief Late-interaction scores from per-row rankings: each row's best dot per key, summed per key over
 *      its query, the largest an output row holds landing largest first.
 *
 *  Every row is one query token's ranking under the inner kind, where a distance is @c 1 @c - @c dot, so a
 *  key a row never ranked adds nothing for that row. The staging is sized once by @c reserve, and both the
 *  grouping and the selection run in place, so a reduction never allocates.
 */
template <typename key_type_, typename distance_type_, typename allocator_type_>
class maxsim_reducer {
  public:
    using key_t = key_type_;
    using distance_t = distance_type_;
    using entry_t = maxsim_entry<key_t>;

    explicit maxsim_reducer(allocator_type_ allocator) noexcept : staged_(allocator), row_best_(allocator) {}

    /** Sizes the staging for queries of @p rows_per_query rankings, each @p width hits wide; a
     *  query wider than @c u32_t hits refuses with @c config_mismatch_k. */
    status_t reserve(std::size_t rows_per_query, std::size_t width) noexcept {
        std::size_t const hits = rows_per_query * width;
        if (hits > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        rows_per_query_ = rows_per_query;
        if (status_t const sized = staged_.resize_uninitialized(hits); failed(sized)) return sized;
        if (status_t const sized = row_best_.resize_uninitialized(rows_per_query); failed(sized)) return sized;
        for (std::size_t row = 0; row != rows_per_query; ++row) row_best_[row] = unscored_k;
        return status_t::success_k;
    }

    /**
     *  @param[in] row_keys Keys of every row's ranking, one ranking per row, at most the reserved
     *      width.
     *  @param[in] row_distances Distances of the same rankings, parallel to @p row_keys.
     *  @param[in] row_counts Hits ranked per row.
     *  @param[out] keys The best keys of each query, one query per row, as many as its width.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, one per query.
     *  @return The keys written over every query.
     */
    std::size_t operator()(nk::matrix_view<key_t> row_keys, nk::matrix_view<distance_t> row_distances,
                           std::span<std::size_t const> row_counts, nk::matrix_span<key_t> keys,
                           nk::matrix_span<f64_t> scores, std::span<std::size_t> counts) noexcept {
        entry_t* const staged = staged_.data();
        std::size_t written = 0;
        for (std::size_t query = 0; query != counts.size(); ++query) {
            std::size_t ranked = 0;
            for (std::size_t row = 0; row != rows_per_query_; ++row) {
                std::size_t const ranking = query * rows_per_query_ + row;
                for (std::size_t hit = 0; hit != row_counts[ranking]; ++hit)
                    staged[ranked++] = {row_keys(ranking, hit), static_cast<u32_t>(row),
                                        1 - static_cast<f64_t>(row_distances(ranking, hit))};
            }

            // Grouped by key in place, each key's run adds every row's best dot once, restoring the row slots.
            radix_group(to_writable_span(staged_).first(ranked), &entry_t::key);
            std::size_t unique = 0;
            for (std::size_t run = 0, end = 0; run != ranked; run = end) {
                while (end != ranked && same_key_(staged[end].key, staged[run].key)) ++end;
                for (std::size_t hit = run; hit != end; ++hit)
                    row_best_[staged[hit].row] = max_of(row_best_[staged[hit].row], staged[hit].score);
                f64_t sum = 0;
                for (std::size_t hit = run; hit != end; ++hit)
                    if (f64_t& best = row_best_[staged[hit].row]; best != unscored_k) sum += best, best = unscored_k;
                staged[unique++] = {staged[run].key, 0, sum};
            }

            std::size_t const kept = min_of(keys.extent(1), unique);
            partial_sort(to_writable_span(staged_).first(unique), kept, better_);
            for (std::size_t rank = 0; rank != kept; ++rank)
                keys(query, rank) = staged[rank].key, scores(query, rank) = staged[rank].score;
            counts[query] = kept;
            written += kept;
        }
        return written;
    }

  private:

    /** Marks a row slot no hit of the current key has scored yet. */
    static constexpr f64_t unscored_k = std::numeric_limits<f64_t>::lowest();

    static bool same_key_(key_t const& first, key_t const& second) noexcept {
        return std::memcmp(&first, &second, sizeof(key_t)) == 0;
    }

    /** Larger scores first, ties broken by key so equal scores rank reproducibly. */
    static bool better_(entry_t const& first, entry_t const& second) noexcept {
        if (first.score != second.score) return first.score > second.score;
        return std::memcmp(&first.key, &second.key, sizeof(key_t)) < 0;
    }

    growing_buffer<entry_t, allocator_type_> staged_;
    growing_buffer<f64_t, allocator_type_> row_best_;
    std::size_t rows_per_query_ = 0;
};

} // namespace unum::usearch::dense
