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

/** Graph node identifier within one dense shard - 32 bits, the 4-billion ceiling every dense tower
 *  and the C ABI share. */
using node_id_t = u32_t;

/** Reserved all-ones value, never a valid node ID within a shard; the lock & empty-slot sentinel
 *  of every dense neighbor list. */
inline constexpr node_id_t node_id_unknown_k = node_id_limits<node_id_t>::unknown_k;

/** The form every dense tower names its entries by. */
using shard_node_t = shard_node<node_id_t>;
static_assert(sizeof(shard_node_t) == 8);

/** Capability cascade for every scalar, shared by all its metrics: each capability group - the
 *  CPU's in @c ram_metric.hpp, NVIDIA's in @c vram_nvidia_metric.cuh - specializes it under a
 *  constraint on @p caps_. Reading order is the priority order; the first kit this build compiled
 *  whose @c capability_k @p caps_ admits wins, else the fallback does. */
template <nk::numeric_dtype scalar_, nk_capability_t caps_>
struct quads_ladder;

/** The GPU kernels only dense indexes launch, the first template argument of each vendor's dense
 *  kernel template - @c cuda_dense_kernel and @c amd_dense_kernel - and the words of Apple's dense
 *  exported names; the walks basic graphs share stay in @c utilities::vram_kernel_t. A vendor
 *  sizes its table of loaded dense kernels by @c count_k. */
enum class vram_dense_kernel_t : u32_t {

    /** A search rescoring its candidates through the indexer heads. */
    search_indexer_k,

    /** Scores every visible row against each query, without walking the graph. */
    scan_k,

    /** A walk keeping every scored pair, rather than ranked matches. */
    multiply_k,

    /** A walk folding the values of the rows it attends to, per head. */
    attend_k,

    /** Folds each walked node's forward picks and the walkers chained onto it into its row. */
    merge_back_links_k,

    /** Re-selects each row of a fold bin's shard from its members and its cross results, or merges
     *  the picks other shards of the bin made of it into its row. */
    fold_k,

    /** Rebuilds the norms column from the stored vectors. */
    derive_norms_k,

    /** Merges the attended rows of several shards' walks by the flash rule. */
    merge_attention_k,

    /** Ranks a walk's matches into the queries' rankings. */
    resolve_matches_k,

    /** Merges the rankings of several shards' walks per query. */
    merge_matches_k,

    /** Casts rows into the stored scalar. */
    cast_k,

    /** Casts rows into the stored scalar, scaled to unit RMS. */
    rmscast_k,

    /** The number of kernels, sizing the tables of loaded kernels. */
    count_k,
};

/** The Itanium spelling of @ref vram_dense_kernel_t inside a dense kernel's own scope, @c S1_, as
 *  @c vram_kernel_symbol documents. */
constexpr char const* kernel_enum_mangling(vram_dense_kernel_t) noexcept { return "NS1_19vram_dense_kernel_tE"; }

/** The breadth a GPU insert walks, @p expansion_factor or, when it asks for none, a measured 128:
 *  on SIFT-10M u8 at connectivity 32, inserting at the 64 of @ref default_expansion_factor instead
 *  dropped recall@10 at search breadths 64, 128 and 1024 from 0.9302, 0.9717 and 0.9989 to 0.9158,
 *  0.9610 and 0.9978. */
constexpr std::size_t vram_insert_expansion(std::size_t expansion_factor) noexcept {
    return expansion_factor ? expansion_factor : 128;
}

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

/** Orders scored nodes nearest-first; the comparator of the sorted result candidates. */
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

/** Vector dimensions and neighbor slots per node, fixed when a dense index is constructed. */
struct geometry_t {
    std::size_t dimensions = 0;
    connectivity_t connectivity {};
    bool operator==(geometry_t const&) const noexcept = default;
};

/** What a compaction targets: the most rows any target shard may hold, and the breadth of every
 *  re-selection walk. */
struct compact_config_t {

    /** Rows one target shard may hold; zero, or anything past the 32-bit ID space, means 2^32 - 2.
     *  A source shard larger than this is cut, so a smaller capacity redistributes. */
    std::size_t shard_capacity = 0;

    /** Breadth of every re-selection walk and block build; zero means twice the connectivity. */
    u32_t expansion_factor = 0;

    /** Bytes a closing refine may hold beyond the index, as @c refine_config_t takes them. */
    std::size_t memory_limit = 0;
};

/** Outcome of a compaction, whatever its residencies. */
struct [[nodiscard]] compact_result_t {

    /** Shards the sources held, and shards the target holds after. */
    std::size_t source_shards = 0, target_shards = 0;

    /** Rows that changed shard, and edges dropped because a cut separated their ends. */
    std::size_t moved_entries = 0, cut_edges = 0;

    /** The re-selections and the heal, counted as every refine counts them. */
    refine_report_t graph;
};

/** The breadth every graph engine counts its candidates in. */
using candidates_count_t = u32_t;

/** Optional tuning knobs for an insertion's walks; zeros mean "derive from connectivity". */
struct insert_config_t {

    /** How many candidates a walk keeps during wiring; defaults to twice the connectivity. */
    candidates_count_t expansion_factor = 0;
};

/** Optional tuning knobs for a refinement's re-selection walks; zeros mean "derive from
 *  connectivity". */
struct refine_config_t {

    /** How many candidates a re-selection keeps; defaults to twice the connectivity, like every
     *  walk. */
    candidates_count_t expansion_factor = 0;

    /** Bytes the whole pass may hold beyond the index: the audit and a bit per row carried across
     *  tiles, then tiles of contiguous rows at 8 B per neighbor slot plus 4 B each, as many rows as
     *  fit, or on VRAM at 4 B per slot past the walks' arena too. Zero walks one storage chunk per
     *  tile, or on VRAM every walked row in one tile. */
    std::size_t memory_limit = 0;
};

/** Optional tuning knobs for every ranking walk; zeros mean "derive from connectivity". */
struct search_config_t {

    /** How many candidates a walk keeps; defaults to twice the connectivity. Raise it to trade
     *  latency for recall - the caller reranks anyway. */
    candidates_count_t expansion_factor = 0;

    /** Reader contract; @c immutable_k - or stronger - drops every atomic in the traversal when
     *  the caller certifies no concurrent mutators. */
    synchronization_t synchronization = synchronization_t::concurrent_k;

    /** How entries are reached; @c scan_k scores every published entry exactly, keeping the
     *  breadth's nearest. */
    traversal_t traversal = traversal_t::walk_k;

    /** Bytes of cache each executor thread keeps to itself: a RAM scan streams its rows in blocks
     *  that stay there while every pack of its queries scores them, and zero streams every chunk
     *  whole; ignored elsewhere. */
    std::size_t cache_bytes = 0;
};

/** Where a ranking writes: up to as many keys and scores per query as the matrices are wide, and
 *  how many each query kept. */
template <typename key_type_, typename score_type_>
struct ranked_matches_span {
    nk::matrix_span<key_type_> keys;
    nk::matrix_span<score_type_> scores;
    std::span<std::size_t> counts;
};

/** The keys each query's walk starts from ahead of its random landmarks: up to as many per query
 *  as the matrix is wide, and how many each query holds. It is the read-only half of a ranking, so
 *  one search's matches hint the next. Empty counts hint nothing. */
template <typename key_type_>
struct ranked_hints_view {
    nk::matrix_view<key_type_> keys;
    std::span<std::size_t const> counts;
};

/** The hints of query @p query: the first @c counts[query] keys of its row, or none when @p hints
 *  is empty. */
template <typename key_type_>
std::span<key_type_ const> hints_of(ranked_hints_view<key_type_> const& hints, std::size_t query) noexcept {
    if (hints.counts.empty()) return {};
    return to_span(hints.keys.row(query).as_vector()).first(hints.counts[query]);
}

/**
 *  @brief Why @p hints cannot seed @p queries queries.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k Non-empty @p hints hold other than one row per query.
 *  - @c shape_mismatch_k A query counts more keys than the hints are wide.
 */
template <typename key_type_>
status_t hints_refusal(ranked_hints_view<key_type_> const& hints, std::size_t queries) noexcept {
    if (hints.counts.empty()) return status_t::success_k;
    if (hints.counts.size() != queries || hints.keys.extent(0) != queries) return status_t::config_mismatch_k;
    for (std::size_t const count : hints.counts)
        if (count > hints.keys.extent(1)) return status_t::shape_mismatch_k;
    return status_t::success_k;
}

/**
 *  @brief Where a product or MaxSim writes, in host or device memory: each query's keys and
 *      scores back to back from its offset, one more offset than queries, the first offset
 *      written by the caller.
 *
 *  The offsets index the keys and scores, so a window of a batch keeps the whole keys and scores
 *  and cuts its own offsets, the first holding where the window before it ended. A host tower
 *  stages every query's run at @ref staging_of, each thread owning whole queries, and
 *  @c publish_runs lays the window out once the threads are done; a device tower appends on the
 *  device from the first offset.
 */
template <typename key_type_, typename score_type_>
struct products_tape {
    std::span<key_type_> keys;
    std::span<score_type_> scores;
    std::span<std::size_t> offsets;

    /** Room per query in the keys and scores: how many pairs a query keeps at most. */
    std::size_t stride = 0;

    /** The queries the window cuts. */
    std::size_t count() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }

    /** Where query @p query 's run stages on a host tape before @c publish_runs: its stride past
     *  the window's first offset, which the runs of a window published fit under. */
    std::size_t staging_of(std::size_t query) const noexcept { return offsets.front() + query * stride; }

    /**
     *  @brief Why the window cannot take @p queries queries counted in @p scored.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The offsets or @p scored are not one per query, or the scores hold
     *    less room than the keys.
     */
    status_t refusal(std::size_t queries, std::span<std::size_t const> scored) const noexcept {
        if (count() != queries || scored.size() != queries || scores.size() < keys.size())
            return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** Publishes every offset after the first from the one before it, each query keeping the
     *  @p scored pairs it ranked, at most a stride, and moves each run from its staging down to
     *  its offset, in order, so a move overwrites only runs already moved. */
    void publish_runs(std::span<std::size_t const> scored) const noexcept {
        for (std::size_t query = 0; query != count(); ++query) {
            std::size_t const kept = min_of(scored[query], stride), start = offsets[query];
            std::size_t const staged = staging_of(query);
            offsets[query + 1] = start + kept;
            if (!kept || start == staged) continue;
            std::memmove(keys.data() + start, keys.data() + staged, kept * sizeof(key_type_));
            std::memmove(scores.data() + start, scores.data() + staged, kept * sizeof(score_type_));
        }
    }
};

/** The online-softmax carry of one attended row, in natural-log units - the C++ face of
 *  @c usearch_dense_softmax_state_t. */
struct softmax_state_t {

    /** Largest logit seen so far. */
    f32_t maximum_score;

    /** Sum of exponentials taken against that maximum. */
    f32_t partition;
};

/** Fixed 32-byte descriptor fronting every serialized dense index payload - volatile in-RAM
 *  snapshots, GPU-resident uploads, and persisted on-NVMe layouts alike. It describes the graph &
 *  its geometry, never the residency: what sections follow and how they are laid out is the
 *  per-storage contract versioned by @c version_k. */
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

    /** Scalar dimensions per vector - one half of the geometry. */
    u32_t dimensions;

    /** Neighbor IDs per node - the other half of the geometry. */
    u32_t connectivity;

    /** Published nodes in the payload; the chunk layout re-derives from it. */
    u32_t node_count;
};
static_assert(sizeof(index_header_t) == 32 && std::is_trivially_copyable_v<index_header_t>);

/** Whether @p scalar_type_ stores real numbers, as indexer scoring needs of its keys. */
template <typename scalar_type_>
concept floating_scalar = !nk::is_integral_dtype<scalar_type_>();

/** A lightning indexer, as DeepSeek V4 and GLM 5 pick sparse-attention keys: a node scores <tt>Σₕ
 *  wₕ·ReLU(qₕ·k)</tt> against these heads, and its distance is <tt>1 − score</tt>, while the walk
 *  navigates by the query rows, conventionally <tt>Σₕ wₕ·qₕ</tt>. */
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

/** A stored vector's sum and sum of squares in the dot product's dtype, which the compensated
 *  finalizes subtract and the norms column persists. */
template <typename dot_result_type_>
struct vector_norms {
    dot_result_type_ sum;
    dot_result_type_ sumsq;
};

/** The walks' default: no heads, so every node scores as the kit scored it. */
struct dummy_heads_t {
    dummy_heads_t row(std::size_t) const noexcept { return {}; }
};

/** Runs @p window over every @p window_rows rows of a batch of @p rows in order, as
 *  <tt>window(first, taking)</tt>, summing what each reports; each window writes its own rows'
 *  counts, wherever they live, and the first refusal stops the batch, keeping every window already
 *  written and leaving the rows past it unwritten. */
template <typename search_result_type_, typename window_type_>
expected<search_result_type_> for_each_window(std::size_t rows, std::size_t window_rows,
                                              window_type_&& window) noexcept {
    search_result_type_ total;
    for (std::size_t first = 0; first < rows; first += window_rows) {
        expected<search_result_type_> const searched = window(first, min_of(window_rows, rows - first));
        total.count += searched.value.count;
        total.visited_members += searched.value.visited_members;
        total.computed_distances += searched.value.computed_distances;
        total.restarts += searched.value.restarts;
        total.tail_compactions += searched.value.tail_compactions;
        total.tail_dropped += searched.value.tail_dropped;
        total.visited_saturations += searched.value.visited_saturations;
        if (!searched) return {total, searched.status};
    }
    return {total, status_t::success_k};
}

/**
 *  Why @p span cannot take @p rows copied rows of @p columns values each, packed within a row; rows
 *  may stride past @p columns, and wider rows keep their tails.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k The span's rows are not @p rows, or are narrower than @p columns.
 *  - @c unsupported_layout_k The values of a row are not packed.
 */
template <typename value_type_>
status_t rows_refusal(std::size_t rows, nk::matrix_span<value_type_> span, std::size_t columns) noexcept {
    if (span.extent(0) != rows || (rows && span.extent(1) < columns)) return status_t::config_mismatch_k;
    if (rows && span.stride_bytes(1) != sizeof(value_type_)) return status_t::unsupported_layout_k;
    return status_t::success_k;
}

/** Whether @p first and @p second walk by one kernel over one geometry, as two shards of a sharded
 *  index or the two sides of a copy between indexes must. */
template <typename index_type_>
bool same_geometry(index_type_ const& first, index_type_ const& second) noexcept {
    return first.kernel() == second.kernel() && first.geometry() == second.geometry();
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
 *  Every verb takes its input rows this way and imports them into its variant's scalar per
 *  thread, so a typed view converts and passes untouched when it already holds that scalar.
 *  Sub-byte rows fill whole bytes, as NumKong casts them and sub-byte indexes store them.
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

/** One candidate a MaxSim query ranks: its position among the plan's candidates and the sum of its
 *  per-query-row maxima, in NumKong's aggregation type for the stored scalar. */
template <typename total_type_>
struct maxsim_ranked {

    /** The candidate's position among the plan's candidates. */
    u32_t candidate = 0;

    /** Its summed maxima. */
    total_type_ total {};
};

static_assert(sizeof(maxsim_ranked<nk::f32_t>) == 8 && sizeof(maxsim_ranked<nk::i64_t>) == 16 &&
              sizeof(maxsim_ranked<nk::u64_t>) == 16);

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
 *  @brief Late-interaction scores from per-row rankings: each row's best dot per key, summed per
 *      key over its query, the largest an output row holds written largest first.
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

    /**
     *  Sizes the staging for queries of @p rows_per_query rankings, each @p width hits wide; a
     *  query wider than @c u32_t hits refuses with @c config_mismatch_k.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A query holds more hits than @c u32_t counts.
     */
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
     *  @param[out] scores Their summed maxima as @p score_type_, largest first, parallel to
     *      @p keys.
     *  @param[out] counts Keys written per query, one per query.
     *  @param[out] scored Keys scored per query, before the width of @p keys cut any.
     *  @return The keys written over every query.
     */
    template <typename score_type_>
    std::size_t operator()(nk::matrix_view<key_t> row_keys, nk::matrix_view<distance_t> row_distances,
                           std::span<std::size_t const> row_counts, nk::matrix_span<key_t> keys,
                           nk::matrix_span<score_type_> scores, std::span<std::size_t> counts,
                           std::span<std::size_t> scored) noexcept {
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
                keys(query, rank) = staged[rank].key,
                            scores(query, rank) = static_cast<score_type_>(staged[rank].score);
            counts[query] = kept;
            scored[query] = unique;
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

/**
 *  @brief Why @p keys and @p scores cannot hold @p queries rankings in contiguous rows.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k The row counts are not @p queries, or the widths differ.
 *  - @c unsupported_layout_k A row of keys or scores is not contiguous.
 */
template <typename key_type_, typename score_type_>
status_t rankings_refusal(std::size_t queries, nk::matrix_span<key_type_> keys,
                          nk::matrix_span<score_type_> scores) noexcept {
    if (keys.extent(0) != queries || scores.extent(0) != queries || scores.extent(1) != keys.extent(1))
        return status_t::config_mismatch_k;
    if (keys.stride_bytes(1) != sizeof(key_type_) || scores.stride_bytes(1) != sizeof(score_type_))
        return status_t::unsupported_layout_k;
    return status_t::success_k;
}

} // namespace unum::usearch::dense
