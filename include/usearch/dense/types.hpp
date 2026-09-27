/**
 *  @file include/usearch/dense/types.hpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief The dense tier's vocabulary, shared by its RAM, VRAM & NVMe indexes.
 */
#pragma once

#include <cstring> // `std::memcmp` - keys compare by their bytes, whatever their type

#include <algorithm> // `std::make_heap`, `std::sort_heap`, `std::push_heap`, `std::pop_heap` - the MaxSim selection
#include <limits>    // `std::numeric_limits` - the reducer's hit ceiling & unscored rows
#include <span>      // `std::span` - the reducer's rankings & outputs

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the reducer's staging
#include "usearch/utilities/ram_sorts.hpp"   // `radix_group` - the reducer's grouping

namespace unum {
namespace usearch {

/** Dense node identifier within one sub-index - 32 bits, matching the 4-billion ceiling. */
using dense_node_id_t = u32_t;

/** Distance representation shared by all metrics, regardless of the stored dtype. */
using dense_distance_t = f32_t;

/** Reserved all-ones value, never a valid node id within a sub-index. Doubles as the
 *  lock & empty-slot sentinel in neighbor lists of every index design. */
inline constexpr dense_node_id_t dense_node_id_unknown_k = 0xFFFFFFFFu;

/** One traversal candidate: a node id paired with its distance to the query. */
struct dense_scored_node_t {

    /** Identifier of the scored node within its sub-index. */
    dense_node_id_t node_id;

    /** Distance to the query, in the instantiated kernel's metric. */
    dense_distance_t distance;
};

/** Orders scored nodes nearest-first; the comparator of the sorted results pool. */
struct dense_closer_scored_node_t {
    inline bool operator()(dense_scored_node_t first, dense_scored_node_t second) const noexcept {
        return first.distance < second.distance;
    }
};

/** Orders scored nodes furthest-first; the comparator of the max-heap frontier. */
struct dense_further_scored_node_t {
    inline bool operator()(dense_scored_node_t first, dense_scored_node_t second) const noexcept {
        return first.distance > second.distance;
    }
};

/** Fixed 32-byte descriptor fronting every serialized dense index payload - volatile
 *  in-RAM snapshots, GPU-resident uploads, and persisted on-NVMe layouts alike. It describes
 *  the graph & its frozen geometry, never the residency: what sections follow and how they
 *  are laid out is the per-storage contract versioned by @c version_k. */
struct dense_index_header_t {

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

    /** The @c nk_kernel_kind_t metric; must match the loading instantiation. */
    u32_t kernel;

    /** Scalar dimensions per vector - one half of the frozen geometry. */
    u32_t dimensions;

    /** Neighbor ids per node - the other half of the frozen geometry. */
    u32_t connectivity;

    /** Published nodes in the payload; the chunk layout re-derives from it. */
    u32_t node_count;
};
static_assert(sizeof(dense_index_header_t) == 32 && std::is_trivially_copyable_v<dense_index_header_t>);

/** One ranked hit while @c dense_maxsim_reducer folds a query, then one document's summed maxima. */
template <typename key_type_>
struct dense_maxsim_entry {

    /** The document the ranked token belongs to. */
    key_type_ key;

    /** The query row that ranked it, counted within its query. */
    u32_t row;

    /** Its dot against that row, then the document's summed per-row maxima once the query folds. */
    f64_t score;
};

/**
 *  @brief Late-interaction scores from per-row rankings: each row's best dot per key, summed per key over
 *      its query, the @c wanted largest landing largest first.
 *
 *  Every row is one query token's ranking under the inner kind, where a distance is @c 1 @c - @c dot, so a
 *  key a row never ranked adds nothing for that row. The staging is sized once by @c reserve, and both the
 *  grouping and the selection run in place, so a reduction never allocates.
 */
template <typename key_type_, typename allocator_type_>
class dense_maxsim_reducer {
  public:
    using key_t = key_type_;
    using entry_t = dense_maxsim_entry<key_t>;

    explicit dense_maxsim_reducer(allocator_type_ allocator) noexcept : staged_(allocator), row_best_(allocator) {}

    /** Sizes the staging for queries of @p rows_per_query rankings, each @p width hits wide, keeping
     *  @p wanted keys per query; a query wider than @c u32_t hits refuses with @c config_mismatch_k. */
    status_t reserve(std::size_t rows_per_query, std::size_t width, std::size_t wanted) noexcept {
        std::size_t const hits = rows_per_query * width;
        if (hits > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        rows_per_query_ = rows_per_query, width_ = width, wanted_ = wanted;
        if (status_t const sized = staged_.resize_uninitialized(hits); failed(sized)) return sized;
        if (status_t const sized = row_best_.resize_uninitialized(rows_per_query); failed(sized)) return sized;
        for (std::size_t row = 0; row != rows_per_query; ++row) row_best_[row] = unscored_k;
        return status_t::success_k;
    }

    /**
     *  @param[in] row_keys Keys of every row's ranking, @b [rows,width] row-major.
     *  @param[in] row_distances Distances of the same rankings, parallel to @p row_keys.
     *  @param[in] row_counts Hits ranked per row, at most the reserved width.
     *  @param[out] keys The best keys of each query, @b [queries,wanted] row-major.
     *  @param[out] scores Their summed maxima, largest first, parallel to @p keys.
     *  @param[out] counts Keys written per query, one per query.
     *  @return The keys written over every query.
     */
    template <typename distance_type_>
    std::size_t operator()(std::span<key_t const> row_keys, std::span<distance_type_ const> row_distances,
                           std::span<std::size_t const> row_counts, std::span<key_t> keys, std::span<f64_t> scores,
                           std::span<std::size_t> counts) noexcept {
        entry_t* const staged = staged_.data();
        std::size_t written = 0;
        for (std::size_t query = 0; query != counts.size(); ++query) {
            std::size_t ranked = 0;
            for (std::size_t row = 0; row != rows_per_query_; ++row) {
                std::size_t const ranking = query * rows_per_query_ + row;
                for (std::size_t hit = 0; hit != row_counts[ranking]; ++hit)
                    staged[ranked++] = {row_keys[ranking * width_ + hit], static_cast<u32_t>(row),
                                        1 - static_cast<f64_t>(row_distances[ranking * width_ + hit])};
            }

            // Grouped by key in place, each key's run adds every row's best dot once, restoring the row slots.
            radix_group(std::span<entry_t> {staged, ranked}, &entry_t::key);
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

            // A heap topped by the worst kept key selects the best `wanted` in place.
            std::size_t const kept = min_of(wanted_, unique);
            std::make_heap(staged, staged + kept, better_);
            for (std::size_t candidate = kept; kept && candidate != unique; ++candidate) {
                if (!better_(staged[candidate], staged[0])) continue;
                std::pop_heap(staged, staged + kept, better_);
                staged[kept - 1] = staged[candidate];
                std::push_heap(staged, staged + kept, better_);
            }
            std::sort_heap(staged, staged + kept, better_);
            for (std::size_t rank = 0; rank != kept; ++rank)
                keys[query * wanted_ + rank] = staged[rank].key, scores[query * wanted_ + rank] = staged[rank].score;
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
    std::size_t rows_per_query_ = 0, width_ = 0, wanted_ = 0;
};

} // namespace usearch
} // namespace unum
