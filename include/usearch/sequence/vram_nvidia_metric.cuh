/**
 *  @file include/usearch/sequence/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device kits for the sequence VRAM indexes, the GPU twin of @c ram_metric.hpp, their
 *      ladders, the kernels, and the members of @c vram_nvidia_index.hpp reading them.
 *
 *  A kit binds one sequence distance to one GPU generation, named like the dense kits; the
 *  generation slot names the oldest tier whose instructions it needs. Levenshtein and overlap need
 *  nothing past the @c ampere floor: StringZilla's grouped Myers scorers and its rolling window
 *  hashes run on any NVIDIA device, so the distances are the same integers the CPU kits score.
 *
 *  A Levenshtein kit scores on a rung, the Myers words of the longest query its kernels serve,
 *  which each kernel takes beside the ladder's inputs, so a rung's kernels only reach the scorers
 *  of the word counts below it and keep their registers as narrow as the queries allow. Every
 *  scorer is out of line, one body per word count and symbol in the unit, whatever number of walks
 *  call it. The overlap kit scores on one rung, a warp to a row. Only the modality's NVIDIA unit
 *  compiles this file, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <cstring> // `std::memcpy` - lane zero's prepared overlap query, a word at a time

#include <bit>         // `std::bit_ceil`, `std::popcount` - a row's distinct pages and bytes
#include <iterator>    // `std::size` - every rung's kernels
#include <limits>      // `std::numeric_limits` - the widths and rows a `u32_t` states
#include <span>        // `std::span` - the overlap kind's window widths
#include <type_traits> // `std::conditional_t` - the kit a sequence kind scores with
#include <utility>     // `std::make_integer_sequence` - the word counts, every rung's kernels

#include <numkong/types.h>                  // `NUMKONG_ARCH_CUDA_AMPERE_` - the kits this unit carries
#include <stringzilla/cuda.cuh>             // `sz_group_shuffle_up_cuda_` & kin - the lane steps
#include <stringzilla/levenshtein/simt.cuh> // `sz_levenshtein_group_score_simt_` & the query preparation
#include <stringzilla/overlap/simt.cuh>     // `sz_overlap_query_prepare_simt_` & the candidate matches

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"        // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.cuh" // `basic::cuda_kernel_body`
#include "usearch/sequence/ram_metric.hpp"     // `levenshtein_ladder`, `overlap_ladder`
#include "usearch/sequence/vram_nvidia_index.hpp"

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

#pragma region Kits

/** Lanes scoring one candidate for queries of up to @p words Myers words: one up to sixteen,
 *  which a lane's registers hold and where a lane outruns a split group, and a group of four past
 *  them. */
constexpr sz_levenshtein_group_lanes_t cuda_levenshtein_lanes_per_row_ampere(u32_t words) noexcept {
    return words > sz_levenshtein_thread_words_max_simt_k ? sz_levenshtein_group_quad_k : sz_levenshtein_group_thread_k;
}

/** Bytes the @p bytes long row at @p data takes once prepared over @p symbol_, a multiple of 16: a
 *  class map over every byte value, or a page table with a class row per page its runes touch, then
 *  a mask row per class the row's own words long, one class per distinct symbol and one more. */
template <sz_levenshtein_symbol_t symbol_>
std::size_t cuda_levenshtein_span_bytes_ampere(byte_t const* data, std::size_t bytes) noexcept {
    if constexpr (symbol_ == sz_levenshtein_runes_k) {
        // Distinct pages in a bitset over every page, distinct runes in an open table twice the
        // longest row's runes wide, cleared only as wide as this row needs.
        constexpr std::size_t table_slots_k = 2 * 64 * 64;
        u64_t pages[sz_levenshtein_utf8_pages_k / 64] = {};
        u32_t table[table_slots_k];
        std::size_t const slots = min_of<std::size_t>(std::bit_ceil(2 * bytes + 2), table_slots_k);
        for (std::size_t slot = 0; slot != slots; ++slot) table[slot] = ~0u;
        sz_size_t runes = 0, rows = 1, classes = 1;
        for (sz_size_t position = 0; position < bytes; ++runes) {
            sz_rune_t const rune = sz_utf8_next_rune_(reinterpret_cast<sz_cptr_t>(data), bytes, &position);
            // A row past the widest rung is refused before it prepares, so it needs no count.
            if (bytes * 2 > table_slots_k) {
                ++classes;
                continue;
            }
            rows += !(pages[rune >> 14] >> ((rune >> 8) & 63u) & 1u);
            pages[rune >> 14] |= u64_t {1} << ((rune >> 8) & 63u);
            std::size_t slot = (rune * 2654435761u) & (slots - 1);
            while (table[slot] != ~0u && table[slot] != rune) slot = (slot + 1) & (slots - 1);
            classes += table[slot] == ~0u;
            table[slot] = rune;
        }
        return round_up_to_multiple<std::size_t>(
            sz_levenshtein_utf8_span_header_k + sz_levenshtein_utf8_pages_k * sizeof(u16_t) +
                rows * 256 * sizeof(u32_t) + classes * sz_levenshtein_query_words(runes) * sizeof(u64_t),
            16);
    }
    u64_t present[4] = {};
    for (std::size_t position = 0; position != bytes; ++position) {
        u8_t const value = static_cast<u8_t>(data[position]);
        present[value >> 6] |= u64_t {1} << (value & 63u);
    }
    std::size_t const distinct = std::popcount(present[0]) + std::popcount(present[1]) + std::popcount(present[2]) +
                                 std::popcount(present[3]);
    std::size_t const classes = min_of<std::size_t>(distinct + 1, sz_levenshtein_byte_classes_k);
    return round_up_to_multiple<std::size_t>(
        sz_levenshtein_byte_classes_k + classes * sz_levenshtein_query_words(bytes) * sizeof(u64_t), 16);
}

/** Lays @p row out as a query over @p symbol_ in @p span, every lane of the warp taking part. */
template <sz_levenshtein_symbol_t symbol_>
__device__ __forceinline__ sz_levenshtein_query_t cuda_levenshtein_prepare_ampere(basic::vram_row_t row, byte_t* span,
                                                                                  u32_t lane) noexcept {
    sz_levenshtein_query_t query;
    if constexpr (symbol_ == sz_levenshtein_runes_k)
        sz_levenshtein_query_prepare_utf8_simt_(reinterpret_cast<sz_cptr_t>(row.data), row.bytes,
                                                reinterpret_cast<sz_ptr_t>(span), &query, lane);
    else
        sz_levenshtein_query_prepare_simt_(reinterpret_cast<sz_cptr_t>(row.data), row.bytes,
                                           sz_levenshtein_query_words(row.bytes), reinterpret_cast<sz_ptr_t>(span),
                                           &query, lane);
    return query;
}

/** No edit sequence is shorter than the two lengths' difference; a row of @c b bytes holds between
 *  a quarter of @c b runes and @c b of them. */
template <sz_levenshtein_symbol_t symbol_>
__device__ __forceinline__ distance_t cuda_levenshtein_bound_ampere(sz_levenshtein_query_t const& query,
                                                                    basic::vram_row_t row) noexcept {
    u64_t const fewest = symbol_ == sz_levenshtein_runes_k ? divide_round_up<u64_t>(row.bytes, 4) : row.bytes;
    u64_t const most = row.bytes;
    return static_cast<distance_t>(fewest > query.length ? fewest - query.length
                                   : query.length > most ? query.length - most
                                                         : 0);
}

/** The distance over @p symbol_ from @p query to @p row, scored by a group of @p lanes_ lanes
 *  owning @p lane_words_ Myers words apiece, exact within @p limit and the running score past it.
 *  Out of line, so each body compiles once per unit, whatever number of walks call it. */
template <sz_levenshtein_symbol_t symbol_, sz_levenshtein_group_lanes_t lanes_, u32_t lane_words_>
__device__ __noinline__ sz_ssize_t cuda_levenshtein_group_ampere(sz_levenshtein_query_t query, basic::vram_row_t row,
                                                                 sz_ssize_t limit, u32_t lane) noexcept {
    sz_cptr_t const text = reinterpret_cast<sz_cptr_t>(row.data);
    if constexpr (symbol_ == sz_levenshtein_runes_k)
        return sz_levenshtein_group_score_utf8_simt_(&query, text, row.bytes, lane_words_, lanes_, limit, lane % lanes_,
                                                     sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_);
    else
        return sz_levenshtein_group_score_simt_(&query, text, row.bytes, lane_words_, lanes_, limit, lane % lanes_,
                                                sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_);
}

/** Scores a lone lane's candidate at the query's exact word count, through the scorer of that count
 *  among @p counts_ ; the host never launches a rung narrower than its longest query. */
template <sz_levenshtein_symbol_t symbol_, u32_t... counts_>
__device__ __forceinline__ sz_ssize_t
cuda_levenshtein_thread_ampere(sz_levenshtein_query_t const& query, basic::vram_row_t row, sz_ssize_t limit,
                               std::integer_sequence<u32_t, counts_...>) noexcept {
    sz_ssize_t distance = 0;
    u32_t const words = static_cast<u32_t>(sz_levenshtein_query_words(query.length));
    [[maybe_unused]] bool const scored =
        ((words == counts_ + 1 &&
          (distance = cuda_levenshtein_group_ampere<symbol_, sz_levenshtein_group_thread_k, counts_ + 1>(query, row,
                                                                                                         limit, 0),
           true)) ||
         ...);
    return distance;
}

/**
 *  @brief Scores over @p symbol_ for queries of up to @p words_ Myers words, exact within @p radius
 *      and the running score past it; the lanes of a group return the same.
 *
 *  Up to sixteen words a lane scores a candidate alone at the query's exact word count, so the
 *  verticals stay in registers. Past them the four lanes of a group split a candidate's words, a
 *  quarter of the rung's apiece, and a block's four warps score a hop's candidates together. An
 *  empty side scores the other's symbols, a UTF-8 row's counted in runes.
 */
template <sz_levenshtein_symbol_t symbol_, u32_t words_>
__device__ __forceinline__ distance_t cuda_levenshtein_score_ampere(sz_levenshtein_query_t const& query,
                                                                    basic::vram_row_t row, distance_t radius,
                                                                    u32_t lane) noexcept {
    if (!query.length || !row.bytes) {
        sz_size_t symbols = row.bytes;
        if constexpr (symbol_ == sz_levenshtein_runes_k) {
            symbols = 0;
            for (sz_size_t position = 0; position < row.bytes; ++symbols)
                sz_utf8_next_rune_(reinterpret_cast<sz_cptr_t>(row.data), row.bytes, &position);
        }
        return static_cast<distance_t>(query.length + symbols);
    }
    sz_ssize_t const limit = radius < 0x1p62f ? static_cast<sz_ssize_t>(radius) : STRINGZILLA_SSIZE_MAX;
    constexpr sz_levenshtein_group_lanes_t lanes_k = cuda_levenshtein_lanes_per_row_ampere(words_);
    if constexpr (lanes_k == sz_levenshtein_group_thread_k)
        return static_cast<distance_t>(
            cuda_levenshtein_thread_ampere<symbol_>(query, row, limit, std::make_integer_sequence<u32_t, words_> {}));
    else
        return static_cast<distance_t>(
            cuda_levenshtein_group_ampere<symbol_, lanes_k, words_ / lanes_k>(query, row, limit, lane));
}

/** Levenshtein over bytes on the Ampere floor, through StringZilla's grouped Myers scorers, the
 *  query's classes and masks in the span its walk prepares. Serves queries of up to 64 symbols a
 *  word. */
struct u8_ampere_levenshtein_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;

    using warp_t = cuda_warp_t;
    using query_t = sz_levenshtein_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    static constexpr u32_t lanes_per_row(u32_t words) noexcept { return cuda_levenshtein_lanes_per_row_ampere(words); }
    static constexpr bool serves(u32_t words, std::size_t bytes) noexcept { return bytes <= std::size_t {words} * 64; }
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes) noexcept {
        return cuda_levenshtein_span_bytes_ampere<sz_levenshtein_bytes_k>(data, bytes);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return cuda_levenshtein_prepare_ampere<sz_levenshtein_bytes_k>(row, span, lane);
    }
    static __device__ __forceinline__ distance_t bound(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_row_t row) noexcept {
        return cuda_levenshtein_bound_ampere<sz_levenshtein_bytes_k>(query, row);
    }
    template <u32_t words_>
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query, basic::vram_row_t row,
                                                       distance_t radius, u32_t lane) noexcept {
        return cuda_levenshtein_score_ampere<sz_levenshtein_bytes_k, words_>(query, row, radius, lane);
    }
};

/** Levenshtein over UTF-8 runes on the Ampere floor, each rune's class from the page table its
 *  query's span holds. A row of @c b bytes holds at most @c b runes, so the byte kit's rows bound
 *  its rungs too. */
struct utf8_ampere_levenshtein_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;

    using warp_t = cuda_warp_t;
    using query_t = sz_levenshtein_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    static constexpr u32_t lanes_per_row(u32_t words) noexcept { return cuda_levenshtein_lanes_per_row_ampere(words); }
    static constexpr bool serves(u32_t words, std::size_t bytes) noexcept { return bytes <= std::size_t {words} * 64; }
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes) noexcept {
        return cuda_levenshtein_span_bytes_ampere<sz_levenshtein_runes_k>(data, bytes);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        return cuda_levenshtein_prepare_ampere<sz_levenshtein_runes_k>(row, span, lane);
    }
    static __device__ __forceinline__ distance_t bound(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_row_t row) noexcept {
        return cuda_levenshtein_bound_ampere<sz_levenshtein_runes_k>(query, row);
    }
    template <u32_t words_>
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query, basic::vram_row_t row,
                                                       distance_t radius, u32_t lane) noexcept {
        return cuda_levenshtein_score_ampere<sz_levenshtein_runes_k, words_>(query, row, radius, lane);
    }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cuda_k) != 0)
struct levenshtein_ladder<sz_levenshtein_bytes_k, caps_> {
    using kit_t = typename highest_kit<caps_, u8_ampere_levenshtein_kit_t>::type;
    static constexpr sz_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cuda_k) != 0)
struct levenshtein_ladder<sz_levenshtein_runes_k, caps_> {
    using kit_t = typename highest_kit<caps_, utf8_ampere_levenshtein_kit_t>::type;
    static constexpr sz_capability_t picked_k = caps_ & kit_t::capability_k;
};

/** The full scale of an overlap distance, as the CPU's @c overlap_metric::resolution_k. */
inline constexpr u64_t cuda_overlap_resolution_k = u64_t {1} << 31;

/**
 *  @brief The overlap distance from @p query to @p row, every lane of the warp taking part: the
 *      share of the longer side's windows the candidate misses, at @c cuda_overlap_resolution_k,
 *      exact within @p radius and above it past.
 *
 *  A float radius rounds, so the misses StringZilla may stop past are those of the integer radius a
 *  float spacing above it, @c 2^7 at this scale. Out of line, so the body compiles once per unit.
 */
__device__ __noinline__ inline u64_t cuda_overlap_score_ampere(sz_overlap_query_t query, basic::vram_row_t row,
                                                               distance_t radius, u32_t lane) noexcept {
    u64_t const candidate_windows = sz_overlap_windows(row.bytes, query.widths, query.widths_count);
    u64_t const longer = max_of<u64_t>(candidate_windows, query.windows);
    if (!longer) return cuda_overlap_resolution_k;
    sz_size_t misses_limit = STRINGZILLA_SIZE_MAX;
    if (radius < static_cast<distance_t>(cuda_overlap_resolution_k)) {
        u64_t const admitted = static_cast<u64_t>(radius > 0 ? radius : 0) + (u64_t {1} << 7);
        misses_limit = ((admitted + 1) * longer - 1) / cuda_overlap_resolution_k;
    }
    u64_t const matches = sz_overlap_candidate_matches_simt_(&query, reinterpret_cast<sz_cptr_t>(row.data), row.bytes,
                                                             misses_limit, lane % 32u, sz_shuffle_index_cuda_);
    return (longer - matches) * cuda_overlap_resolution_k / longer;
}

/** Window overlap on the Ampere floor: a warp lays the query's window hashes at every width out as
 *  a tree in its span, and a warp counts each candidate's windows the tree holds, through
 *  StringZilla's rolling hashes, whose residues are the CPU kits'. The widths are the parameters
 *  the face uploads, their count first. */
struct u8_ampere_overlap_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    inline static constexpr u32_t lanes_per_row_k = cuda_warp_t::width_k;

    using warp_t = cuda_warp_t;
    using query_t = sz_overlap_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** A warp scores a row of any length its byte count can state. */
    static constexpr bool serves(u32_t, std::size_t bytes) noexcept {
        return bytes <= std::numeric_limits<u32_t>::max();
    }
    static std::size_t span_bytes(std::span<u32_t const> widths, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(sz_overlap_query_span_bytes(bytes, widths.data(), widths.size()), 16);
    }

    /** Prepares @p row with the warp, then hands every lane lane zero's query, the only one
     *  StringZilla records. */
    static __device__ __forceinline__ query_t prepare(byte_t const* parameters, basic::vram_row_t row, byte_t* span,
                                                      u32_t lane) noexcept {
        static_assert(sizeof(query_t) % sizeof(u32_t) == 0, "The query crosses lanes a word at a time");
        u32_t const* const widths = reinterpret_cast<u32_t const*>(parameters);
        query_t query;
        sz_overlap_query_prepare_simt_(reinterpret_cast<sz_cptr_t>(row.data), row.bytes, widths + 1, widths[0],
                                       reinterpret_cast<sz_ptr_t>(span), &query, lane, sz_shuffle_up_cuda_,
                                       sz_shuffle_index_cuda_, sz_lanes_ballot_cuda_);
        u32_t words[sizeof(query_t) / sizeof(u32_t)];
        std::memcpy(words, &query, sizeof(query_t));
        for (u32_t& word : words) word = warp_t::shuffle(word, 0);
        std::memcpy(&query, words, sizeof(query_t));
        return query;
    }

    /** No candidate matches more windows than the shorter side holds. */
    static __device__ __forceinline__ distance_t bound(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_row_t row) noexcept {
        u64_t const candidate_windows = sz_overlap_windows(row.bytes, query.widths, query.widths_count);
        u64_t const longer = max_of<u64_t>(candidate_windows, query.windows);
        u64_t const shorter = min_of<u64_t>(candidate_windows, query.windows);
        if (!longer) return static_cast<distance_t>(cuda_overlap_resolution_k);
        return static_cast<distance_t>((longer - shorter) * cuda_overlap_resolution_k / longer);
    }
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query, basic::vram_row_t row,
                                                       distance_t radius, u32_t lane) noexcept {
        return static_cast<distance_t>(cuda_overlap_score_ampere(query, row, radius, lane));
    }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cuda_k) != 0)
struct overlap_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, u8_ampere_overlap_kit_t>::type;
    static constexpr sz_capability_t picked_k = caps_ & kit_t::capability_k;
};

/** The kit scoring the sequence kind @p metric_ under @p caps_: the overlap ladder's, or the
 *  Levenshtein ladder's over the kind's symbols. */
template <usearch_sequence_metric_t metric_, sz_capability_t caps_>
using vram_sequence_kit_t = std::conditional_t<
    metric_ == usearch_sequence_metric_overlap_k, typename overlap_ladder<caps_>::kit_t,
    typename levenshtein_ladder<metric_ == usearch_sequence_metric_levenshtein_utf8_k ? sz_levenshtein_runes_k
                                                                                      : sz_levenshtein_bytes_k,
                                caps_>::kit_t>;

#pragma endregion Kits

#pragma region Face

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
status_t vram_nvidia_index<metric_, node_id_type_, capability_, kernels_type_>::window_widths(
    std::span<std::size_t const> widths) noexcept {
    if ((metric_ == usearch_sequence_metric_overlap_k) == widths.empty() || this->connectivity())
        return status_t::config_mismatch_k;
    if (widths.empty()) return status_t::success_k;
    // The count leads, as the kit reads the parameters.
    if (status_t const reserved = window_widths_.reserve_exactly(widths.size() + 1); failed(reserved)) return reserved;
    window_widths_.clear();
    window_widths_.push_back_reserved(static_cast<u32_t>(widths.size()));
    for (std::size_t const width : widths) {
        if (!width || width > std::numeric_limits<u32_t>::max()) return status_t::config_mismatch_k;
        window_widths_.push_back_reserved(static_cast<u32_t>(width));
    }
    return status_t::success_k;
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
std::size_t vram_nvidia_index<metric_, node_id_type_, capability_, kernels_type_>::rung_serving_(
    std::size_t longest) noexcept {
    using kit_t = vram_sequence_kit_t<metric_, capability_>;
    constexpr std::span<u32_t const> rungs = vram_sequence_rungs(metric_);
    std::size_t rung = 0;
    while (rung != rungs_k && !kit_t::serves(rungs[rung], longest)) ++rung;
    return rung;
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
u32_t vram_nvidia_index<metric_, node_id_type_, capability_, kernels_type_>::lanes_per_row_(std::size_t rung) noexcept {
    using kit_t = vram_sequence_kit_t<metric_, capability_>;
    if constexpr (metric_ == usearch_sequence_metric_overlap_k) return kit_t::lanes_per_row_k;
    else {
        constexpr std::span<u32_t const> rungs = vram_sequence_rungs(metric_);
        return kit_t::lanes_per_row(rungs[rung]);
    }
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
std::size_t vram_nvidia_index<metric_, node_id_type_, capability_, kernels_type_>::span_bytes_(
    byte_t const* data, std::size_t bytes) const noexcept {
    using kit_t = vram_sequence_kit_t<metric_, capability_>;
    if constexpr (metric_ == usearch_sequence_metric_overlap_k)
        return kit_t::span_bytes(to_span(window_widths_).subspan(1), bytes);
    else return kit_t::span_bytes(data, bytes);
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
status_t vram_nvidia_index<metric_, node_id_type_, capability_, kernels_type_>::load_kernels_(int device) noexcept {
    if (metric_ == usearch_sequence_metric_overlap_k && window_widths_.empty()) return status_t::uninitialized_k;
    auto [module, status] = cuda_module_t::load(device, kernels_type_::bytes());
    if (failed(status)) return status;
    module_ = std::move(module);
    vram_kernel_symbol_t symbol;
    constexpr std::span<u32_t const> rungs = vram_sequence_rungs(metric_);
    for (std::size_t rung = 0; rung != rungs_k; ++rung)
        for (basic::cuda_rung_t const instantiated : basic::cuda_rungs) {
            vram_kernel_symbol(symbol, "sequence", "cuda_kernel", {"vram_arguments", 4, itanium_scope_t::basic_k},
                               instantiated.kernel, capability_, metric_, rungs[rung], itanium_type<node_id_type_> {},
                               instantiated.slots);
            cuda_function_handle_t& function =
                kernels_[rung][static_cast<u32_t>(instantiated.kernel)][gpu_rung_index(instantiated.slots)];
            if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
        }
    if (window_widths_.empty()) return status_t::success_k;
    device_widths_ = cuda_make_unique(cuda_device_allocator<u32_t> {}, window_widths_.size());
    if (!device_widths_) return status_t::bad_alloc_k;
    if (cuda_copy(device_widths_.get(), window_widths_.data(), window_widths_.size() * sizeof(u32_t)) != cuda_success_k)
        return status_t::device_error_k;
    this->parameters_ = reinterpret_cast<byte_t const*>(device_widths_.get());
    return status_t::success_k;
}

#pragma endregion Face

#pragma region Kernels

/** Every NVIDIA sequence kernel: @p kernel_ picks the walk, @p capability_ and @p metric_ the kit
 *  through its ladder, @p words_ its rung, @p node_id_type_ the IDs of the records, and @p slots_
 *  the register rung. */
template <vram_kernel_t kernel_, sz_capability_t capability_, usearch_sequence_metric_t metric_, u32_t words_,
          typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, basic::cuda_resident_blocks(kernel_, slots_))
    cuda_kernel(basic::vram_arguments<node_id_type_> arguments) {
    using kit_t = vram_sequence_kit_t<metric_, capability_>;
    basic::cuda_kernel_body<kernel_, kit_t, words_, slots_>(arguments);
}

template <sz_capability_t capability_, usearch_sequence_metric_t metric_, u32_t words_, typename node_id_type_,
          std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<basic::cuda_rungs[rungs_].kernel, capability_, metric_, words_, node_id_type_,
                                    basic::cuda_rungs[rungs_].slots>),
     ...);
}

/** The rung at @p index of the kit of @p metric, as a template argument in device code. */
constexpr u32_t vram_sequence_rung(usearch_sequence_metric_t metric, std::size_t index) noexcept {
    std::span<u32_t const> const rungs = vram_sequence_rungs(metric);
    return rungs[index];
}

template <sz_capability_t capability_, usearch_sequence_metric_t metric_, typename node_id_type_, std::size_t... words_>
__device__ void cuda_instantiate_words(std::index_sequence<words_...>) noexcept {
    (cuda_instantiate_rungs<capability_, metric_, vram_sequence_rung(metric_, words_), node_id_type_>(
         std::make_index_sequence<std::size(basic::cuda_rungs)> {}),
     ...);
}

/** Instantiates every kernel of every rung of the kind @p metric_ under @p capability_ with
 *  @p node_id_type_ IDs; the unit compiling the fatbin names it once per kind and node-ID type.
 *  Taking the kernels' addresses in device code makes NVCC emit them, while a host-only pass emits
 *  no launch stub. */
template <sz_capability_t capability_, usearch_sequence_metric_t metric_, typename node_id_type_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_words<capability_, metric_, node_id_type_>(
        std::make_index_sequence<vram_sequence_rungs(metric_).size()> {});
}

#pragma endregion Kernels

} // namespace unum::usearch::sequence
