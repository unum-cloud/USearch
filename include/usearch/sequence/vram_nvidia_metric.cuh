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
 *  call it. The overlap kit scores on one rung, a warp to a sequence. The Needleman-Wunsch kit
 *  scores a sequence with a warp, an anti-diagonal wavefront over stripes of query codes, each lane
 *  holding a rung's worth of rows in registers. Only the modality's NVIDIA unit compiles this file,
 *  so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <cstring> // `std::memcpy` - lane zero's prepared overlap query, a word at a time

#include <bit>         // `std::bit_ceil`, `std::popcount` - a sequence's distinct pages and bytes
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

/** Bytes the @p bytes long sequence at @p data takes once prepared over @p symbol_, a multiple of
 *  16: a class map over every byte value, or a page table with a class row per page its runes
 *  touch, then a mask row per class at the walk stride of the sequence's own words, one class per
 *  distinct symbol and one more. */
template <sz_levenshtein_symbol_t symbol_>
std::size_t cuda_levenshtein_span_bytes_ampere(byte_t const* data, std::size_t bytes) noexcept {
    if constexpr (symbol_ == sz_levenshtein_runes_k) {
        // Distinct pages in a bitset over every page, distinct runes in an open table twice the
        // widest rung's runes wide, cleared only as wide as this sequence needs.
        constexpr std::size_t table_slots_k = 2 * vram_levenshtein_widest_bytes_k;
        u64_t pages[sz_levenshtein_utf8_pages_k / 64] = {};
        u32_t table[table_slots_k];
        std::size_t const slots = min_of<std::size_t>(std::bit_ceil(2 * bytes + 2), table_slots_k);
        for (std::size_t slot = 0; slot != slots; ++slot) table[slot] = ~0u;
        sz_size_t runes = 0, rows = 1, classes = 1;
        for (sz_size_t position = 0; position < bytes; ++runes) {
            sz_rune_t const rune = sz_utf8_next_rune_(reinterpret_cast<sz_cptr_t>(data), bytes, &position);
            // A sequence past the widest rung is refused before it prepares, so it needs no count.
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
                rows * 256 * sizeof(u32_t) +
                classes * sz_levenshtein_walk_stride(sz_levenshtein_query_words(runes)) * sizeof(u64_t),
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
        sz_levenshtein_byte_classes_k +
            classes * sz_levenshtein_walk_stride(sz_levenshtein_query_words(bytes)) * sizeof(u64_t),
        16);
}

/** Lays @p sequence out as a query over @p symbol_ in @p span, every lane of the warp taking
 *  part. */
template <sz_levenshtein_symbol_t symbol_>
__device__ __forceinline__ sz_levenshtein_query_t cuda_levenshtein_prepare_ampere(basic::vram_entry_t sequence,
                                                                                  byte_t* span, u32_t lane) noexcept {
    sz_levenshtein_query_t query;
    if constexpr (symbol_ == sz_levenshtein_runes_k)
        sz_levenshtein_query_prepare_utf8_simt_(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes,
                                                reinterpret_cast<sz_ptr_t>(span), &query, lane);
    else
        sz_levenshtein_query_prepare_simt_(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes,
                                           sz_levenshtein_walk_stride(sz_levenshtein_query_words(sequence.bytes)),
                                           reinterpret_cast<sz_ptr_t>(span), &query, lane);
    return query;
}

/** No edit sequence is shorter than the two lengths' difference; a sequence of @c b bytes holds
 *  between a quarter of @c b runes and @c b of them. */
template <sz_levenshtein_symbol_t symbol_>
__device__ __forceinline__ distance_t cuda_levenshtein_bound_ampere(sz_levenshtein_query_t const& query,
                                                                    basic::vram_entry_t sequence) noexcept {
    u64_t const fewest = symbol_ == sz_levenshtein_runes_k ? divide_round_up<u64_t>(sequence.bytes, 4) : sequence.bytes;
    u64_t const most = sequence.bytes;
    return static_cast<distance_t>(fewest > query.length ? fewest - query.length
                                   : query.length > most ? query.length - most
                                                         : 0);
}

/** The distance over @p symbol_ from @p query to @p sequence, scored by a group of @p lanes_ lanes
 *  owning @p lane_words_ Myers words apiece, exact within @p limit and the running score past it.
 *  Out of line, so each body compiles once per unit, whatever number of walks call it. */
template <sz_levenshtein_symbol_t symbol_, sz_levenshtein_group_lanes_t lanes_, u32_t lane_words_>
__device__ __noinline__ sz_ssize_t cuda_levenshtein_group_ampere(sz_levenshtein_query_t query,
                                                                 basic::vram_entry_t sequence, sz_ssize_t limit,
                                                                 u32_t lane) noexcept {
    sz_cptr_t const text = reinterpret_cast<sz_cptr_t>(sequence.data);
    if constexpr (symbol_ == sz_levenshtein_runes_k)
        return sz_levenshtein_group_score_utf8_simt_(&query, text, sequence.bytes, lane_words_, lanes_, limit,
                                                     lane % lanes_, sz_group_shuffle_up_cuda_,
                                                     sz_group_shuffle_index_cuda_);
    else
        return sz_levenshtein_group_score_simt_(&query, text, sequence.bytes, lane_words_, lanes_, limit, lane % lanes_,
                                                sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_);
}

/** Scores a lone lane's candidate at the query's exact word count, through the scorer of that count
 *  among @p counts_ ; the host never launches a rung narrower than its longest query. */
template <sz_levenshtein_symbol_t symbol_, u32_t... counts_>
__device__ __forceinline__ sz_ssize_t
cuda_levenshtein_thread_ampere(sz_levenshtein_query_t const& query, basic::vram_entry_t sequence, sz_ssize_t limit,
                               std::integer_sequence<u32_t, counts_...>) noexcept {
    sz_ssize_t distance = 0;
    u32_t const words = static_cast<u32_t>(sz_levenshtein_query_words(query.length));
    [[maybe_unused]] bool const scored =
        ((words == counts_ + 1 &&
          (distance = cuda_levenshtein_group_ampere<symbol_, sz_levenshtein_group_thread_k, counts_ + 1>(
               query, sequence, limit, 0),
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
 *  empty side scores the other's symbols, a UTF-8 sequence's counted in runes.
 */
template <sz_levenshtein_symbol_t symbol_, u32_t words_>
__device__ __forceinline__ distance_t cuda_levenshtein_score_ampere(sz_levenshtein_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t radius,
                                                                    u32_t lane) noexcept {
    if (!query.length || !sequence.bytes) {
        sz_size_t symbols = sequence.bytes;
        if constexpr (symbol_ == sz_levenshtein_runes_k) {
            symbols = 0;
            for (sz_size_t position = 0; position < sequence.bytes; ++symbols)
                sz_utf8_next_rune_(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes, &position);
        }
        return static_cast<distance_t>(query.length + symbols);
    }
    sz_ssize_t const limit = radius < 0x1p62f ? static_cast<sz_ssize_t>(radius) : STRINGZILLA_SSIZE_MAX;
    constexpr sz_levenshtein_group_lanes_t lanes_k = cuda_levenshtein_lanes_per_row_ampere(words_);
    if constexpr (lanes_k == sz_levenshtein_group_thread_k)
        return static_cast<distance_t>(cuda_levenshtein_thread_ampere<symbol_>(
            query, sequence, limit, std::make_integer_sequence<u32_t, words_> {}));
    else
        return static_cast<distance_t>(
            cuda_levenshtein_group_ampere<symbol_, lanes_k, words_ / lanes_k>(query, sequence, limit, lane));
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
    static constexpr bool serves(u32_t words, std::size_t bytes) noexcept {
        return bytes <= vram_levenshtein_bytes(words);
    }
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes) noexcept {
        return cuda_levenshtein_span_bytes_ampere<sz_levenshtein_bytes_k>(data, bytes);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_entry_t sequence, byte_t* span, u32_t lane) noexcept {
        return cuda_levenshtein_prepare_ampere<sz_levenshtein_bytes_k>(sequence, span, lane);
    }
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t, query_t const& query,
                                                             basic::vram_entry_t sequence) noexcept {
        return cuda_levenshtein_bound_ampere<sz_levenshtein_bytes_k>(query, sequence);
    }
    template <u32_t words_>
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_entry_t sequence, distance_t radius,
                                                       u32_t lane) noexcept {
        return cuda_levenshtein_score_ampere<sz_levenshtein_bytes_k, words_>(query, sequence, radius, lane);
    }
};

/** Levenshtein over UTF-8 runes on the Ampere floor, each rune's class from the page table its
 *  query's span holds. A sequence of @c b bytes holds at most @c b runes, so the byte kit's
 *  sequences bound its rungs too. */
struct utf8_ampere_levenshtein_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;

    using warp_t = cuda_warp_t;
    using query_t = sz_levenshtein_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    static constexpr u32_t lanes_per_row(u32_t words) noexcept { return cuda_levenshtein_lanes_per_row_ampere(words); }
    static constexpr bool serves(u32_t words, std::size_t bytes) noexcept {
        return bytes <= vram_levenshtein_bytes(words);
    }
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes) noexcept {
        return cuda_levenshtein_span_bytes_ampere<sz_levenshtein_runes_k>(data, bytes);
    }

    static __device__ __forceinline__ query_t prepare(basic::vram_entry_t sequence, byte_t* span, u32_t lane) noexcept {
        return cuda_levenshtein_prepare_ampere<sz_levenshtein_runes_k>(sequence, span, lane);
    }
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t, query_t const& query,
                                                             basic::vram_entry_t sequence) noexcept {
        return cuda_levenshtein_bound_ampere<sz_levenshtein_runes_k>(query, sequence);
    }
    template <u32_t words_>
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_entry_t sequence, distance_t radius,
                                                       u32_t lane) noexcept {
        return cuda_levenshtein_score_ampere<sz_levenshtein_runes_k, words_>(query, sequence, radius, lane);
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

/**
 *  @brief The overlap distance from @p query to @p sequence, the aligned group of @p lanes lanes
 *      holding @p lane taking part: the share of the longer side's windows the candidate misses, at
 *      @c overlap_resolution_k, exact within @p radius and above it past.
 *
 *  A float radius rounds, so the misses StringZilla may stop past are those of the integer radius a
 *  float spacing above it, @c 2^7 at this scale. Out of line, so the body compiles once per unit.
 */
__device__ __noinline__ inline u64_t cuda_overlap_score_ampere(sz_overlap_query_t query, basic::vram_entry_t sequence,
                                                               distance_t radius, u32_t lane, u32_t lanes) noexcept {
    u64_t const candidate_windows = sz_overlap_windows(sequence.bytes, query.widths, query.widths_count);
    u64_t const longer = max_of<u64_t>(candidate_windows, query.windows);
    if (!longer) return overlap_distance(longer, 0);
    sz_size_t misses_limit = STRINGZILLA_SIZE_MAX;
    if (radius < static_cast<distance_t>(overlap_resolution_k)) {
        u64_t const admitted = static_cast<u64_t>(radius > 0 ? radius : 0) + (u64_t {1} << 7);
        misses_limit = ((admitted + 1) * longer - 1) / static_cast<u64_t>(overlap_resolution_k);
    }
    u64_t const matches = sz_overlap_candidate_matches_simt_(&query, reinterpret_cast<sz_cptr_t>(sequence.data),
                                                             sequence.bytes, misses_limit, lane % lanes, lanes,
                                                             sz_group_shuffle_index_cuda_);
    return overlap_distance(longer, matches);
}

/** Window overlap on the Ampere floor: a warp lays the query's window hashes at every width out as
 *  a tree in its span, and a group of a rung's lanes counts each candidate's windows the tree
 *  holds, through StringZilla's rolling hashes, whose residues are the CPU kits'. The widths are
 *  the parameters the face uploads, their count first. */
struct u8_ampere_overlap_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;

    using warp_t = cuda_warp_t;
    using query_t = sz_overlap_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** A rung is its lanes per sequence. */
    static constexpr u32_t lanes_per_row(u32_t lanes) noexcept { return lanes; }
    static_assert(vram_overlap_rungs_k[std::size(vram_overlap_rungs_k) - 1] <= warp_t::width_k);

    /** Every rung scores a sequence of any length its byte count can state. */
    static constexpr bool serves(u32_t, std::size_t bytes) noexcept {
        return bytes <= std::numeric_limits<u32_t>::max();
    }
    static std::size_t span_bytes(std::span<u32_t const> widths, std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(sz_overlap_query_span_bytes(bytes, widths.data(), widths.size()), 16);
    }

    /** Prepares @p sequence with the warp, then hands every lane lane zero's query, the only one
     *  StringZilla records. */
    static __device__ __forceinline__ query_t prepare(byte_t const* parameters, basic::vram_entry_t sequence,
                                                      byte_t* span, u32_t lane) noexcept {
        static_assert(sizeof(query_t) % sizeof(u32_t) == 0, "The query crosses lanes a word at a time");
        u32_t const* const widths = reinterpret_cast<u32_t const*>(parameters);
        query_t query;
        sz_overlap_query_prepare_simt_(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes, widths + 1,
                                       widths[0], reinterpret_cast<sz_ptr_t>(span), &query, lane, sz_shuffle_up_cuda_,
                                       sz_shuffle_index_cuda_, sz_lanes_ballot_cuda_);
        u32_t words[sizeof(query_t) / sizeof(u32_t)];
        std::memcpy(words, &query, sizeof(query_t));
        for (u32_t& word : words) word = warp_t::shuffle(word, 0);
        std::memcpy(&query, words, sizeof(query_t));
        return query;
    }

    /** No candidate matches more windows than the shorter side holds. */
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t, query_t const& query,
                                                             basic::vram_entry_t sequence) noexcept {
        u64_t const candidate_windows = sz_overlap_windows(sequence.bytes, query.widths, query.widths_count);
        u64_t const longer = max_of<u64_t>(candidate_windows, query.windows);
        u64_t const shorter = min_of<u64_t>(candidate_windows, query.windows);
        return static_cast<distance_t>(overlap_distance(longer, shorter));
    }
    template <u32_t lanes_>
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_entry_t sequence, distance_t radius,
                                                       u32_t lane) noexcept {
        return static_cast<distance_t>(cuda_overlap_score_ampere(query, sequence, radius, lane, lanes_));
    }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cuda_k) != 0)
struct overlap_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, u8_ampere_overlap_kit_t>::type;
    static constexpr sz_capability_t picked_k = caps_ & kit_t::capability_k;
};

/** Bytes of a Needleman-Wunsch profile stripe: a 64-bit word of costs per class and lane. */
inline constexpr std::size_t cuda_needleman_wunsch_stripe_bytes_k = needleman_wunsch_classes_k * cuda_warp_t::width_k *
                                                                    sizeof(u64_t);

/** Codes of the longest sequence the widest Needleman-Wunsch rung scores, as many as the widest
 *  Levenshtein rung's bytes; longer ones are refused. */
inline constexpr std::size_t cuda_needleman_wunsch_widest_codes_k = vram_levenshtein_widest_bytes_k;

/** Query codes each lane holds for a query of @p codes: the fewest that fit the query in one
 *  stripe of a warp's lanes, the widest rung's past it. */
constexpr u32_t cuda_needleman_wunsch_rows_per_lane(std::size_t codes) noexcept {
    constexpr u32_t widest_k = vram_needleman_wunsch_rows_k[std::size(vram_needleman_wunsch_rows_k) - 1];
    return static_cast<u32_t>(min_of<std::size_t>(
        widest_k, max_of<std::size_t>(1, divide_round_up<std::size_t>(codes, cuda_warp_t::width_k))));
}

/** Stripes a query of @p codes takes, one at least. */
constexpr u32_t cuda_needleman_wunsch_stripes(std::size_t codes) noexcept {
    std::size_t const stripe_rows = std::size_t {cuda_warp_t::width_k} * cuda_needleman_wunsch_rows_per_lane(codes);
    return static_cast<u32_t>(max_of<std::size_t>(1, divide_round_up(codes, stripe_rows)));
}

/** A query a Needleman-Wunsch walk scores rows against, laid out in its span. */
struct cuda_needleman_wunsch_query_t {

    /** The span: a profile stripe per stripe of the query, the cost of every class against each
     *  lane's codes, then, past one stripe, a boundary row per warp of a block. */
    byte_t* span;

    /** Codes of the query. */
    u32_t codes;

    /** Query codes each lane holds, at most its rung's. */
    u8_t rows_per_lane;

    /** Bits a packed code takes. */
    u8_t code_bits;

    /** Cost of one code facing a gap. */
    u8_t gap;
};

/** Lays the packed @p sequence out as a query in @p span under the costs at @p parameters, every
 *  lane of the warp writing its own profile words. */
__device__ __forceinline__ cuda_needleman_wunsch_query_t cuda_needleman_wunsch_prepare_ampere(
    byte_t const* parameters, basic::vram_entry_t sequence, byte_t* span, u32_t lane) noexcept {
    needleman_wunsch_costs_t const& costs = *reinterpret_cast<needleman_wunsch_costs_t const*>(parameters);
    cuda_needleman_wunsch_query_t query;
    query.span = span, query.code_bits = costs.code_bits, query.gap = costs.gap;
    query.codes = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes, costs.code_bits));
    query.rows_per_lane = static_cast<u8_t>(cuda_needleman_wunsch_rows_per_lane(query.codes));
    u32_t const stripes = cuda_needleman_wunsch_stripes(query.codes);
    u64_t* const profile = reinterpret_cast<u64_t*>(span);
    for (u32_t stripe = 0; stripe != stripes; ++stripe) {
        u32_t const lane_first = (stripe * cuda_warp_t::width_k + lane) * query.rows_per_lane;
        for (u32_t code = 0; code != needleman_wunsch_classes_k; ++code) {
            u64_t costs_word = 0;
            for (u32_t row_in_lane = 0; row_in_lane != query.rows_per_lane; ++row_in_lane) {
                u32_t const index = lane_first + row_in_lane;
                if (index < query.codes)
                    costs_word |=
                        u64_t {costs.substitutions[packed_code_at(sequence.data, index, costs.code_bits)][code]}
                        << (8 * row_in_lane);
            }
            profile[(stripe * needleman_wunsch_classes_k + code) * cuda_warp_t::width_k + lane] = costs_word;
        }
    }
    cuda_warp_t::sync();
    return query;
}

/**
 *  @brief The Needleman-Wunsch cost from @p query to @p sequence, a warp's anti-diagonal wavefront
 *      over stripes of query codes, @p rows_ codes at most to a lane: exact within @p radius, and
 *      past it a stripe's bottom row minimum above it.
 *
 *  Lane @c l holds a stripe's rows from its @c l -th share down and enters column @c j at step
 *  @c j+l, taking the cell above and the next column's code from lane @c l-1 in one shuffled word,
 *  the first lane from a word the warp loads per 32 columns. Each stripe hands its bottom row to
 *  the next through the warp's boundary row, which the last lane writes 31 columns behind the
 *  warp's reads. Out of line, so the body compiles once per rung and unit.
 */
template <u32_t rows_>
__device__ __noinline__ distance_t cuda_needleman_wunsch_score_ampere(cuda_needleman_wunsch_query_t query,
                                                                      basic::vram_entry_t sequence, distance_t radius,
                                                                      u32_t lane) noexcept {
    using warp_t = cuda_warp_t;
    constexpr u32_t lanes_k = warp_t::width_k;
    u32_t const gap = query.gap, rows = query.codes, rows_per_lane = query.rows_per_lane;
    u32_t const columns = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes, query.code_bits));
    if (!rows || !columns) return static_cast<distance_t>((rows + columns) * gap);
    u32_t const stripe_rows = lanes_k * rows_per_lane, stripes = cuda_needleman_wunsch_stripes(rows);
    u64_t const* const profile = reinterpret_cast<u64_t const*>(query.span);
    u32_t* const boundary = reinterpret_cast<u32_t*>(query.span + stripes * cuda_needleman_wunsch_stripe_bytes_k) +
                            (threadIdx.x / lanes_k) * (cuda_needleman_wunsch_widest_codes_k + 1);
    u32_t result = 0;
    for (u32_t stripe = 0; stripe != stripes; ++stripe) {
        u32_t const first_row = stripe * stripe_rows, lane_first = first_row + lane * rows_per_lane;
        u32_t const active = min_of(lanes_k, divide_round_up(rows - first_row, rows_per_lane));
        u32_t const last_lane = (rows - 1 - first_row) / rows_per_lane,
                    last_cell = (rows - 1 - first_row) % rows_per_lane;
        bool const closing = stripe + 1 == stripes;
        u32_t cells[rows_];
#pragma unroll
        for (u32_t cell = 0; cell != rows_; ++cell) cells[cell] = (lane_first + cell + 1) * gap;
        u64_t const* const stripe_profile = profile + stripe * needleman_wunsch_classes_k * lanes_k + lane;
        // The first lane's top cell of a column and the next column's code, a column to a lane.
        auto const load_columns = [&](u32_t first_column) noexcept {
            u32_t const column = first_column + lane;
            u32_t const top = column > columns ? 0u : stripe ? boundary[column] : column * gap;
            u32_t const next_code = column < columns ? packed_code_at(sequence.data, column, query.code_bits) : 0u;
            return top << 5 | next_code;
        };
        // A cell rides one word with the code its receiver scores a step later, the cell in the
        // high bits, as far as 2^27 reaches; so each step loads the next step's costs before its
        // own cells.
        u32_t const first_code = packed_code_at(sequence.data, 0, query.code_bits);
        u32_t handed = lane == 0 ? first_code : 0u, loaded = load_columns(1);
        u64_t costs_word = stripe_profile[first_code * lanes_k];
        u32_t diagonal_above = lane_first * gap, bottom_minimum = (stripe + 1) * stripe_rows * gap;
        u32_t const steps = columns + active - 1;
        for (u32_t step = 0; step != steps; ++step) {
            if (step % lanes_k == 0 && step) loaded = load_columns(step + 1);
            u32_t const from_loaded = warp_t::shuffle(loaded, step % lanes_k);
            u32_t const from_above = warp_t::shuffle_up(handed, 1);
            u32_t const incoming = lane == 0 ? from_loaded : from_above;
            u32_t const next_code = incoming & 31u, top = incoming >> 5;
            u64_t const next_costs_word = stripe_profile[next_code * lanes_k];
            u32_t const column = step - lane + 1;
            u32_t above = top;
            if (lane < active && step >= lane && column <= columns) {
                u32_t diagonal = diagonal_above;
#pragma unroll
                for (u32_t cell = 0; cell != rows_; ++cell)
                    if (cell < rows_per_lane) {
                        u32_t const substitution = static_cast<u32_t>(costs_word >> (8 * cell)) & 0xFFu;
                        u32_t const opened = min_of(diagonal + substitution, cells[cell] + gap);
                        u32_t const scored = min_of(opened, above + gap);
                        diagonal = cells[cell], cells[cell] = scored, above = scored;
                    }
                diagonal_above = top;
                if (!closing && lane == lanes_k - 1)
                    boundary[column] = above, bottom_minimum = min_of(bottom_minimum, above);
                if (closing && lane == last_lane && column == columns)
#pragma unroll
                    for (u32_t cell = 0; cell != rows_; ++cell)
                        if (cell == last_cell) result = cells[cell];
            }
            handed = above << 5 | next_code, costs_word = next_costs_word;
        }
        warp_t::sync();
        if (closing) return static_cast<distance_t>(warp_t::shuffle(result, last_lane));
        // Every alignment crosses a stripe's bottom row, so none ends below its minimum.
        u32_t const crossing = warp_t::shuffle(bottom_minimum, lanes_k - 1);
        if (static_cast<distance_t>(crossing) > radius) return static_cast<distance_t>(crossing);
    }
    return static_cast<distance_t>(result);
}

/** Needleman-Wunsch over packed codes on the Ampere floor: a warp to a sequence, an anti-diagonal
 *  wavefront down stripes of the query, its costs a profile the walk lays out in its span from the
 *  folded table the face uploads as the parameters. */
struct packed_ampere_needleman_wunsch_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cuda_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;

    using warp_t = cuda_warp_t;
    using query_t = cuda_needleman_wunsch_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** Every rung scores a sequence with the whole warp. */
    static constexpr u32_t lanes_per_row(u32_t) noexcept { return warp_t::width_k; }

    /** A rung serves queries of up to @p codes whose stripe it holds whole, the widest one any
     *  number of its stripes. */
    static constexpr bool serves(u32_t rows, std::size_t codes) noexcept {
        constexpr u32_t widest_k = vram_needleman_wunsch_rows_k[std::size(vram_needleman_wunsch_rows_k) - 1];
        return rows == widest_k ? codes <= cuda_needleman_wunsch_widest_codes_k
                                : codes <= std::size_t {rows} * warp_t::width_k;
    }

    /** Bytes the packed @p bytes long sequence at @p data takes once prepared: its profile stripes,
     *  and a boundary row per warp of a block when it takes more than one. */
    static std::size_t span_bytes(u32_t code_bits, byte_t const* data, std::size_t bytes) noexcept {
        u32_t const stripes = cuda_needleman_wunsch_stripes(packed_codes_count(data, bytes, code_bits));
        std::size_t const boundaries =
            stripes > 1 ? vram_warps_per_block_k * (cuda_needleman_wunsch_widest_codes_k + 1) * sizeof(u32_t) : 0;
        return stripes * cuda_needleman_wunsch_stripe_bytes_k + boundaries;
    }

    static __device__ __forceinline__ query_t prepare(byte_t const* parameters, basic::vram_entry_t sequence,
                                                      byte_t* span, u32_t lane) noexcept {
        return cuda_needleman_wunsch_prepare_ampere(parameters, sequence, span, lane);
    }

    /** Every code one side holds past the other's length faces a gap. */
    static __device__ __forceinline__ distance_t lower_bound(nk_kernel_kind_t, query_t const& query,
                                                             basic::vram_entry_t sequence) noexcept {
        u32_t const columns = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes, query.code_bits));
        u32_t const difference = columns > query.codes ? columns - query.codes : query.codes - columns;
        return static_cast<distance_t>(difference * query.gap);
    }
    template <u32_t rows_>
    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query,
                                                       basic::vram_entry_t sequence, distance_t radius,
                                                       u32_t lane) noexcept {
        return cuda_needleman_wunsch_score_ampere<rows_>(query, sequence, radius, lane);
    }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cuda_k) != 0)
struct needleman_wunsch_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, packed_ampere_needleman_wunsch_kit_t>::type;
    static constexpr sz_capability_t picked_k = caps_ & kit_t::capability_k;
};

/** The kit scoring the sequence kind @p metric_ under @p caps_: the overlap or Needleman-Wunsch
 *  ladder's, or the Levenshtein ladder's over the kind's symbols. */
template <usearch_sequence_metric_t metric_, sz_capability_t caps_>
using vram_sequence_kit = std::conditional_t<
    metric_ == usearch_sequence_metric_needleman_wunsch_k, typename needleman_wunsch_ladder<caps_>::kit_t,
    std::conditional_t<
        metric_ == usearch_sequence_metric_overlap_k, typename overlap_ladder<caps_>::kit_t,
        typename levenshtein_ladder<metric_ == usearch_sequence_metric_levenshtein_utf8_k ? sz_levenshtein_runes_k
                                                                                          : sz_levenshtein_bytes_k,
                                    caps_>::kit_t>>;

#pragma endregion Kits

#pragma region Face

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
status_t vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>::window_widths(
    std::span<std::size_t const> widths) noexcept {
    if ((metric_ == usearch_sequence_metric_overlap_k) == widths.empty()) return status_t::config_mismatch_k;
    if (widths.empty()) return status_t::success_k;
    // The count leads, as the kit reads the parameters.
    if (status_t const reserved = host_parameters_.reserve_exactly(widths.size() + 1); failed(reserved))
        return reserved;
    host_parameters_.clear();
    host_parameters_.push_back_reserved(static_cast<u32_t>(widths.size()));
    for (std::size_t const width : widths) {
        if (!width || width > std::numeric_limits<u32_t>::max()) return status_t::config_mismatch_k;
        host_parameters_.push_back_reserved(static_cast<u32_t>(width));
    }
    return status_t::success_k;
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
status_t vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>::costs(
    needleman_wunsch_costs_t const& costs) noexcept {
    if ((metric_ == usearch_sequence_metric_needleman_wunsch_k) != (costs.code_bits != 0))
        return status_t::config_mismatch_k;
    if (!costs.code_bits) return status_t::success_k;
    if (status_t const sized = host_parameters_.resize(divide_round_up(sizeof(costs), sizeof(u32_t))); failed(sized))
        return sized;
    std::memcpy(host_parameters_.data(), &costs, sizeof(costs));
    return status_t::success_k;
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
std::size_t vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>::rung_serving_(
    std::size_t longest) const noexcept {
    using kit_t = vram_sequence_kit<metric_, capability_>;
    constexpr std::span<u32_t const> rungs = vram_sequence_rungs(metric_);
    // A Needleman-Wunsch rung serves codes, the most a sequence of these bytes packs.
    if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k) {
        if (host_parameters_.empty()) return rungs_k;
        longest = longest ? (longest - 1) * 8 / costs_().code_bits : 0;
    }
    std::size_t rung = 0;
    while (rung != rungs_k && !kit_t::serves(rungs[rung], longest)) ++rung;
    if constexpr (metric_ == usearch_sequence_metric_overlap_k) {
        // Lanes past the longest sequence's widest windows would roll none of their own.
        u32_t widest = 0;
        for (std::size_t index = 1; index < host_parameters_.size(); ++index)
            widest = max_of(widest, host_parameters_[index]);
        while (rung + 1 < rungs_k && std::size_t {rungs[rung + 1]} * widest <= longest) ++rung;
    }
    return rung;
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
u32_t vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>::lanes_per_entry_(
    std::size_t rung) noexcept {
    using kit_t = vram_sequence_kit<metric_, capability_>;
    constexpr std::span<u32_t const> rungs = vram_sequence_rungs(metric_);
    return kit_t::lanes_per_row(rungs[rung]);
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
std::size_t vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>::span_bytes_(
    byte_t const* data, std::size_t bytes) const noexcept {
    using kit_t = vram_sequence_kit<metric_, capability_>;
    if constexpr (metric_ == usearch_sequence_metric_overlap_k)
        return kit_t::span_bytes(to_span(host_parameters_).subspan(1), bytes);
    else if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k)
        return kit_t::span_bytes(costs_().code_bits, data, bytes);
    else return kit_t::span_bytes(data, bytes);
}

template <usearch_sequence_metric_t metric_, typename node_id_type_, sz_capability_t capability_,
          typename kernels_type_>
status_t vram_nvidia_kits<metric_, node_id_type_, capability_, kernels_type_>::load_kernels_(cuda_api_t const& api,
                                                                                             int device) noexcept {
    bool const parametrized = metric_ == usearch_sequence_metric_overlap_k ||
                              metric_ == usearch_sequence_metric_needleman_wunsch_k;
    if (parametrized && host_parameters_.empty()) return status_t::uninitialized_k;
    auto [module, status] = cuda_module_t::load(api, device, kernels_type_::bytes());
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
    if (host_parameters_.empty()) return status_t::success_k;
    device_parameters_ = cuda_make_unique(cuda_device_allocator<u32_t> {api}, host_parameters_.size());
    if (!device_parameters_) return status_t::bad_alloc_k;
    if (cuda_copy(api, device_parameters_.get(), host_parameters_.data(), host_parameters_.size() * sizeof(u32_t)) !=
        cuda_success_k)
        return status_t::device_error_k;
    return status_t::success_k;
}

#pragma endregion Face

#pragma region Kernels

/** Every NVIDIA sequence kernel: @p kernel_ picks the walk, @p capability_ and @p metric_ the kit
 *  through its ladder, @p rung_ its rung, Myers words or lanes per sequence, @p node_id_type_ the
 *  IDs of the records, and @p slots_ the register rung. */
template <vram_kernel_t kernel_, sz_capability_t capability_, usearch_sequence_metric_t metric_, u32_t rung_,
          typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, basic::cuda_resident_blocks(kernel_, slots_))
    cuda_kernel(basic::vram_arguments<node_id_type_> arguments) {
    using kit_t = vram_sequence_kit<metric_, capability_>;
    basic::cuda_kernel_body<kernel_, kit_t, rung_, slots_>(arguments);
}

template <sz_capability_t capability_, usearch_sequence_metric_t metric_, u32_t rung_, typename node_id_type_,
          std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<basic::cuda_rungs[rungs_].kernel, capability_, metric_, rung_, node_id_type_,
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
