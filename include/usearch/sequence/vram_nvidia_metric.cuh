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

#include <iterator>    // `std::size` - every rung's kernels
#include <limits>      // `std::numeric_limits` - the widths and rows a `u32_t` states
#include <span>        // `std::span` - the overlap kind's window widths
#include <type_traits> // `std::is_empty_v` - a kit taking no parameters
#include <utility>     // `std::make_integer_sequence` - the word counts, every rung's kernels
#include <variant>     // `std::monostate` - the Levenshtein kits' parameters

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

/** Myers words of the longest query each Levenshtein rung's kernels score, narrowest first: a warp
 *  walks each query up to sixteen words, 1 KiB, a block past them, up to 4 KiB, where a block's
 *  shared memory runs out. */
inline constexpr u32_t vram_levenshtein_words_k[] = {2, 4, 8, 16, 32, 64};

/** Bytes of the longest sequence @p words Myers words score, a bit of a 64-bit word per byte; a
 *  sequence of @c b bytes holds at most @c b runes, so it bounds rune sequences too. */
constexpr std::size_t vram_levenshtein_bytes(u32_t words) noexcept {
    return std::size_t {words} * std::numeric_limits<u64_t>::digits;
}

/** Bytes of the longest sequence the widest Levenshtein rung scores; longer ones are refused. */
inline constexpr std::size_t vram_levenshtein_widest_bytes_k = vram_levenshtein_bytes(
    vram_levenshtein_words_k[std::size(vram_levenshtein_words_k) - 1]);

/** Query codes each lane of a Needleman-Wunsch rung holds, narrowest first: a warp's stripe of
 *  rows is as tall as the lanes times these, and the widest rung repeats its stripe down a query
 *  as long as the widest Levenshtein sequence. */
inline constexpr u32_t vram_needleman_wunsch_rows_k[] = {1, 2, 3, 4, 5, 6, 7, 8};

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
__host__ __device__ std::size_t cuda_levenshtein_span_bytes_ampere(byte_t const* data, std::size_t bytes) noexcept {
    if constexpr (symbol_ == sz_levenshtein_runes_k) {
        // Distinct pages in a bitset over every page, distinct runes in an open table twice the
        // widest rung's runes wide, cleared only as wide as this sequence needs.
        constexpr std::size_t table_slots_k = 2 * vram_levenshtein_widest_bytes_k;
        u64_t pages[sz_levenshtein_utf8_pages_k / 64] = {};
        u32_t table[table_slots_k];
        // Doubled by hand: `std::bit_ceil` evaluates wrong in device code under relaxed constexpr.
        std::size_t slots = 1;
        while (slots < 2 * bytes + 2 && slots < table_slots_k) slots *= 2;
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
                (classes + 1) * sz_levenshtein_walk_stride(sz_levenshtein_query_words(runes)) * sizeof(u64_t),
            16);
    }
    u64_t present[4] = {};
    for (std::size_t position = 0; position != bytes; ++position) {
        u8_t const value = static_cast<u8_t>(data[position]);
        present[value >> 6] |= u64_t {1} << (value & 63u);
    }
    // Counted by hand: `std::popcount` evaluates to zero in device code under relaxed constexpr.
    std::size_t distinct = 0;
    for (u64_t word : present)
        for (; word; word &= word - 1) ++distinct;
    std::size_t const classes = min_of<std::size_t>(distinct + 1, sz_levenshtein_byte_classes_k);
    return round_up_to_multiple<std::size_t>(
        sz_levenshtein_byte_classes_k +
            (classes + 1) * sz_levenshtein_walk_stride(sz_levenshtein_query_words(bytes)) * sizeof(u64_t),
        16);
}

/** Raises @p load 's span word to the widest its stored sequences prepare into over @p symbol_,
 *  a lane per sequence, each warp folding its lanes' spans before one compare-exchange. */
template <sz_levenshtein_symbol_t symbol_, typename node_id_type_>
__device__ __forceinline__ void cuda_levenshtein_size_spans_ampere(
    basic::vram_load<node_id_type_> const& load) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<cuda_warp_t>();
    u32_t const threads = warp.count * cuda_warp_t::width_k;
    for (u32_t first = warp.index * cuda_warp_t::width_k; first < load.count; first += threads) { // Warp-uniform.
        u32_t const node = first + warp.lane;
        u64_t span = 0;
        if (node < load.count)
            span = cuda_levenshtein_span_bytes_ampere<symbol_>(
                load.tape + load.stored_offsets[node],
                static_cast<std::size_t>(load.stored_offsets[node + 1] - load.stored_offsets[node]));
        for (u32_t mask = cuda_warp_t::width_k / 2; mask; mask /= 2)
            span = max_of(span, cuda_warp_t::shuffle_xor(span, mask));
        if (warp.lane != 0) continue;
        u64_t widest = cuda_warp_t::load_relaxed(*load.span_bytes);
        while (widest < span && !cuda_warp_t::compare_exchange_relaxed(*load.span_bytes, widest, span))
            widest = cuda_warp_t::load_relaxed(*load.span_bytes);
    }
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

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "u8_ampere_levenshtein_kit_t";

    using warp_t = cuda_warp_t;
    using query_t = sz_levenshtein_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** A Levenshtein distance takes no parameters. */
    using parameters_t = std::monostate;

    /** Myers words of the longest query each rung's kernels score, narrowest first. */
    inline static constexpr std::span<u32_t const> rungs_k = vram_levenshtein_words_k;

    static constexpr u32_t lanes_per_row(u32_t words) noexcept { return cuda_levenshtein_lanes_per_row_ampere(words); }

    /** The first rung whose words cover a @p longest -byte sequence, or past the widest. */
    static std::size_t rung_serving(std::size_t longest, parameters_t) noexcept {
        std::size_t rung = 0;
        while (rung != std::size(rungs_k) && longest > vram_levenshtein_bytes(rungs_k[rung])) ++rung;
        return rung;
    }

    /** Bytes the @p bytes long sequence at @p data takes once prepared, and none from a length
     *  alone, since the device counts a load's spans. */
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes, parameters_t) noexcept {
        return cuda_levenshtein_span_bytes_ampere<sz_levenshtein_bytes_k>(data, bytes);
    }
    static std::size_t span_bytes(std::size_t, parameters_t) noexcept { return 0; }

    /** A load stores sequences as saved, then counts their spans on the device. */
    template <typename node_id_type_>
    static status_t queue_load_values(basic::vram_load_step<node_id_type_> const& step) noexcept {
        if (status_t const copied = step.copy_values(); failed(copied)) return copied;
        return step.launch_kits(vram_kernel_t::size_spans_k);
    }

    /** Counts a load's spans on the device, from each sequence's distinct bytes. */
    template <typename node_id_type_>
    static __device__ __forceinline__ void size_spans(basic::vram_load<node_id_type_> const& load) noexcept {
        cuda_levenshtein_size_spans_ampere<sz_levenshtein_bytes_k>(load);
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

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "utf8_ampere_levenshtein_kit_t";

    using warp_t = cuda_warp_t;
    using query_t = sz_levenshtein_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** A Levenshtein distance takes no parameters. */
    using parameters_t = std::monostate;

    /** Myers words of the longest query each rung's kernels score, narrowest first. */
    inline static constexpr std::span<u32_t const> rungs_k = vram_levenshtein_words_k;

    static constexpr u32_t lanes_per_row(u32_t words) noexcept { return cuda_levenshtein_lanes_per_row_ampere(words); }

    /** The first rung whose words cover a @p longest -byte sequence, or past the widest. */
    static std::size_t rung_serving(std::size_t longest, parameters_t) noexcept {
        std::size_t rung = 0;
        while (rung != std::size(rungs_k) && longest > vram_levenshtein_bytes(rungs_k[rung])) ++rung;
        return rung;
    }

    /** Bytes the @p bytes long sequence at @p data takes once prepared, and none from a length
     *  alone, since the device counts a load's spans. */
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes, parameters_t) noexcept {
        return cuda_levenshtein_span_bytes_ampere<sz_levenshtein_runes_k>(data, bytes);
    }
    static std::size_t span_bytes(std::size_t, parameters_t) noexcept { return 0; }

    /** A load stores sequences as saved, then counts their spans on the device. */
    template <typename node_id_type_>
    static status_t queue_load_values(basic::vram_load_step<node_id_type_> const& step) noexcept {
        if (status_t const copied = step.copy_values(); failed(copied)) return copied;
        return step.launch_kits(vram_kernel_t::size_spans_k);
    }

    /** Counts a load's spans on the device, from each sequence's pages and distinct runes. */
    template <typename node_id_type_>
    static __device__ __forceinline__ void size_spans(basic::vram_load<node_id_type_> const& load) noexcept {
        cuda_levenshtein_size_spans_ampere<sz_levenshtein_runes_k>(load);
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

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "u8_ampere_overlap_kit_t";

    using warp_t = cuda_warp_t;
    using query_t = sz_overlap_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The window widths a face hashes with, set once before the first insert. */
    using parameters_t = std::span<u32_t const>;

    /** Lanes rolling one sequence's windows, narrowest first: one lane alone, then every power of
     *  two from a block's warps up to the kit's warp. */
    inline static constexpr u32_t rungs_k[] = {1, 4, 8, 16, 32};

    /** A rung is its lanes per sequence. */
    static constexpr u32_t lanes_per_row(u32_t lanes) noexcept { return lanes; }
    static_assert(rungs_k[std::size(rungs_k) - 1] <= warp_t::width_k);

    /** The widest rung whose lanes still roll a widest window of a @p longest -byte sequence, or
     *  past every rung for a length no byte count states. */
    static std::size_t rung_serving(std::size_t longest, parameters_t widths) noexcept {
        if (longest > (std::numeric_limits<u32_t>::max)()) return std::size(rungs_k);
        u32_t widest = 0;
        for (u32_t const width : widths) widest = max_of(widest, width);
        std::size_t rung = 0;
        while (rung + 1 < std::size(rungs_k) && std::size_t {rungs_k[rung + 1]} * widest <= longest) ++rung;
        return rung;
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone. */
    static std::size_t span_bytes(byte_t const*, std::size_t bytes, parameters_t widths) noexcept {
        return span_bytes(bytes, widths);
    }
    static std::size_t span_bytes(std::size_t bytes, parameters_t widths) noexcept {
        return round_up_to_multiple<std::size_t>(sz_overlap_query_span_bytes(bytes, widths.data(), widths.size()), 16);
    }

    /** A load stores sequences as saved; spans size from lengths alone. */
    template <typename node_id_type_>
    static status_t queue_load_values(basic::vram_load_step<node_id_type_> const& step) noexcept {
        return step.copy_values();
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

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "packed_ampere_needleman_wunsch_kit_t";

    using warp_t = cuda_warp_t;
    using query_t = cuda_needleman_wunsch_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The folded costs a face scores with, set once before the first insert. */
    using parameters_t = needleman_wunsch_costs_t;

    /** Query codes each lane holds, narrowest first. */
    inline static constexpr std::span<u32_t const> rungs_k = vram_needleman_wunsch_rows_k;

    /** Every rung scores a sequence with the whole warp. */
    static constexpr u32_t lanes_per_row(u32_t) noexcept { return warp_t::width_k; }

    /** The first rung holding whole the stripe of the most codes a @p longest -byte pack holds,
     *  the widest one any number of its stripes, or past every rung before any costs. */
    static std::size_t rung_serving(std::size_t longest, parameters_t const& costs) noexcept {
        if (!costs.code_bits) return std::size(rungs_k);
        // A pack of these bytes holds at most this many codes past its one-byte header.
        std::size_t const codes = longest ? (longest - 1) * 8 / costs.code_bits : 0;
        std::size_t const widest = std::size(rungs_k) - 1;
        std::size_t rung = 0;
        while (rung != std::size(rungs_k) && codes > (rung == widest ? cuda_needleman_wunsch_widest_codes_k
                                                                     : std::size_t {rungs_k[rung]} * warp_t::width_k))
            ++rung;
        return rung;
    }

    /** Bytes a query of @p codes codes takes once prepared: its profile stripes, and a boundary
     *  row per warp of a block when it takes more than one. */
    static std::size_t span_bytes_of_codes(std::size_t codes) noexcept {
        u32_t const stripes = cuda_needleman_wunsch_stripes(codes);
        std::size_t const boundaries =
            stripes > 1 ? vram_warps_per_block_k * (cuda_needleman_wunsch_widest_codes_k + 1) * sizeof(u32_t) : 0;
        return stripes * cuda_needleman_wunsch_stripe_bytes_k + boundaries;
    }

    /** Bytes the packed @p bytes long sequence at @p data takes once prepared, and the bound for
     *  any pack of up to @p longest bytes. */
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes, parameters_t const& costs) noexcept {
        return span_bytes_of_codes(packed_codes_count(data, bytes, costs.code_bits));
    }
    static std::size_t span_bytes(std::size_t longest, parameters_t const& costs) noexcept {
        return span_bytes_of_codes(longest ? (longest - 1) * 8 / costs.code_bits : 0);
    }

    /** A load stores sequences as saved; spans size from lengths alone. */
    template <typename node_id_type_>
    static status_t queue_load_values(basic::vram_load_step<node_id_type_> const& step) noexcept {
        return step.copy_values();
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

/** Spells a sequence kit into a kernel's symbol by its name, inside the kernel's own scope, @c S1_,
 *  as @c vram_kernel_symbol documents. */
template <typename kit_type_>
    requires requires { kit_type_::name_k; }
void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<kit_type_>) noexcept {
    symbol.append("NS1_");
    symbol.append_name(kit_type_::name_k);
    symbol.append("E");
}

#pragma endregion Kits

#pragma region Face

/** Lays a kit's @p parameters out in @p laid as its kernels read them on the device, and keeps
 *  them in @p kept as its host members read them: nothing for none, the window widths after their
 *  count, or a table's bytes. */
inline status_t vram_lay_out_parameters(std::monostate, growing_buffer<u32_t, std::allocator<byte_t>>&,
                                        std::monostate&) noexcept {
    return status_t::success_k;
}
inline status_t vram_lay_out_parameters(std::span<u32_t const> widths,
                                        growing_buffer<u32_t, std::allocator<byte_t>>& laid,
                                        std::span<u32_t const>& kept) noexcept {
    if (status_t const sized = laid.resize(widths.size() + 1); failed(sized)) return sized;
    laid[0] = static_cast<u32_t>(widths.size());
    for (std::size_t index = 0; index != widths.size(); ++index) laid[index + 1] = widths[index];
    kept = to_span(laid).subspan(1);
    return status_t::success_k;
}
template <typename table_type_>
status_t vram_lay_out_parameters(table_type_ const& table, growing_buffer<u32_t, std::allocator<byte_t>>& laid,
                                 table_type_& kept) noexcept {
    if (status_t const sized = laid.resize(divide_round_up(sizeof(table_type_), sizeof(u32_t))); failed(sized))
        return sized;
    std::memcpy(laid.data(), &table, sizeof(table_type_));
    kept = table;
    return status_t::success_k;
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::parameters(parameters_t parameters) noexcept {
    return vram_lay_out_parameters(parameters, host_parameters_, kept_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::rung_serving_(
    std::size_t longest) const noexcept {
    return kit_type_::rung_serving(longest, kept_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
u32_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::lanes_per_entry_(std::size_t rung) noexcept {
    return kit_type_::lanes_per_row(kit_type_::rungs_k[rung]);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::span_bytes_(byte_t const* data,
                                                                                   std::size_t bytes) const noexcept {
    return kit_type_::span_bytes(data, bytes, kept_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::span_bytes_(std::size_t longest) const noexcept {
    return kit_type_::span_bytes(longest, kept_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::queue_load_values_(
    basic::vram_load_step<node_id_type_> const& step) const noexcept {
    return kit_type_::queue_load_values(step);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
void vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::spell_kits_kernel_(
    vram_kernel_symbol_t& symbol, std::size_t rung, basic::cuda_rung_t instantiated) noexcept {
    vram_kernel_symbol(symbol, "sequence", "cuda_kernel", {"vram_arguments", 3, itanium_scope_t::basic_k},
                       instantiated.kernel, itanium_type<kit_type_> {}, kit_type_::rungs_k[rung],
                       itanium_type<node_id_type_> {}, instantiated.slots);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_nvidia_kits<kit_type_, node_id_type_, kernels_type_>::upload_parameters_(cuda_api_t const& api) noexcept {
    if (host_parameters_.empty())
        return std::is_empty_v<parameters_t> ? status_t::success_k : status_t::uninitialized_k;
    device_parameters_ = cuda_make_unique(cuda_device_allocator<u32_t> {api}, host_parameters_.size());
    if (!device_parameters_) return status_t::bad_alloc_k;
    if (cuda_copy(api, device_parameters_.get(), host_parameters_.data(), host_parameters_.size() * sizeof(u32_t)) !=
        cuda_success_k)
        return status_t::device_error_k;
    return status_t::success_k;
}

#pragma endregion Face

#pragma region Kernels

/** Every NVIDIA sequence kernel: @p kernel_ picks the walk, @p kit_type_ scores at its @p rung_ of
 *  Myers words or lanes per sequence, @p slots_ is the register rung, and @p node_id_type_ is the
 *  type of the record IDs it walks. */
template <vram_kernel_t kernel_, typename kit_type_, u32_t rung_, typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, basic::cuda_resident_blocks(kernel_, slots_))
    cuda_kernel(basic::vram_arguments<node_id_type_> arguments) {
    basic::cuda_kernel_body<kernel_, kit_type_, rung_, slots_>(arguments);
}

template <typename kit_type_, u32_t rung_, typename node_id_type_, std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<basic::cuda_rungs[rungs_].kernel, kit_type_, rung_, node_id_type_,
                                    basic::cuda_rungs[rungs_].slots>),
     ...);
}

template <typename kit_type_, typename node_id_type_, std::size_t... words_>
__device__ void cuda_instantiate_words(std::index_sequence<words_...>) noexcept {
    (cuda_instantiate_rungs<kit_type_, kit_type_::rungs_k[words_], node_id_type_>(
         std::make_index_sequence<std::size(basic::cuda_rungs)> {}),
     ...);
}

/** Instantiates every kernel of every rung of @p kit_type_ with @p node_id_type_ IDs; the unit
 *  compiling the fatbin names it once per kit and node-ID type. Taking the kernels' addresses in
 *  device code makes NVCC emit them, while a host-only pass emits no launch stub. */
template <typename kit_type_, typename node_id_type_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_words<kit_type_, node_id_type_>(std::make_index_sequence<std::size(kit_type_::rungs_k)> {});
    constexpr u32_t first_rung = kit_type_::rungs_k[0];
    static_cast<void>(&cuda_kernel<vram_kernel_t::load_values_k, kit_type_, first_rung, node_id_type_, 0>);
    static_cast<void>(&cuda_kernel<vram_kernel_t::size_spans_k, kit_type_, first_rung, node_id_type_, 0>);
}

#pragma endregion Kernels

} // namespace unum::usearch::sequence
