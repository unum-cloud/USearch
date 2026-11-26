/**
 *  @file include/usearch/sequence/vram_amd_metric.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Device kits for the sequence VRAM indexes on AMD GPUs, the GPU twin of @c ram_metric.hpp,
 *      their ladders, the kernels, and the members of @c vram_amd_index.hpp reading them.
 *
 *  A kit binds one sequence distance to one AMD generation, named like the dense AMD kits; the
 *  generation slot names the oldest tier whose instructions it needs. @c cdna3 kits run on the
 *  64-lane wavefronts of gfx942 and gfx950, @c cdna5 kits on the 32-lane wavefronts of gfx1250 and
 *  gfx1251. The window hashes run on either width, so the distances are the same integers the CPU
 *  and NVIDIA kits score. Each pair of kits shares one body over its wavefront, through the
 *  portable ROCm steps for every cross-lane move.
 *
 *  StringZilla prepares a query on 32 lanes, so on a 64-lane wavefront the upper half skips the
 *  preparation and takes the lower half's query, the span being the wavefront's alone; every scorer
 *  runs within an aligned group of at most 32 lanes, so it runs as written on either width. The
 *  alignment wavefront spans the whole wavefront, its stripes 64 or 32 lanes tall.
 *
 *  The containment kit scores on its lanes per sequence. The banded overlap kit scores a sequence
 *  with one lane, which votes a shift through the query's window table and aligns the band around
 *  it, the query beside its complement image, and the lower of the two distances stands. The
 *  alignment kit scores a sequence with a wavefront, an anti-diagonal sweep over stripes of query
 *  codes, each lane holding a rung's worth of rows in registers.
 *
 *  A sketching kit also carries the sketch its builders score: a wavefront writes one stored
 *  sequence's sketch, each of its first 32 lanes its share of the columns or the slots, the slots
 *  through StringZilla's lane steps, and one lane scores a pair of sketches. Alignment sketches
 *  smoothly and containment by hashes; banded overlap has no sketch and builds by its canonical
 *  score. Only the modality's AMD units compile this file, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <cstring> // `std::memcpy` - a kit's parameters on the host

#include <iterator>    // `std::size` - every rung's kernels
#include <limits>      // `std::numeric_limits` - the widths and rows a `u32_t` states
#include <span>        // `std::span` - a kit's rungs
#include <type_traits> // `std::is_empty_v` - a kit taking no parameters
#include <utility>     // `std::make_integer_sequence` - every rung's kernels
#include <variant>     // `std::monostate` - a kit reading no sketch tables

#include <stringzilla/ngrams/simt.cuh> // `sz_ngrams_query_prepare_simt`, `sz_ngrams_sketch_simt` & kin

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"     // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_amd_device.hip" // `hip_warp`, `cdna3_scan_t`, `cdna5_scan_t`
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_amd_index.hip" // `basic::hip_kernel_body`
#include "usearch/sequence/ram_metric.hpp"  // the ladders, `spectrum_view_of`, `containment_distance`
#include "usearch/sequence/vram_amd_index.hpp"

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

#pragma region Kits

/** Query codes each lane of an alignment rung holds, narrowest first: a wavefront's stripe
 *  of rows is as tall as the lanes times these, and the widest rung repeats its stripe down
 *  the longest query. */
inline constexpr u32_t vram_alignment_rows_k[] = {1, 2, 3, 4, 5, 6, 7, 8};

/** Columns of a smooth sketch one lane sums in a pass, its sums, angles and steps in registers. */
inline constexpr u32_t hip_spectrum_lane_columns_k = 16;

/** Adds the window whose class codes @p classes hold to a lane's @p sums at its @p owned columns
 *  from @p first, two phases a word, and steps their @p angles by @p deltas. */
__device__ __forceinline__ void hip_sketch_window(spectrum_view_t const& tables,
                                                  u8_t const (&classes)[spectrum_width_max_k], u32_t first, u32_t owned,
                                                  u32_t (&sums)[hip_spectrum_lane_columns_k],
                                                  u32_t (&angles)[hip_spectrum_lane_columns_k],
                                                  u32_t const (&deltas)[hip_spectrum_lane_columns_k]) noexcept {
    constexpr u32_t share_k = hip_spectrum_lane_columns_k;
    spectrum_head_t const& head = *tables.head;
#pragma unroll
    for (u32_t column = 0; column != share_k; column += 2) {
        if (column >= owned) break;
        // Two phases ride one word, and the masked add keeps a carry out of the upper one.
        u32_t phases = (angles[column] >> 16) | (angles[column + 1] & 0xFFFF0000u);
        for (u32_t place = 0; place != head.width; ++place) {
            u32_t const* const row = reinterpret_cast<u32_t const*>(
                tables.phases + std::size_t {place * head.classes + classes[place]} * head.columns);
            u32_t const added = row[(first + column) / 2];
            phases = ((phases & 0x7FFF7FFFu) + (added & 0x7FFF7FFFu)) ^ ((phases ^ added) & 0x80008000u);
        }
        sums[column] += static_cast<u32_t>(head.cosines[(phases & 0xFFFFu) >> 6]);
        sums[column + 1] += static_cast<u32_t>(head.cosines[phases >> 22]);
        angles[column] += deltas[column], angles[column + 1] += deltas[column + 1];
    }
}

/** Writes at @p sketch lane @p lane 's shares of the @p columns -byte smooth sketch of the packed
 *  class codes of @p sequence, every window of the codes in the order @ref sketch_packed_codes sums
 *  them, as @ref spectrum_step_serial sums every column; every lane of a wavefront of @p warp_type_
 *  calls it, each summing its own consecutive columns of a pass, two phases a word, so no lane
 *  waits on another, and a sketch wider than a wavefront's pass takes several. */
template <typename warp_type_>
__device__ __forceinline__ void hip_sketch_packed(spectrum_view_t const& tables, basic::vram_entry_t sequence,
                                                  u32_t lane, byte_t* sketch, u32_t columns) noexcept {
    constexpr u32_t share_k = hip_spectrum_lane_columns_k;
    spectrum_head_t const& head = *tables.head;
    u32_t const width = head.width;
    u32_t const codes = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes));
    u32_t const windows = codes >= width ? codes - width + 1 : 0;
    // A step under half a turn times any reference stays under 2⁶³, so the product never wraps.
    u64_t const span = u64_t {head.reference} << 16;
    u32_t const rounding = (1u << head.shift) >> 1;
    for (u32_t pass = 0; pass < columns; pass += warp_type_::width_k * share_k) {
        // Columns come in fours, so a lane's share is even, and a lane past the last owns none.
        u32_t const first = pass + lane * share_k;
        u32_t const owned = first < columns ? min_of(columns - first, share_k) : 0;
        u32_t sums[share_k] = {}, angles[share_k] = {}, deltas[share_k] = {};
#pragma unroll
        for (u32_t column = 0; column != share_k; ++column)
            if (column < owned && windows)
                deltas[column] = static_cast<u32_t>(tables.steps[first + column] * span / windows);
        u8_t classes[spectrum_width_max_k] = {};
        for (u32_t index = 0; index != codes; ++index) {
            for (u32_t place = 0; place + 1 < width; ++place) classes[place] = classes[place + 1];
            classes[width - 1] = static_cast<u8_t>(packed_code_at(sequence.data, index));
            if (index + 1 < width || !owned) continue;
            hip_sketch_window(tables, classes, first, owned, sums, angles, deltas);
        }
        i8_t* const shares = reinterpret_cast<i8_t*>(sketch) + first;
#pragma unroll
        for (u32_t column = 0; column != share_k; ++column) {
            if (column >= owned) break;
            i32_t const scaled = static_cast<i32_t>(sums[column] + rounding) >> head.shift;
            shares[column] = static_cast<i8_t>(scaled < -128 ? -128 : scaled > 127 ? 127 : scaled);
        }
    }
}

/** The squared Euclidean distance between the @p bytes -byte smooth sketches at @p first and
 *  @p second, both four-byte aligned: the calling lane alone reads each a word at a time and sums
 *  in 32 bits. */
__device__ __forceinline__ distance_t hip_smooth_distance(byte_t const* first, byte_t const* second,
                                                          u32_t bytes) noexcept {
    u32_t const* const first_words = reinterpret_cast<u32_t const*>(first);
    u32_t const* const second_words = reinterpret_cast<u32_t const*>(second);
    u32_t sum = 0;
    for (u32_t word = 0; word != bytes / sizeof(u32_t); ++word) {
        u32_t const first_word = first_words[word], second_word = second_words[word];
#pragma unroll
        for (u32_t byte = 0; byte != sizeof(u32_t); ++byte) {
            i32_t const difference = static_cast<i8_t>(first_word >> (8 * byte)) -
                                     static_cast<i8_t>(second_word >> (8 * byte));
            sum += static_cast<u32_t>(difference * difference);
        }
    }
    return static_cast<distance_t>(sum);
}

/** The slots differing between the @p bytes -byte hash sketches at @p first and @p second, both
 *  four-byte aligned: the calling lane alone reads each a word at a time. */
__device__ __forceinline__ distance_t hip_hash_distance(byte_t const* first, byte_t const* second,
                                                        u32_t bytes) noexcept {
    u32_t const* const first_words = reinterpret_cast<u32_t const*>(first);
    u32_t const* const second_words = reinterpret_cast<u32_t const*>(second);
    u32_t unequal = 0;
    for (u32_t word = 0; word != bytes / sizeof(u32_t); ++word)
        unequal += sz_ngrams_sketch_word_distance(first_words[word], second_words[word]);
    return static_cast<distance_t>(unequal);
}

/** Writes at @p image the complement image of the @p sequence stored a class byte a
 *  symbol under @p costs, its symbols reversed and each complemented, with the wavefront
 *  of @p warp_type_, synchronized. */
template <typename warp_type_>
__device__ __forceinline__ void hip_complement_image(alignment_costs_t const& costs, basic::vram_entry_t sequence,
                                                     byte_t* image, u32_t lane) noexcept {
    // The span may still serve the previous query's readers until every lane gets here.
    warp_type_::sync();
    for (u32_t symbol = lane; symbol < sequence.bytes; symbol += warp_type_::width_k)
        image[symbol] = static_cast<byte_t>(
            costs.complemented[static_cast<u8_t>(sequence.data[sequence.bytes - 1 - symbol])]);
    warp_type_::sync();
}

/** Bytes the window table of a @p bytes long sequence takes at the widths of @p windows, a
 *  multiple of 16. */
constexpr std::size_t hip_window_table_bytes(std::size_t bytes, window_parameters_t const& windows) noexcept {
    return round_up_to_multiple<std::size_t>(sz_ngrams_query_span_bytes(bytes, windows.widths, windows.widths_count),
                                             16);
}

/** Files the distinct windows of @p sequence, at the widths of @p windows, into a table in @p span
 *  with the wavefront of @p warp_type_, every lane receiving the table's header. StringZilla files
 *  with 32 lanes, so the upper half of a 64-lane wavefront skips the filing and takes the header
 *  from lane zero. */
template <typename warp_type_>
__device__ __forceinline__ sz_ngrams_query_t hip_window_table(window_parameters_t const& windows,
                                                              basic::vram_entry_t sequence, byte_t* span,
                                                              u32_t lane) noexcept {
    sz_ngrams_query_t table;
    if (warp_type_::width_k <= 32 || lane < 32) {
        u32_t distinct = sz_ngrams_query_prepare_simt(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes,
                                                      windows.widths, windows.widths_count,
                                                      reinterpret_cast<sz_ptr_t>(span), &table, lane);
        for (u32_t offset = 16; offset != 0; offset >>= 1) distinct += warp_type_::shuffle_xor(distinct, offset);
        table.distinct = distinct;
    }
    if constexpr (warp_type_::width_k > 32) {
        static_assert(sizeof(sz_ngrams_query_t) % sizeof(u32_t) == 0, "The header crosses lanes a word at a time");
        u32_t words[sizeof(sz_ngrams_query_t) / sizeof(u32_t)];
        warp_type_::copy_bytes(words, &table, sizeof(sz_ngrams_query_t));
        for (u32_t& word : words) word = warp_type_::shuffle(word, 0);
        warp_type_::copy_bytes(&table, words, sizeof(sz_ngrams_query_t));
        warp_type_::sync();
    }
    return table;
}

/** A prepared containment query: the window table of the sequence and the hit words its lane
 *  groups count in, @c (windows+31)/32 words to a group. */
struct hip_containment_query_t {
    sz_ngrams_query_t table;
    u32_t* hit_words;
};

/**
 *  @brief The containment distance from @p query to @p sequence, the aligned group of @p lanes
 *      lanes of @p warp_type_ holding @p lane taking part.
 *
 *  Out of line, so the body compiles once per unit.
 */
template <typename warp_type_>
__device__ __noinline__ u64_t hip_containment_score(hip_containment_query_t query, basic::vram_entry_t sequence,
                                                    u32_t lane, u32_t lanes) noexcept {
    // A block schedule shares wavefront zero's span with every wavefront; groups count across them.
    u32_t const width = warp_type_::width_k;
    u32_t const group = (threadIdx.x / width * (width / lanes) + lane / lanes) % width;
    u32_t* const hit_words = query.hit_words + group * divide_round_up<std::size_t>(query.table.ngrams, 32);
    u64_t const windows = sz_ngrams_count(sequence.bytes, query.table.widths, query.table.widths_count);
    u32_t held = sz_ngrams_candidate_matches_simt(&query.table, reinterpret_cast<sz_cptr_t>(sequence.data),
                                                  sequence.bytes, hit_words, lane % lanes, lanes);
    for (u32_t offset = lanes / 2; offset != 0; offset >>= 1) held += warp_type_::shuffle_xor(held, offset);
    sz_ngrams_candidate_clear_simt(&query.table, hit_words, lane % lanes, lanes);
    // No group starts its next candidate before every lane's words are zero.
    warp_type_::sync();
    return containment_distance(query.table.distinct, held, windows);
}

/**
 *  @brief Window containment over folded UTF-8 on CDNA3 and CDNA4: the lower half of a
 *      64-lane wavefront files the query's distinct windows into its span, and a rung's lanes
 *      count each candidate's.
 *
 *  Windows hash through StringZilla, whose residues are the CPU kits'. The parameters are the
 *  window widths, which the hash sketch samples too, so builders link texts sharing the windows
 *  the score counts.
 */
struct utf8_cdna3_containment_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cdna3_k | sz_cap_cdna4_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "utf8_cdna3_containment_kit_t";

    using warp_t = hip_warp<64>;
    using scan_t = cdna3_scan_t;
    using canonical_query_t = hip_containment_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The window widths a face hashes with, set once before the first insert. */
    using canonical_parameters_t = window_parameters_t;

    /** Lanes hashing one sequence's windows, narrowest first: one lane alone, then every power of
     *  two from a block's wavefronts up to the 32 lanes a StringZilla vote spans. */
    inline static constexpr u32_t canonical_rungs_k[] = {1, 4, 8, 16, 32};

    /** A rung is its lanes per sequence. */
    static constexpr u32_t canonical_lanes_per_entry(u32_t lanes) noexcept { return lanes; }
    static_assert(canonical_rungs_k[std::size(canonical_rungs_k) - 1] <= warp_t::width_k);

    /** The lane groups of the narrowest rung, each with its own hit words. */
    inline static constexpr u32_t canonical_groups_k = warp_t::width_k / canonical_rungs_k[0];

    /** The widest rung whose lanes each take two aligned words of a @p longest -byte sequence, or
     *  past every rung for one longer than a window table places its windows in. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const&) noexcept {
        if (longest > sz_ngrams_query_bytes_max_k) return std::size(canonical_rungs_k);
        std::size_t rung = 0;
        while (rung + 1 < std::size(canonical_rungs_k) && std::size_t {canonical_rungs_k[rung + 1]} * 8 <= longest)
            ++rung;
        return rung;
    }

    /** Bytes every lane group's hit words take for a @p bytes long sequence, a multiple of 16. */
    static constexpr std::size_t canonical_hit_words_bytes(std::size_t bytes,
                                                           canonical_parameters_t const& windows) noexcept {
        std::size_t const counted = sz_ngrams_count(bytes, windows.widths, windows.widths_count);
        return round_up_to_multiple<std::size_t>(
            canonical_groups_k * divide_round_up<std::size_t>(counted, 32) * sizeof(u32_t), 16);
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone: its table and
     *  every lane group's hit words. */
    static std::size_t canonical_span_bytes(byte_t const*, std::size_t bytes,
                                            canonical_parameters_t const& windows) noexcept {
        return hip_window_table_bytes(bytes, windows) + canonical_hit_words_bytes(bytes, windows);
    }

    /** Prepares @p sequence with the wavefront and zeroes every group's hit words, which each walk
     *  leaves zero. */
    static __device__ __forceinline__ canonical_query_t canonical_prepare(byte_t const* parameters,
                                                                          basic::vram_entry_t sequence, byte_t* span,
                                                                          u32_t lane) noexcept {
        canonical_parameters_t const& windows = *reinterpret_cast<canonical_parameters_t const*>(parameters);
        canonical_query_t query;
        query.table = hip_window_table<warp_t>(windows, sequence, span, lane);
        query.hit_words = reinterpret_cast<u32_t*>(span + hip_window_table_bytes(sequence.bytes, windows));
        std::size_t const words_count = canonical_groups_k * divide_round_up<std::size_t>(query.table.ngrams, 32);
        for (std::size_t word = lane; word < words_count; word += warp_t::width_k) query.hit_words[word] = 0;
        warp_t::sync();
        return query;
    }

    /** No candidate holds more of the query's distinct windows than it has windows. */
    static __device__ __forceinline__ distance_t canonical_lower_bound(nk_kernel_kind_t, canonical_query_t const& query,
                                                                       basic::vram_entry_t sequence) noexcept {
        u64_t const candidate_windows = sz_ngrams_count(sequence.bytes, query.table.widths, query.table.widths_count);
        return static_cast<distance_t>(containment_distance(
            query.table.distinct, min_of<u64_t>(query.table.distinct, candidate_windows), candidate_windows));
    }
    template <u32_t lanes_>
    static __device__ __forceinline__ distance_t canonical_distance(nk_kernel_kind_t, canonical_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t,
                                                                    u32_t lane) noexcept {
        return static_cast<distance_t>(hip_containment_score<warp_t>(query, sequence, lane, lanes_));
    }

    /** A hash sketch reads no tables, and takes a byte per slot in front of every stored
     *  sequence. */
    using sketch_tables_t = std::monostate;

    /** Each warp's slot ranks, four bytes a slot, in shared memory while it sketches. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(u32_t);
    }

    /** Writes the @p sketch_bytes -slot hash sketch of the stored @p sequence, folded UTF-8, at
     *  @p sketch, four-byte aligned, over the windows the score counts, ranking slots in the
     *  wavefront's @p scratch ; every lane of one wavefront calls it for one value, and on 64 lanes
     *  the lower half alone sketches, its windows spread over 32 lanes. */
    static __device__ __forceinline__ void sketch_write(byte_t const*, basic::vram_entry_t sequence, u32_t lane,
                                                        byte_t* sketch, u32_t sketch_bytes, byte_t* scratch) noexcept {
        // Copied by constant index, as device code takes no address of a host constant.
        sz_size_t const widths[] = {containment_window_widths_k[0], containment_window_widths_k[1]};
        static_assert(std::size(widths) == std::size(containment_window_widths_k));
        if (warp_t::width_k <= 32 || lane < 32)
            sz_ngrams_sketch_simt(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes, widths, std::size(widths),
                                  reinterpret_cast<sz_u32_t*>(scratch), sketch_bytes,
                                  reinterpret_cast<sz_u8_t*>(sketch), lane);
    }

    /** Slots differing between the @p bytes -byte sketches at @p first and @p second, both
     *  four-byte aligned, counted by the calling lane alone; symmetric. */
    static __device__ __forceinline__ distance_t sketch_distance(byte_t const* first, byte_t const* second, u32_t bytes,
                                                                 u32_t) noexcept {
        return hip_hash_distance(first, second, bytes);
    }
};

/** Window containment on CDNA5, as the CDNA3 kit counts it, the whole 32-lane wavefront filing
 *  the query. */
struct utf8_cdna5_containment_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cdna5_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "utf8_cdna5_containment_kit_t";

    using warp_t = hip_warp<32>;
    using scan_t = cdna5_scan_t;
    using canonical_query_t = hip_containment_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The window widths a face hashes with, set once before the first insert. */
    using canonical_parameters_t = window_parameters_t;

    /** Lanes hashing one sequence's windows, narrowest first: one lane alone, then every power of
     *  two from a block's wavefronts up to the 32 lanes a StringZilla vote spans. */
    inline static constexpr u32_t canonical_rungs_k[] = {1, 4, 8, 16, 32};

    /** A rung is its lanes per sequence. */
    static constexpr u32_t canonical_lanes_per_entry(u32_t lanes) noexcept { return lanes; }
    static_assert(canonical_rungs_k[std::size(canonical_rungs_k) - 1] <= warp_t::width_k);

    /** The lane groups of the narrowest rung, each with its own hit words. */
    inline static constexpr u32_t canonical_groups_k = warp_t::width_k / canonical_rungs_k[0];

    /** The widest rung whose lanes each take two aligned words of a @p longest -byte sequence, or
     *  past every rung for one longer than a window table places its windows in. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const&) noexcept {
        if (longest > sz_ngrams_query_bytes_max_k) return std::size(canonical_rungs_k);
        std::size_t rung = 0;
        while (rung + 1 < std::size(canonical_rungs_k) && std::size_t {canonical_rungs_k[rung + 1]} * 8 <= longest)
            ++rung;
        return rung;
    }

    /** Bytes every lane group's hit words take for a @p bytes long sequence, a multiple of 16. */
    static constexpr std::size_t canonical_hit_words_bytes(std::size_t bytes,
                                                           canonical_parameters_t const& windows) noexcept {
        std::size_t const counted = sz_ngrams_count(bytes, windows.widths, windows.widths_count);
        return round_up_to_multiple<std::size_t>(
            canonical_groups_k * divide_round_up<std::size_t>(counted, 32) * sizeof(u32_t), 16);
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone: its table and
     *  every lane group's hit words. */
    static std::size_t canonical_span_bytes(byte_t const*, std::size_t bytes,
                                            canonical_parameters_t const& windows) noexcept {
        return hip_window_table_bytes(bytes, windows) + canonical_hit_words_bytes(bytes, windows);
    }

    /** Prepares @p sequence with the wavefront and zeroes every group's hit words, which each walk
     *  leaves zero. */
    static __device__ __forceinline__ canonical_query_t canonical_prepare(byte_t const* parameters,
                                                                          basic::vram_entry_t sequence, byte_t* span,
                                                                          u32_t lane) noexcept {
        canonical_parameters_t const& windows = *reinterpret_cast<canonical_parameters_t const*>(parameters);
        canonical_query_t query;
        query.table = hip_window_table<warp_t>(windows, sequence, span, lane);
        query.hit_words = reinterpret_cast<u32_t*>(span + hip_window_table_bytes(sequence.bytes, windows));
        std::size_t const words_count = canonical_groups_k * divide_round_up<std::size_t>(query.table.ngrams, 32);
        for (std::size_t word = lane; word < words_count; word += warp_t::width_k) query.hit_words[word] = 0;
        warp_t::sync();
        return query;
    }

    /** No candidate holds more of the query's distinct windows than it has windows. */
    static __device__ __forceinline__ distance_t canonical_lower_bound(nk_kernel_kind_t, canonical_query_t const& query,
                                                                       basic::vram_entry_t sequence) noexcept {
        u64_t const candidate_windows = sz_ngrams_count(sequence.bytes, query.table.widths, query.table.widths_count);
        return static_cast<distance_t>(containment_distance(
            query.table.distinct, min_of<u64_t>(query.table.distinct, candidate_windows), candidate_windows));
    }
    template <u32_t lanes_>
    static __device__ __forceinline__ distance_t canonical_distance(nk_kernel_kind_t, canonical_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t,
                                                                    u32_t lane) noexcept {
        return static_cast<distance_t>(hip_containment_score<warp_t>(query, sequence, lane, lanes_));
    }

    /** A hash sketch reads no tables, and takes a byte per slot in front of every stored
     *  sequence. */
    using sketch_tables_t = std::monostate;

    /** Each warp's slot ranks, four bytes a slot, in shared memory while it sketches. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(u32_t);
    }

    /** Writes the @p sketch_bytes -slot hash sketch of the stored @p sequence, folded UTF-8, at
     *  @p sketch, four-byte aligned, over the windows the score counts, ranking slots in the
     *  wavefront's @p scratch ; every lane of one wavefront calls it for one value, and on 64 lanes
     *  the lower half alone sketches, its windows spread over 32 lanes. */
    static __device__ __forceinline__ void sketch_write(byte_t const*, basic::vram_entry_t sequence, u32_t lane,
                                                        byte_t* sketch, u32_t sketch_bytes, byte_t* scratch) noexcept {
        // Copied by constant index, as device code takes no address of a host constant.
        sz_size_t const widths[] = {containment_window_widths_k[0], containment_window_widths_k[1]};
        static_assert(std::size(widths) == std::size(containment_window_widths_k));
        if (warp_t::width_k <= 32 || lane < 32)
            sz_ngrams_sketch_simt(reinterpret_cast<sz_cptr_t>(sequence.data), sequence.bytes, widths, std::size(widths),
                                  reinterpret_cast<sz_u32_t*>(scratch), sketch_bytes,
                                  reinterpret_cast<sz_u8_t*>(sketch), lane);
    }

    /** Slots differing between the @p bytes -byte sketches at @p first and @p second, both
     *  four-byte aligned, counted by the calling lane alone; symmetric. */
    static __device__ __forceinline__ distance_t sketch_distance(byte_t const* first, byte_t const* second, u32_t bytes,
                                                                 u32_t) noexcept {
        return hip_hash_distance(first, second, bytes);
    }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_rocm_k) != 0)
struct containment_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, utf8_cdna3_containment_kit_t, utf8_cdna5_containment_kit_t>::type;
};

/** Bytes every lane's tile of vote counters takes for a @p bytes long query on @p warp_type_, a
 *  multiple of 16: a counter per shift of a tile as wide as the query, so the votes are bounded
 *  by the query alone. */
template <typename warp_type_>
constexpr std::size_t hip_banded_overlap_votes_bytes(std::size_t bytes) noexcept {
    return round_up_to_multiple<std::size_t>(warp_type_::width_k * bytes * sizeof(u32_t), 16);
}

/** Cells one lane's band keeps for a @p bytes long query, whose overlaps span no more symbols. */
constexpr u32_t hip_banded_overlap_cells(std::size_t bytes) noexcept { return 2 * banded_overlap_band(bytes) + 1; }

/** A prepared banded overlap query: the window table and the symbols of the sequence, its
 *  complement image's beside them, the costs a band aligns under, and every lane's own
 *  vote counters and band cells. */
struct hip_banded_overlap_query_t {
    sz_ngrams_query_t table;
    sz_ngrams_query_t complement;
    byte_t const* symbols;
    byte_t const* complement_symbols;
    alignment_costs_t const* costs;

    /** A tile of counters per lane, as many as the query has symbols. */
    u32_t* votes;

    /** One band's cells per lane, @ref hip_banded_overlap_cells of them. */
    u32_t* cells;

    /** One, or two where the costs complement. */
    u32_t images;
};

/**
 *  @brief Counts into @p votes the shifts of one tile, @p tile of them from @p tile_first, that
 *      the windows of @p sequence vote for against the image filed in @p table.
 *
 *  Every window of the sequence the image holds once votes for the shift between the two places,
 *  counted from the furthest the sequence starts before the image. A place votes a shift no
 *  further below it than the image's last symbol, so a tile scans the places from that far before
 *  its first shift up to its last, each width rolling its own residue over them.
 */
__device__ __forceinline__ void hip_banded_overlap_tile(sz_ngrams_query_t const& table, basic::vram_entry_t sequence,
                                                        u32_t tile_first, u32_t tile, u32_t* votes) noexcept {
    u32_t const query_bytes = static_cast<u32_t>(table.length), bytes = sequence.bytes;
    u32_t const first_place = tile_first >= query_bytes ? tile_first - (query_bytes - 1) : 0;
    u32_t const end_place = min_of(bytes, tile_first + tile);
    for (u32_t counter = 0; counter != tile; ++counter) votes[counter] = 0;
    for (sz_size_t index = 0; index != table.widths_count; ++index) {
        u32_t const width = static_cast<u32_t>(table.widths[index]);
        if (!width || width > query_bytes || width > bytes) continue;
        u32_t const power = sz_ngrams_power(width);
        u32_t const last_ending = min_of(bytes, end_place + width - 1);
        u32_t residue = 0;
        for (u32_t ending = first_place + 1; ending <= last_ending; ++ending) {
            u32_t const rolled = ending - first_place;
            u8_t const outgoing = rolled > width ? static_cast<u8_t>(sequence.data[ending - width - 1]) : 0;
            residue = sz_ngrams_roll_simt(residue, static_cast<u8_t>(sequence.data[ending - 1]), outgoing, power);
            if (rolled < width) continue;
            sz_u64_t const slot = sz_ngrams_query_probe(&table, residue, index);
            // A window the image repeats names no single shift.
            if (!slot || sz_ngrams_slot_repeated(slot)) continue;
            u32_t const shift = ending - width + (query_bytes - 1) - static_cast<u32_t>(sz_ngrams_slot_first(slot));
            if (shift >= tile_first && shift - tile_first < tile) ++votes[shift - tile_first];
        }
    }
}

/**
 *  @brief The banded overlap distance from one image of a query, its @p symbols filed in @p table,
 *      to @p sequence under @p costs, scored by one lane in its own @p votes and @p cells.
 *
 *  A span is sized from its query alone, so the shifts are counted a tile at a time, a tile as
 *  wide as the image: tiles ascend, and a later shift wins only with strictly more votes, so the
 *  least of the most voted shifts wins, as one pass over every shift would pick it. The band is
 *  @ref alignment_band 's.
 */
__device__ __forceinline__ u64_t hip_banded_overlap_image(alignment_costs_t const& costs,
                                                          sz_ngrams_query_t const& table, byte_t const* symbols,
                                                          basic::vram_entry_t sequence, u32_t* votes,
                                                          u32_t* cells) noexcept {
    u32_t const query_bytes = static_cast<u32_t>(table.length), bytes = sequence.bytes;
    if (!query_bytes || !bytes) return banded_overlap_ceiling_k;
    u32_t const tile = query_bytes, shifts = query_bytes + bytes - 1;
    u32_t most_votes = 0, voted_shift = 0;
    for (u32_t tile_first = 0; tile_first < shifts; tile_first += tile) {
        hip_banded_overlap_tile(table, sequence, tile_first, tile, votes);
        for (u32_t counter = 0; counter != tile && tile_first + counter < shifts; ++counter)
            if (votes[counter] > most_votes) most_votes = votes[counter], voted_shift = tile_first + counter;
    }
    if (most_votes < banded_overlap_votes_k) return banded_overlap_ceiling_k;
    // The voted shift lays the image's symbols from `low` over the sequence's from `start`.
    u32_t const low = voted_shift < query_bytes - 1 ? query_bytes - 1 - voted_shift : 0;
    u32_t const start = voted_shift > query_bytes - 1 ? voted_shift - (query_bytes - 1) : 0;
    u32_t const span = min_of(query_bytes - low, bytes - start);
    u32_t const shorter = min_of(query_bytes, bytes), longer = max_of(query_bytes, bytes);
    u32_t const gap = banded_overlap_gap(costs);
    u32_t const cost = alignment_band(costs, symbols + low, sequence.data + start, span, banded_overlap_band(span),
                                      cells) +
                       gap * (shorter - span);
    return banded_overlap_distance(cost, gap, span, shorter, longer);
}

/** The banded overlap distance from @p query to @p sequence, the lower of its images', scored by
 *  lane @p lane alone. Out of line, so the body compiles once per unit. */
__device__ __noinline__ u64_t hip_banded_overlap_score(hip_banded_overlap_query_t query, basic::vram_entry_t sequence,
                                                       u32_t lane) noexcept {
    u32_t* const votes = query.votes + std::size_t {lane} * query.table.length;
    u32_t* const cells = query.cells + lane * hip_banded_overlap_cells(query.table.length);
    u64_t distance = hip_banded_overlap_image(*query.costs, query.table, query.symbols, sequence, votes, cells);
    if (query.images != 1)
        distance = min_of(distance, hip_banded_overlap_image(*query.costs, query.complement, query.complement_symbols,
                                                             sequence, votes, cells));
    return distance;
}

/**
 *  @brief Banded overlap on CDNA3 and CDNA4: the lower half of a 64-lane wavefront files the
 *      query's distinct windows into its span, and each lane scores a candidate of its own, voting
 *      the shift most of the candidate's windows agree on and aligning the band around it.
 *
 *  A lane to a candidate, because a band's steps follow one another and take most of a score, so
 *  a wavefront aligns as many candidates at once as it has lanes. Windows hash through
 *  StringZilla, whose residues are the CPU kits'. The parameters are the window widths and the
 *  folded costs with the complement.
 */
struct u8_cdna3_banded_overlap_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cdna3_k | sz_cap_cdna4_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "u8_cdna3_banded_overlap_kit_t";

    using warp_t = hip_warp<64>;
    using scan_t = cdna3_scan_t;
    using canonical_query_t = hip_banded_overlap_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The window widths a face hashes with and the costs and complement it scores under, set once
     *  before the first insert. */
    using canonical_parameters_t = window_parameters_t;

    /** Lanes scoring one sequence: one, on the kit's one rung. */
    inline static constexpr u32_t canonical_rungs_k[] = {1};

    /** A rung is its lanes per sequence. */
    static constexpr u32_t canonical_lanes_per_entry(u32_t lanes) noexcept { return lanes; }

    /** The one rung, or past it for a sequence longer than a window table places its windows in. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const&) noexcept {
        return longest > sz_ngrams_query_bytes_max_k ? std::size(canonical_rungs_k) : 0;
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone: its table, under
     *  the costs complement the complement image's bytes and table, then the vote counters and the
     *  band's cells of every lane. */
    static std::size_t canonical_span_bytes(byte_t const*, std::size_t bytes,
                                            canonical_parameters_t const& windows) noexcept {
        std::size_t const table_bytes = hip_window_table_bytes(bytes, windows);
        std::size_t const complement_bytes = windows.costs.complements
                                                 ? round_up_to_multiple<std::size_t>(bytes, 16) + table_bytes
                                                 : 0;
        return table_bytes + complement_bytes + hip_banded_overlap_votes_bytes<warp_t>(bytes) +
               round_up_to_multiple<std::size_t>(
                   std::size_t {warp_t::width_k} * hip_banded_overlap_cells(bytes) * sizeof(u32_t), 16);
    }

    /** Prepares @p sequence with the wavefront, and where the costs complement its complement image
     *  behind the table; the counters and cells past them are each lane's to overwrite. */
    static __device__ __forceinline__ canonical_query_t canonical_prepare(byte_t const* parameters,
                                                                          basic::vram_entry_t sequence, byte_t* span,
                                                                          u32_t lane) noexcept {
        canonical_parameters_t const& windows = *reinterpret_cast<canonical_parameters_t const*>(parameters);
        std::size_t const table_bytes = hip_window_table_bytes(sequence.bytes, windows);
        canonical_query_t query;
        query.costs = &windows.costs, query.images = static_cast<u32_t>(complement_images(windows.costs));
        query.table = query.complement = hip_window_table<warp_t>(windows, sequence, span, lane);
        query.symbols = query.complement_symbols = sequence.data;
        byte_t* counters = span + table_bytes;
        if (windows.costs.complements) {
            byte_t* const image = counters;
            hip_complement_image<warp_t>(windows.costs, sequence, image, lane);
            byte_t* const complement_span = image + round_up_to_multiple<std::size_t>(sequence.bytes, 16);
            query.complement = hip_window_table<warp_t>(windows, {image, sequence.bytes}, complement_span, lane);
            query.complement_symbols = image, counters = complement_span + table_bytes;
        }
        query.votes = reinterpret_cast<u32_t*>(counters);
        query.cells = reinterpret_cast<u32_t*>(counters + hip_banded_overlap_votes_bytes<warp_t>(sequence.bytes));
        return query;
    }

    /** No overlap costs less than nothing, nor spans more than the shorter side. */
    static __device__ __forceinline__ distance_t canonical_lower_bound(nk_kernel_kind_t, canonical_query_t const& query,
                                                                       basic::vram_entry_t sequence) noexcept {
        u64_t const query_bytes = query.table.length;
        return static_cast<distance_t>(banded_overlap_lower_bound(min_of<u64_t>(query_bytes, sequence.bytes),
                                                                  max_of<u64_t>(query_bytes, sequence.bytes)));
    }
    template <u32_t lanes_>
    static __device__ __forceinline__ distance_t canonical_distance(nk_kernel_kind_t, canonical_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t,
                                                                    u32_t lane) noexcept {
        return static_cast<distance_t>(hip_banded_overlap_score(query, sequence, lane));
    }

    /** No sketch: builders score canonically, and a stored value is its class bytes closed to a
     *  word. */
    using sketch_tables_t = std::monostate;

    /** No sketch, so no scratch beside one. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }
};

/** Banded overlap on CDNA5, as the CDNA3 kit scores it, the whole 32-lane wavefront filing the
 *  query. */
struct u8_cdna5_banded_overlap_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cdna5_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "u8_cdna5_banded_overlap_kit_t";

    using warp_t = hip_warp<32>;
    using scan_t = cdna5_scan_t;
    using canonical_query_t = hip_banded_overlap_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The window widths a face hashes with and the costs and complement it scores under, set once
     *  before the first insert. */
    using canonical_parameters_t = window_parameters_t;

    /** Lanes scoring one sequence: one, on the kit's one rung. */
    inline static constexpr u32_t canonical_rungs_k[] = {1};

    /** A rung is its lanes per sequence. */
    static constexpr u32_t canonical_lanes_per_entry(u32_t lanes) noexcept { return lanes; }

    /** The one rung, or past it for a sequence longer than a window table places its windows in. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const&) noexcept {
        return longest > sz_ngrams_query_bytes_max_k ? std::size(canonical_rungs_k) : 0;
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone: its table, under
     *  the costs complement the complement image's bytes and table, then the vote counters and the
     *  band's cells of every lane. */
    static std::size_t canonical_span_bytes(byte_t const*, std::size_t bytes,
                                            canonical_parameters_t const& windows) noexcept {
        std::size_t const table_bytes = hip_window_table_bytes(bytes, windows);
        std::size_t const complement_bytes = windows.costs.complements
                                                 ? round_up_to_multiple<std::size_t>(bytes, 16) + table_bytes
                                                 : 0;
        return table_bytes + complement_bytes + hip_banded_overlap_votes_bytes<warp_t>(bytes) +
               round_up_to_multiple<std::size_t>(
                   std::size_t {warp_t::width_k} * hip_banded_overlap_cells(bytes) * sizeof(u32_t), 16);
    }

    /** Prepares @p sequence with the wavefront, and where the costs complement its complement image
     *  behind the table; the counters and cells past them are each lane's to overwrite. */
    static __device__ __forceinline__ canonical_query_t canonical_prepare(byte_t const* parameters,
                                                                          basic::vram_entry_t sequence, byte_t* span,
                                                                          u32_t lane) noexcept {
        canonical_parameters_t const& windows = *reinterpret_cast<canonical_parameters_t const*>(parameters);
        std::size_t const table_bytes = hip_window_table_bytes(sequence.bytes, windows);
        canonical_query_t query;
        query.costs = &windows.costs, query.images = static_cast<u32_t>(complement_images(windows.costs));
        query.table = query.complement = hip_window_table<warp_t>(windows, sequence, span, lane);
        query.symbols = query.complement_symbols = sequence.data;
        byte_t* counters = span + table_bytes;
        if (windows.costs.complements) {
            byte_t* const image = counters;
            hip_complement_image<warp_t>(windows.costs, sequence, image, lane);
            byte_t* const complement_span = image + round_up_to_multiple<std::size_t>(sequence.bytes, 16);
            query.complement = hip_window_table<warp_t>(windows, {image, sequence.bytes}, complement_span, lane);
            query.complement_symbols = image, counters = complement_span + table_bytes;
        }
        query.votes = reinterpret_cast<u32_t*>(counters);
        query.cells = reinterpret_cast<u32_t*>(counters + hip_banded_overlap_votes_bytes<warp_t>(sequence.bytes));
        return query;
    }

    /** No overlap costs less than nothing, nor spans more than the shorter side. */
    static __device__ __forceinline__ distance_t canonical_lower_bound(nk_kernel_kind_t, canonical_query_t const& query,
                                                                       basic::vram_entry_t sequence) noexcept {
        u64_t const query_bytes = query.table.length;
        return static_cast<distance_t>(banded_overlap_lower_bound(min_of<u64_t>(query_bytes, sequence.bytes),
                                                                  max_of<u64_t>(query_bytes, sequence.bytes)));
    }
    template <u32_t lanes_>
    static __device__ __forceinline__ distance_t canonical_distance(nk_kernel_kind_t, canonical_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t,
                                                                    u32_t lane) noexcept {
        return static_cast<distance_t>(hip_banded_overlap_score(query, sequence, lane));
    }

    /** No sketch: builders score canonically, and a stored value is its class bytes closed to a
     *  word. */
    using sketch_tables_t = std::monostate;

    /** No sketch, so no scratch beside one. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_rocm_k) != 0)
struct banded_overlap_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, u8_cdna3_banded_overlap_kit_t, u8_cdna5_banded_overlap_kit_t>::type;
};

/** Bytes of an alignment profile stripe on @p warp_type_: a 64-bit word of costs per class
 *  and lane. */
template <typename warp_type_>
inline constexpr std::size_t hip_alignment_stripe_bytes = alignment_classes_k * warp_type_::width_k * sizeof(u64_t);

/** Codes of the longest sequence the widest alignment rung scores, the most a power of two whose
 *  costliest alignment a 16-bit cell holds; longer ones are refused. */
inline constexpr std::size_t hip_alignment_widest_codes_k = 4096;

/** Query codes each lane of @p warp_type_ holds for a query of @p codes: the fewest that fit the
 *  query in one stripe of a wavefront's lanes, the widest rung's past it. */
template <typename warp_type_>
constexpr u32_t hip_alignment_rows_per_lane(std::size_t codes) noexcept {
    constexpr u32_t widest_k = vram_alignment_rows_k[std::size(vram_alignment_rows_k) - 1];
    return static_cast<u32_t>(min_of<std::size_t>(
        widest_k, max_of<std::size_t>(1, divide_round_up<std::size_t>(codes, warp_type_::width_k))));
}

/** Stripes a query of @p codes takes on @p warp_type_, one at least. */
template <typename warp_type_>
constexpr u32_t hip_alignment_stripes(std::size_t codes) noexcept {
    std::size_t const stripe_rows = std::size_t {warp_type_::width_k} * hip_alignment_rows_per_lane<warp_type_>(codes);
    return static_cast<u32_t>(max_of<std::size_t>(1, divide_round_up(codes, stripe_rows)));
}

/** A query an alignment walk scores rows against, laid out in its span. */
struct hip_alignment_query_t {

    /** The profile: a stripe per stripe of the query, the cost of every class against each lane's
     *  codes. */
    byte_t* span;

    /** Past the profile, for a query of more than one stripe, a boundary row per wavefront of a
     *  block. */
    byte_t* boundaries;

    /** Codes of the query. */
    u32_t codes;

    /** Query codes each lane holds, at most its rung's. */
    u8_t rows_per_lane;

    /** Cost of one code facing a gap. */
    u8_t gap;
};

/** Writes lane @p lane 's profile words of @p query on @p warp_type_, the packed @p sequence under
 *  @p costs: the cost of every class against each of the lane's codes, a stripe at a time. */
template <typename warp_type_>
__device__ __forceinline__ void hip_alignment_profile(alignment_costs_t const& costs, basic::vram_entry_t sequence,
                                                      hip_alignment_query_t const& query, u32_t lane) noexcept {
    u32_t const stripes = hip_alignment_stripes<warp_type_>(query.codes);
    u64_t* const profile = reinterpret_cast<u64_t*>(query.span);
    // A word per stripe and class.
    for (u32_t word = 0; word != stripes * alignment_classes_k; ++word) {
        u32_t const stripe = word / alignment_classes_k, code = word % alignment_classes_k;
        u32_t const lane_first = (stripe * warp_type_::width_k + lane) * query.rows_per_lane;
        u64_t costs_word = 0;
        for (u32_t row_in_lane = 0; row_in_lane != query.rows_per_lane; ++row_in_lane) {
            u32_t const index = lane_first + row_in_lane;
            if (index >= query.codes) continue;
            u32_t const query_code = packed_code_at(sequence.data, index);
            costs_word |= u64_t {costs.substitutions[query_code][code]} << (8 * row_in_lane);
        }
        profile[word * warp_type_::width_k + lane] = costs_word;
    }
}

/** Lays the packed @p sequence out as a query in @p span under the costs at @p parameters, every
 *  lane of the wavefront of @p warp_type_ writing its own profile words. */
template <typename warp_type_>
__device__ __forceinline__ hip_alignment_query_t hip_alignment_prepare(byte_t const* parameters,
                                                                       basic::vram_entry_t sequence, byte_t* span,
                                                                       u32_t lane) noexcept {
    alignment_costs_t const& costs = *reinterpret_cast<alignment_costs_t const*>(parameters);
    hip_alignment_query_t query;
    query.span = span, query.gap = costs.gap;
    query.codes = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes));
    query.rows_per_lane = static_cast<u8_t>(hip_alignment_rows_per_lane<warp_type_>(query.codes));
    query.boundaries = span + std::size_t {hip_alignment_stripes<warp_type_>(query.codes)} *
                                  hip_alignment_stripe_bytes<warp_type_>;
    hip_alignment_profile<warp_type_>(costs, sequence, query, lane);
    warp_type_::sync();
    return query;
}

/**
 *  @brief The alignment cost from @p query to @p sequence, a wavefront of @p warp_type_ sweeping
 *      anti-diagonals over stripes of query codes, @p rows_ codes at most to a lane: exact within
 *      @p radius, and past it a stripe's bottom row minimum above it.
 *
 *  Lane @c l holds a stripe's rows from its @c l -th share down and enters column @c j at step
 *  @c j+l, taking the cell above and the next column's code from lane @c l-1 in one shuffled word,
 *  the first lane from a word the wavefront loads per its width of columns. Each stripe hands its
 *  bottom row to the next through the wavefront's boundary row, which the last lane writes a
 *  wavefront's width less one columns behind the reads. Out of line, so the body compiles once per
 *  rung and unit.
 */
template <u32_t rows_, typename warp_type_>
__device__ __noinline__ distance_t hip_alignment_score(hip_alignment_query_t query, basic::vram_entry_t sequence,
                                                       distance_t radius, u32_t lane) noexcept {
    using warp_t = warp_type_;
    constexpr u32_t lanes_k = warp_t::width_k;
    u32_t const gap = query.gap, rows = query.codes, rows_per_lane = query.rows_per_lane;
    u32_t const columns = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes));
    if (!rows || !columns) return static_cast<distance_t>((rows + columns) * gap);
    u32_t const stripe_rows = lanes_k * rows_per_lane, stripes = hip_alignment_stripes<warp_t>(rows);
    u64_t const* const profile = reinterpret_cast<u64_t const*>(query.span);
    u32_t* const boundary = reinterpret_cast<u32_t*>(query.boundaries) +
                            (threadIdx.x / lanes_k) * (hip_alignment_widest_codes_k + 1);
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
        u64_t const* const stripe_profile = profile + stripe * alignment_classes_k * lanes_k + lane;
        // The first lane's top cell of a column and the next column's code, a column to a lane.
        auto const load_columns = [&](u32_t first_column) noexcept {
            u32_t const column = first_column + lane;
            u32_t const top = column > columns ? 0u : stripe ? boundary[column] : column * gap;
            u32_t const next_code = column < columns ? packed_code_at(sequence.data, column) : 0u;
            return top << 5 | next_code;
        };
        // A cell rides one word with the code its receiver scores a step later, the cell in the
        // high bits, as far as 2^27 reaches; so each step loads the costs of the next step before
        // it scores its own cells.
        u32_t const first_code = packed_code_at(sequence.data, 0);
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
            // Lanes past the stripe, or outside it at this step, hand the cell above on unchanged.
            bool const scoring = lane < active && step >= lane && column <= columns;
            u32_t above = top, diagonal = diagonal_above;
#pragma unroll
            for (u32_t cell = 0; cell != rows_; ++cell) {
                bool const live = scoring && cell < rows_per_lane;
                u32_t const substitution = static_cast<u32_t>(costs_word >> (8 * cell)) & 0xFFu;
                u32_t const scored = min_of(min_of(diagonal + substitution, cells[cell] + gap), above + gap);
                diagonal = live ? cells[cell] : diagonal;
                cells[cell] = live ? scored : cells[cell];
                above = live ? scored : above;
            }
            diagonal_above = scoring ? top : diagonal_above;
            if (scoring && !closing && lane == lanes_k - 1)
                boundary[column] = above, bottom_minimum = min_of(bottom_minimum, above);
            u32_t closing_cell = result;
#pragma unroll
            for (u32_t cell = 0; cell != rows_; ++cell) closing_cell = cell == last_cell ? cells[cell] : closing_cell;
            if (scoring && closing && lane == last_lane && column == columns) result = closing_cell;
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

/*  Every cell and the cost added to it fit a 16-bit half, so plain 32-bit adds over two cells side
 *  by side never carry from the low half into the high. */
static_assert((hip_alignment_widest_codes_k + 1) * alignment_cost_ceiling_k <= (std::numeric_limits<u16_t>::max)(),
              "A cell pair's halves must not carry into each other");

/** The lower of each 16-bit half of @p first and @p second, in one packed <tt>v_pk_min_u16</tt>,
 *  which every CDNA and RDNA generation since gfx9 issues. */
__device__ __forceinline__ u32_t hip_min_halves(u32_t first, u32_t second) noexcept {
    u32_t lower;
    asm("v_pk_min_u16 %0, %1, %2" : "=v"(lower) : "v"(first), "v"(second));
    return lower;
}

/** Scores one column of a lane's @p rows_per_lane rows of @p cells against two candidates, the
 *  first in the low halves and the second in the high, from the cells @p above and @p diagonal to
 *  its first row and each row's costs in @p first_costs and @p second_costs; only the @p halves
 *  set take the new cells. Answers the lane's bottom cell. */
template <u32_t rows_>
__device__ __forceinline__ u32_t hip_alignment_column_pair(u32_t (&cells)[rows_], u32_t rows_per_lane,
                                                           u64_t first_costs, u64_t second_costs, u32_t gaps,
                                                           u32_t diagonal, u32_t above, u32_t halves) noexcept {
#pragma unroll
    for (u32_t cell = 0; cell != rows_; ++cell) {
        bool const live = cell < rows_per_lane;
        u32_t const substitutions = (static_cast<u32_t>(first_costs >> (8 * cell)) & 0xFFu) |
                                    (static_cast<u32_t>(second_costs >> (8 * cell)) & 0xFFu) << 16;
        u32_t const scored = hip_min_halves(diagonal + substitutions, hip_min_halves(cells[cell], above) + gaps);
        u32_t const kept = (scored & halves) | (cells[cell] & ~halves);
        diagonal = live ? cells[cell] : diagonal;
        cells[cell] = live ? kept : cells[cell];
        above = live ? kept : above;
    }
    return above;
}

/**
 *  @brief The alignment costs from @p query to @p first and to @p second at once, each the
 *      value @ref hip_alignment_score returns for it under its own radius.
 *
 *  The wavefront of the single scorer with two cells in every 32-bit register, @p first in the low
 *  half and @p second in the high, which plain adds and packed 16-bit minimums step together. A
 *  lane takes both tops from lane @c l-1 in one shuffled word, and both of the next column's codes
 *  in a second from the lane holding them in the wavefront's last two loads of its width of
 *  columns, five bits each. Steps run to the longer candidate; a lane past one candidate's last
 *  column keeps that half's cells, as the single scorer stops, and a candidate a stripe's crossing
 *  settles, or the last stripe finishes, keeps its distance while the other runs on, its half
 *  frozen from then on.
 */
template <u32_t rows_, typename warp_type_>
__device__ __noinline__ basic::vram_distance_pair_t hip_alignment_score_pair(
    hip_alignment_query_t query, basic::vram_entry_t first, basic::vram_entry_t second, distance_t first_radius,
    distance_t second_radius, u32_t lane) noexcept {
    using warp_t = warp_type_;
    constexpr u32_t lanes_k = warp_t::width_k, code_bits_k = packed_code_bits_k;
    constexpr u32_t codes_bits_k = 2 * code_bits_k;
    constexpr u32_t low_k = (std::numeric_limits<u16_t>::max)(), high_k = low_k << 16, code_k = (1u << code_bits_k) - 1;
    u32_t const gap = query.gap, gaps = gap << 16 | gap, rows = query.codes, rows_per_lane = query.rows_per_lane;
    u32_t const first_columns = static_cast<u32_t>(packed_codes_count(first.data, first.bytes));
    u32_t const second_columns = static_cast<u32_t>(packed_codes_count(second.data, second.bytes));
    basic::vram_distance_pair_t distances {static_cast<distance_t>((rows + first_columns) * gap),
                                           static_cast<distance_t>((rows + second_columns) * gap)};
    // A candidate's last column while it scores, zero once it holds its distance.
    u32_t first_limit = rows ? first_columns : 0u, second_limit = rows ? second_columns : 0u;
    if (!first_limit && !second_limit) return distances;
    u32_t const stripe_rows = lanes_k * rows_per_lane, stripes = hip_alignment_stripes<warp_t>(rows);
    u64_t const* const profile = reinterpret_cast<u64_t const*>(query.span);
    u32_t* const boundary = reinterpret_cast<u32_t*>(query.boundaries) +
                            (threadIdx.x / lanes_k) * (hip_alignment_widest_codes_k + 1);
    for (u32_t stripe = 0; stripe != stripes; ++stripe) {
        u32_t const first_row = stripe * stripe_rows, lane_first = first_row + lane * rows_per_lane;
        u32_t const active = min_of(lanes_k, divide_round_up(rows - first_row, rows_per_lane));
        u32_t const last_lane = (rows - 1 - first_row) / rows_per_lane,
                    last_cell = (rows - 1 - first_row) % rows_per_lane;
        bool const closing = stripe + 1 == stripes;
        u32_t cells[rows_];
#pragma unroll
        for (u32_t cell = 0; cell != rows_; ++cell) cells[cell] = (lane_first + cell + 1) * gaps;
        u64_t const* const stripe_profile = profile + stripe * alignment_classes_k * lanes_k + lane;
        // The halves a lane scores at a column: the candidates still running that reach it.
        auto const halves_at = [&](u32_t column) noexcept {
            return (column <= first_limit ? low_k : 0u) | (column <= second_limit ? high_k : 0u);
        };
        // The first lane's top cells of a column, a column to a lane.
        auto const load_tops = [&](u32_t first_column) noexcept {
            u32_t const column = first_column + lane;
            u32_t const halves = halves_at(column);
            return halves ? (stripe ? boundary[column] : column * gaps) & halves : 0u;
        };
        // Both candidates' codes of a column, a column to a lane.
        auto const load_codes = [&](u32_t first_column) noexcept {
            u32_t const column = first_column + lane;
            u32_t const first_code = column < first_limit ? packed_code_at(first.data, column) : 0u;
            u32_t const second_code = column < second_limit ? packed_code_at(second.data, column) : 0u;
            return first_code | second_code << code_bits_k;
        };
        u32_t const opening_codes = (first_limit ? packed_code_at(first.data, 0) : 0u) |
                                    (second_limit ? packed_code_at(second.data, 0) : 0u) << code_bits_k;
        // The previous load's codes in the low ten bits and the current one's above them; the first
        // column's codes stand as the last lane's of a load before the first.
        u32_t codes = (lane == lanes_k - 1 ? opening_codes : 0u) | load_codes(1) << codes_bits_k;
        u32_t tops = load_tops(1), handed = 0;
        u64_t first_costs = stripe_profile[(opening_codes & code_k) * lanes_k],
              second_costs = stripe_profile[(opening_codes >> code_bits_k) * lanes_k];
        u32_t diagonal_above = lane_first * gaps, bottom_minimum = (stripe + 1) * stripe_rows * gaps;
        u32_t const steps = max_of(first_limit, second_limit) + active - 1;
        for (u32_t step = 0; step != steps; ++step) {
            if (step % lanes_k == 0 && step)
                tops = load_tops(step + 1), codes = codes >> codes_bits_k | load_codes(step + 1) << codes_bits_k;
            u32_t const from_loaded = warp_t::shuffle(tops, step % lanes_k);
            u32_t const from_above = warp_t::shuffle_up(handed, 1);
            u32_t const top = lane == 0 ? from_loaded : from_above;
            // Lane `l` scores code `step - l + 1` next, in this load once `step % lanes >= l`.
            u32_t const next_codes = warp_t::shuffle(codes, (step - lane) % lanes_k) >>
                                     (step % lanes_k >= lane ? codes_bits_k : 0u);
            u64_t const next_first_costs = stripe_profile[(next_codes & code_k) * lanes_k];
            u64_t const next_second_costs = stripe_profile[(next_codes >> code_bits_k & code_k) * lanes_k];
            u32_t const column = step - lane + 1;
            // A lane yet to start, below the query or past both candidates scores no half, so what
            // it hands on reaches no lane that scores.
            u32_t const halves = lane < active && step >= lane ? halves_at(column) : 0u;
            u32_t const above = hip_alignment_column_pair<rows_>(cells, rows_per_lane, first_costs, second_costs, gaps,
                                                                 diagonal_above, top, halves);
            diagonal_above = (top & halves) | (diagonal_above & ~halves);
            // A kept half re-offers the bottom cell its last column already offered.
            if (!closing && lane == lanes_k - 1 && halves)
                boundary[column] = above, bottom_minimum = hip_min_halves(bottom_minimum, above);
            handed = above, first_costs = next_first_costs, second_costs = next_second_costs;
        }
        warp_t::sync();
        // A running candidate's half froze at its last column, so its last row's cell is its cost.
        u32_t last_row = 0;
#pragma unroll
        for (u32_t cell = 0; cell != rows_; ++cell) last_row = cell == last_cell ? cells[cell] : last_row;
        if (closing) {
            u32_t const results = warp_t::shuffle(last_row, last_lane);
            if (first_limit) distances.first = static_cast<distance_t>(results & low_k);
            if (second_limit) distances.second = static_cast<distance_t>(results >> 16);
            return distances;
        }
        // Every alignment crosses a stripe's bottom row, so none ends below its minimum.
        u32_t const crossing = warp_t::shuffle(bottom_minimum, lanes_k - 1);
        if (first_limit && static_cast<distance_t>(crossing & low_k) > first_radius)
            distances.first = static_cast<distance_t>(crossing & low_k), first_limit = 0;
        if (second_limit && static_cast<distance_t>(crossing >> 16) > second_radius)
            distances.second = static_cast<distance_t>(crossing >> 16), second_limit = 0;
        if (!first_limit && !second_limit) return distances;
    }
    return distances;
}

/** Alignment over packed codes on CDNA3 and CDNA4: a 64-lane wavefront to a sequence, an
 *  anti-diagonal sweep down stripes 64 lanes tall, its costs a profile the walk lays out in its
 *  span from the folded table the face uploads as the parameters. */
struct packed_cdna3_alignment_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cdna3_k | sz_cap_cdna4_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "packed_cdna3_alignment_kit_t";

    using warp_t = hip_warp<64>;
    using scan_t = cdna3_scan_t;
    using canonical_query_t = hip_alignment_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The folded costs a face scores with, set once before the first insert. */
    using canonical_parameters_t = alignment_costs_t;

    /** Query codes each lane holds, narrowest first. */
    inline static constexpr std::span<u32_t const> canonical_rungs_k = vram_alignment_rows_k;

    /** Every rung scores a sequence with the whole wavefront. */
    static constexpr u32_t canonical_lanes_per_entry(u32_t) noexcept { return warp_t::width_k; }

    /** The first rung holding whole the stripe of the most codes a @p longest -byte pack holds,
     *  the widest one any number of its stripes, or past every rung before any costs. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const& costs) noexcept {
        if (!costs.classes) return std::size(canonical_rungs_k);
        // A pack of these bytes holds at most this many codes past its one-byte header.
        std::size_t const codes = longest ? (longest - 1) * 8 / packed_code_bits_k : 0;
        std::size_t const widest = std::size(canonical_rungs_k) - 1;
        std::size_t rung = 0;
        while (rung != std::size(canonical_rungs_k) &&
               codes > (rung == widest ? hip_alignment_widest_codes_k
                                       : std::size_t {canonical_rungs_k[rung]} * warp_t::width_k))
            ++rung;
        return rung;
    }

    /** Bytes the packed @p bytes long sequence at @p data takes once prepared: its profile
     *  stripes, and past one stripe a boundary row per wavefront of a block. */
    static std::size_t canonical_span_bytes(byte_t const* data, std::size_t bytes,
                                            canonical_parameters_t const&) noexcept {
        u32_t const stripes = hip_alignment_stripes<warp_t>(packed_codes_count(data, bytes));
        std::size_t const boundaries = stripes > 1
                                           ? vram_warps_per_block_k * (hip_alignment_widest_codes_k + 1) * sizeof(u32_t)
                                           : 0;
        return stripes * hip_alignment_stripe_bytes<warp_t> + boundaries;
    }

    static __device__ __forceinline__ canonical_query_t canonical_prepare(byte_t const* parameters,
                                                                          basic::vram_entry_t sequence, byte_t* span,
                                                                          u32_t lane) noexcept {
        return hip_alignment_prepare<warp_t>(parameters, sequence, span, lane);
    }

    /** Every code one side holds past the other's length faces a gap. */
    static __device__ __forceinline__ distance_t canonical_lower_bound(nk_kernel_kind_t, canonical_query_t const& query,
                                                                       basic::vram_entry_t sequence) noexcept {
        u32_t const columns = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes));
        u32_t const difference = columns > query.codes ? columns - query.codes : query.codes - columns;
        return static_cast<distance_t>(difference * query.gap);
    }
    template <u32_t rows_>
    static __device__ __forceinline__ distance_t canonical_distance(nk_kernel_kind_t, canonical_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t radius,
                                                                    u32_t lane) noexcept {
        return hip_alignment_score<rows_, warp_t>(query, sequence, radius, lane);
    }
    template <u32_t rows_>
    static __device__ __forceinline__ basic::vram_distance_pair_t canonical_distance_pair(
        nk_kernel_kind_t, canonical_query_t const& query, basic::vram_entry_t first, basic::vram_entry_t second,
        distance_t first_radius, distance_t second_radius, u32_t lane) noexcept {
        return hip_alignment_score_pair<rows_, warp_t>(query, first, second, first_radius, second_radius, lane);
    }

    /** What the producer reads, uploaded once beside the kit's parameters as the buffer the view
     *  borrows. */
    using sketch_tables_t = spectrum_view_t;

    /** A smooth sketch sums in registers, so it carves no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the @p sketch_bytes -column smooth sketch of the stored @p sequence, packed class
     *  codes, at @p sketch, four-byte aligned, from the tables laid out at @p tables; every lane of
     *  one wavefront calls it for one value. */
    static __device__ __forceinline__ void sketch_write(byte_t const* tables, basic::vram_entry_t sequence, u32_t lane,
                                                        byte_t* sketch, u32_t sketch_bytes, byte_t*) noexcept {
        hip_sketch_packed<warp_t>(spectrum_view_of(tables), sequence, lane, sketch, sketch_bytes);
    }

    /** Squared Euclidean distance between the @p bytes -byte sketches at @p first and @p second,
     *  both four-byte aligned, accumulated in 32 bits by the calling lane alone; symmetric. */
    static __device__ __forceinline__ distance_t sketch_distance(byte_t const* first, byte_t const* second, u32_t bytes,
                                                                 u32_t) noexcept {
        return hip_smooth_distance(first, second, bytes);
    }
};

/** Alignment over packed codes on CDNA5, as the CDNA3 kit scores it, a 32-lane wavefront to
 *  a sequence. */
struct packed_cdna5_alignment_kit_t {
    inline static constexpr sz_capability_t capability_k = sz_cap_cdna5_k;

    /** The kit's identity in kernel symbols. */
    inline static constexpr char name_k[] = "packed_cdna5_alignment_kit_t";

    using warp_t = hip_warp<32>;
    using scan_t = cdna5_scan_t;
    using canonical_query_t = hip_alignment_query_t;
    using selector_t = basic::vram_keep_diverse_t;

    /** The folded costs a face scores with, set once before the first insert. */
    using canonical_parameters_t = alignment_costs_t;

    /** Query codes each lane holds, narrowest first. */
    inline static constexpr std::span<u32_t const> canonical_rungs_k = vram_alignment_rows_k;

    /** Every rung scores a sequence with the whole wavefront. */
    static constexpr u32_t canonical_lanes_per_entry(u32_t) noexcept { return warp_t::width_k; }

    /** The first rung holding whole the stripe of the most codes a @p longest -byte pack holds,
     *  the widest one any number of its stripes, or past every rung before any costs. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const& costs) noexcept {
        if (!costs.classes) return std::size(canonical_rungs_k);
        // A pack of these bytes holds at most this many codes past its one-byte header.
        std::size_t const codes = longest ? (longest - 1) * 8 / packed_code_bits_k : 0;
        std::size_t const widest = std::size(canonical_rungs_k) - 1;
        std::size_t rung = 0;
        while (rung != std::size(canonical_rungs_k) &&
               codes > (rung == widest ? hip_alignment_widest_codes_k
                                       : std::size_t {canonical_rungs_k[rung]} * warp_t::width_k))
            ++rung;
        return rung;
    }

    /** Bytes the packed @p bytes long sequence at @p data takes once prepared: its profile
     *  stripes, and past one stripe a boundary row per wavefront of a block. */
    static std::size_t canonical_span_bytes(byte_t const* data, std::size_t bytes,
                                            canonical_parameters_t const&) noexcept {
        u32_t const stripes = hip_alignment_stripes<warp_t>(packed_codes_count(data, bytes));
        std::size_t const boundaries = stripes > 1
                                           ? vram_warps_per_block_k * (hip_alignment_widest_codes_k + 1) * sizeof(u32_t)
                                           : 0;
        return stripes * hip_alignment_stripe_bytes<warp_t> + boundaries;
    }

    static __device__ __forceinline__ canonical_query_t canonical_prepare(byte_t const* parameters,
                                                                          basic::vram_entry_t sequence, byte_t* span,
                                                                          u32_t lane) noexcept {
        return hip_alignment_prepare<warp_t>(parameters, sequence, span, lane);
    }

    /** Every code one side holds past the other's length faces a gap. */
    static __device__ __forceinline__ distance_t canonical_lower_bound(nk_kernel_kind_t, canonical_query_t const& query,
                                                                       basic::vram_entry_t sequence) noexcept {
        u32_t const columns = static_cast<u32_t>(packed_codes_count(sequence.data, sequence.bytes));
        u32_t const difference = columns > query.codes ? columns - query.codes : query.codes - columns;
        return static_cast<distance_t>(difference * query.gap);
    }
    template <u32_t rows_>
    static __device__ __forceinline__ distance_t canonical_distance(nk_kernel_kind_t, canonical_query_t const& query,
                                                                    basic::vram_entry_t sequence, distance_t radius,
                                                                    u32_t lane) noexcept {
        return hip_alignment_score<rows_, warp_t>(query, sequence, radius, lane);
    }
    template <u32_t rows_>
    static __device__ __forceinline__ basic::vram_distance_pair_t canonical_distance_pair(
        nk_kernel_kind_t, canonical_query_t const& query, basic::vram_entry_t first, basic::vram_entry_t second,
        distance_t first_radius, distance_t second_radius, u32_t lane) noexcept {
        return hip_alignment_score_pair<rows_, warp_t>(query, first, second, first_radius, second_radius, lane);
    }

    /** What the producer reads, uploaded once beside the kit's parameters as the buffer the view
     *  borrows. */
    using sketch_tables_t = spectrum_view_t;

    /** A smooth sketch sums in registers, so it carves no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the @p sketch_bytes -column smooth sketch of the stored @p sequence, packed class
     *  codes, at @p sketch, four-byte aligned, from the tables laid out at @p tables; every lane of
     *  one wavefront calls it for one value. */
    static __device__ __forceinline__ void sketch_write(byte_t const* tables, basic::vram_entry_t sequence, u32_t lane,
                                                        byte_t* sketch, u32_t sketch_bytes, byte_t*) noexcept {
        hip_sketch_packed<warp_t>(spectrum_view_of(tables), sequence, lane, sketch, sketch_bytes);
    }

    /** Squared Euclidean distance between the @p bytes -byte sketches at @p first and @p second,
     *  both four-byte aligned, accumulated in 32 bits by the calling lane alone; symmetric. */
    static __device__ __forceinline__ distance_t sketch_distance(byte_t const* first, byte_t const* second, u32_t bytes,
                                                                 u32_t) noexcept {
        return hip_smooth_distance(first, second, bytes);
    }
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_rocm_k) != 0)
struct alignment_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, packed_cdna3_alignment_kit_t, packed_cdna5_alignment_kit_t>::type;
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
 *  them in @p kept as its host members read them: nothing for none, or a table's bytes. */
inline status_t vram_lay_out_canonical_parameters(std::monostate, growing_buffer<u32_t, std::allocator<byte_t>>&,
                                                  std::monostate&) noexcept {
    return status_code_t::success_k;
}
template <typename table_type_>
status_t vram_lay_out_canonical_parameters(table_type_ const& table,
                                           growing_buffer<u32_t, std::allocator<byte_t>>& laid,
                                           table_type_& kept) noexcept {
    if (status_t const sized = laid.resize(divide_round_up(sizeof(table_type_), sizeof(u32_t))); failed(sized))
        return sized;
    std::memcpy(laid.data(), &table, sizeof(table_type_));
    kept = table;
    return status_code_t::success_k;
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::canonical_parameters(
    canonical_parameters_t parameters) noexcept {
    return vram_lay_out_canonical_parameters(parameters, host_canonical_parameters_, kept_canonical_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::sketch_tables(
    typename kit_type_::sketch_tables_t const& tables, std::size_t sketch_bytes) noexcept {
    using tables_t = typename kit_type_::sketch_tables_t;
    // A sketch lands on whole words, and a kit writing none stores none.
    if (basic::a_vram_sketching_kit<kit_type_> ? !sketch_bytes || sketch_bytes % sizeof(u32_t) : sketch_bytes != 0)
        return {status_code_t::config_mismatch_k, "Sketch bytes not words, or given to a sketchless kit"};
    sketch_bytes_ = static_cast<u32_t>(sketch_bytes);
    // A kit reading no tables keeps none, and none is uploaded.
    if constexpr (std::is_empty_v<tables_t>) return status_code_t::success_k;
    else {
        if (tables.head->columns < sketch_bytes)
            return {status_code_t::config_mismatch_k, "Sketch tables narrower than the sketch"};
        bytes_view_t const laid = spectrum_bytes_of(tables);
        if (status_t const sized = host_sketch_tables_.resize_uninitialized(laid.size()); failed(sized)) return sized;
        std::memcpy(host_sketch_tables_.data(), laid.data(), laid.size());
        return status_code_t::success_k;
    }
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::rung_serving_(score_role_t role,
                                                                                  std::size_t longest) const noexcept {
    if constexpr (basic::a_vram_sketching_kit<kit_type_>)
        if (role == score_role_t::build_k) return 0;
    return kit_type_::canonical_rung_serving(longest, kept_canonical_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
u32_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::lanes_per_entry_(score_role_t role,
                                                                               std::size_t rung) noexcept {
    if constexpr (basic::a_vram_sketching_kit<kit_type_>)
        if (role == score_role_t::build_k) return 1;
    return kit_type_::canonical_lanes_per_entry(kit_type_::canonical_rungs_k[rung]);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
std::size_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::span_bytes_(score_role_t role, byte_t const* data,
                                                                                std::size_t bytes) const noexcept {
    if constexpr (basic::a_vram_sketching_kit<kit_type_>)
        if (role == score_role_t::build_k) return 0;
    return kit_type_::canonical_span_bytes(data, bytes, kept_canonical_parameters_);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
void vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::spell_kits_kernel_(
    vram_kernel_symbol_t& symbol, std::size_t rung, basic::amd_rung_t instantiated) noexcept {
    vram_kernel_symbol(symbol, "sequence", "amd_kernel", {"vram_arguments", 3, itanium_scope_t::basic_k},
                       instantiated.kernel, itanium_type<kit_type_> {}, kit_type_::canonical_rungs_k[rung],
                       itanium_type<node_id_type_> {}, instantiated.slots);
}

template <typename kit_type_, typename node_id_type_, typename kernels_type_>
status_t vram_amd_kits<kit_type_, node_id_type_, kernels_type_>::upload_parameters_(hip_api_t const& api) noexcept {
    using tables_t = typename kit_type_::sketch_tables_t;
    if ((host_sketch_tables_.empty() && !std::is_empty_v<tables_t>) ||
        (basic::a_vram_sketching_kit<kit_type_> && !sketch_bytes_) ||
        (host_canonical_parameters_.empty() && !std::is_empty_v<canonical_parameters_t>))
        return {status_code_t::uninitialized_k, "Sketch tables, sketch bytes or parameters not set"};
    if (!host_sketch_tables_.empty()) {
        device_sketch_tables_ = hip_make_unique(hip_device_allocator<byte_t> {api}, host_sketch_tables_.size());
        if (!device_sketch_tables_) return status_code_t::out_of_memory_k;
        if (api.copy(device_sketch_tables_.get(), host_sketch_tables_.data(), host_sketch_tables_.size(),
                     hip_copy_default_k) != hip_success_k)
            return status_code_t::device_k;
    }
    if (host_canonical_parameters_.empty()) return status_code_t::success_k;
    device_canonical_parameters_ = hip_make_unique(hip_device_allocator<u32_t> {api},
                                                   host_canonical_parameters_.size());
    if (!device_canonical_parameters_) return status_code_t::out_of_memory_k;
    if (api.copy(device_canonical_parameters_.get(), host_canonical_parameters_.data(),
                 host_canonical_parameters_.size() * sizeof(u32_t), hip_copy_default_k) != hip_success_k)
        return status_code_t::device_k;
    return status_code_t::success_k;
}

#pragma endregion Face

#pragma region Kernels

/** Every AMD sequence kernel: @p kernel_ picks the walk, @p kit_type_ scores at its @p rung_ of
 *  lanes per sequence or rows per lane, @p slots_ is the register rung, and @p node_id_type_ is the
 *  type of the record IDs it walks. A block is @c vram_warps_per_block_k of the kit's wavefronts,
 *  and the blocks @ref basic::hip_resident_blocks keeps per compute unit bound its registers, as
 *  waves per SIMD. */
template <vram_kernel_t kernel_, typename kit_type_, u32_t rung_, typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k* kit_type_::warp_t::width_k,
                                  basic::hip_resident_blocks(kernel_, slots_) * vram_warps_per_block_k / 4)
    amd_kernel(basic::vram_arguments<node_id_type_> arguments) {
    basic::hip_kernel_body<kernel_, kit_type_, rung_, slots_>(arguments);
}

/** Instantiates the kernel at @p position_ of @c basic::amd_rungs for @p kit_type_ on its @p rung_
 *  where the kit compiles it: a search on every rung, a builder on every rung too where it scores
 *  canonically and on the first alone where it scores sketches, which no rung sizes, and a seeding
 *  walk only there. */
template <typename kit_type_, u32_t rung_, typename node_id_type_, std::size_t position_>
void amd_instantiate_rung() noexcept {
    constexpr basic::amd_rung_t instantiated_k = basic::amd_rungs[position_];
    constexpr score_role_t role_k = vram_kernel_role(instantiated_k.kernel);
    constexpr bool sketching_k = basic::a_vram_sketching_kit<kit_type_>;
    constexpr bool compiled_k = instantiated_k.kernel == vram_kernel_t::seed_k
                                    ? sketching_k && rung_ == kit_type_::canonical_rungs_k[0]
                                    : role_k == score_role_t::search_k || !sketching_k ||
                                          rung_ == kit_type_::canonical_rungs_k[0];
    if constexpr (compiled_k)
        static_cast<void>(&amd_kernel<instantiated_k.kernel, kit_type_, rung_, node_id_type_, instantiated_k.slots>);
}

template <typename kit_type_, u32_t rung_, typename node_id_type_, std::size_t... rungs_>
void amd_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (amd_instantiate_rung<kit_type_, rung_, node_id_type_, rungs_>(), ...);
}

template <typename kit_type_, typename node_id_type_, std::size_t... kit_rungs_>
void amd_instantiate_kit_rungs(std::index_sequence<kit_rungs_...>) noexcept {
    (amd_instantiate_rungs<kit_type_, kit_type_::canonical_rungs_k[kit_rungs_], node_id_type_>(
         std::make_index_sequence<std::size(basic::amd_rungs)> {}),
     ...);
}

/** Instantiates every kernel of every rung of @p kit_type_ with @p node_id_type_ IDs, and on its
 *  first rung the kernels storing a load's values and sketching an insert's, empty for a kit
 *  writing no sketch; the unit compiling the code object names it once per kit and node-ID type.
 *  Host-side, since HIP-Clang refuses a kernel's address in device code, while a host reference
 *  still makes the device pass emit it; only a device-only unit names it, so no host pass emits
 *  stubs to launch them. */
template <typename kit_type_, typename node_id_type_>
void amd_instantiate() noexcept {
    amd_instantiate_kit_rungs<kit_type_, node_id_type_>(
        std::make_index_sequence<std::size(kit_type_::canonical_rungs_k)> {});
    constexpr u32_t first_rung = kit_type_::canonical_rungs_k[0];
    static_cast<void>(&amd_kernel<vram_kernel_t::load_values_k, kit_type_, first_rung, node_id_type_, 0>);
    // The face loads every kit's sketching kernel, which a kit writing no sketch leaves empty.
    static_cast<void>(&amd_kernel<vram_kernel_t::sketch_values_k, kit_type_, first_rung, node_id_type_, 0>);
}

#pragma endregion Kernels

} // namespace unum::usearch::sequence
