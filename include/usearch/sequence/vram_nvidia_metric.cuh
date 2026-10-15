/**
 *  @file include/usearch/sequence/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device kits for the sequence VRAM indexes, the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one sequence distance to one GPU generation, named like the dense kits; the
 *  generation slot names the oldest tier whose instructions it needs. Byte Levenshtein needs
 *  nothing past the @c ampere floor: StringZilla's serial Myers steps run on the device unchanged
 *  through @c --expt-relaxed-constexpr, so the distances are the same integers the CPU kits score.
 *
 *  Any compiler reads a kit's host facts, the rows it serves and the span a row prepares into, so
 *  the host index picks kits and sizes spans from the kits themselves; the device members compile
 *  only where CUDA does.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bit> // `std::popcount` - a row's distinct bytes

#include <stringzilla/levenshtein/serial.h> // `sz_levenshtein_u64x1_step_serial` & kin - the device-callable steps

#include "usearch/utilities/types.hpp"

#if defined(__CUDACC__)
#include <cstdint> // `std::uintptr_t` - a row's alignment within its word

#include <limits>  // `std::numeric_limits` - a ladder no rung of which serves the query
#include <utility> // `std::make_integer_sequence` - the word-count ladder

#include <stringzilla/levenshtein/simt.cuh> // `sz_levenshtein_query_prepare_simt_`

#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#endif

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/**
 *  @brief Levenshtein over the @p symbol_ of byte strings, bytes or UTF-8 runes, for queries of up
 *      to @p words_ Myers words, through StringZilla's serial steps, the query's classes and masks
 *      in the span its walk prepares.
 *
 *  Up to sixteen words a lane scores a candidate alone, stepping it at the query's exact word count
 *  from a compile-time ladder, so the verticals stay in registers. Past them the four lanes of a
 *  group split a candidate's words, each a character behind the lane below it, so the carry out of
 *  the lane below's top word arrives through one shuffle, and a block's four warps score a hop's
 *  candidates together. A candidate streams through by aligned words of its bytes, or rune by rune,
 *  and stops once its running score less the symbols left exceeds the radius.
 */
template <sz_levenshtein_symbol_t symbol_, u32_t words_>
struct u8_ampere_levenshtein_kit {

    /** The integer naming it in the kernels: the symbol kind above the Myers words of the longest
     *  query it scores. */
    inline static constexpr u32_t kit_k = static_cast<u32_t>(symbol_) << 8 | words_;

    /** Lanes scoring one candidate: one up to sixteen words, which a lane's registers hold and
     *  where a lane outruns a split group, and a group of four past them. */
    inline static constexpr u32_t lanes_per_row_k = words_ > 16 ? 4 : 1;

    /** Whether rows of @p bytes fit its words. */
    static constexpr bool serves(std::size_t bytes) noexcept { return bytes <= std::size_t {words_} * 64; }

    /** Bytes the @p bytes long row at @p data takes once prepared, a multiple of 16: a class map
     *  over every byte value, or a page table with a class row per page its runes touch, then a
     *  mask row per class the row's own words long, a class per distinct symbol and one for the
     *  rest. */
    static constexpr std::size_t span_bytes(byte_t const* data, std::size_t bytes) noexcept {
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
                // A row past the widest kit is refused before it prepares, so it needs no count.
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

#if defined(__CUDACC__)
    using warp_t = cuda_warp_t;
    using query_t = sz_levenshtein_query_t;

    /** Lays @p row out as a query in @p span, every lane of the warp taking part. */
    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t* span, u32_t lane) noexcept {
        query_t query;
        if constexpr (symbol_ == sz_levenshtein_runes_k)
            sz_levenshtein_query_prepare_utf8_simt_(reinterpret_cast<sz_cptr_t>(row.data), row.bytes,
                                                    reinterpret_cast<sz_ptr_t>(span), &query, lane);
        else
            sz_levenshtein_query_prepare_simt_(reinterpret_cast<sz_cptr_t>(row.data), row.bytes,
                                               sz_levenshtein_query_words(row.bytes), reinterpret_cast<sz_ptr_t>(span),
                                               &query, lane);
        return query;
    }

    /** No edit sequence is shorter than the two lengths' difference; a row of @c b bytes holds
     *  between a quarter of @c b runes and @c b of them. */
    static __device__ __forceinline__ distance_t bound(query_t const& query, basic::vram_row_t row) noexcept {
        u64_t const fewest = symbol_ == sz_levenshtein_runes_k ? divide_round_up<u64_t>(row.bytes, 4) : row.bytes;
        u64_t const most = row.bytes;
        return static_cast<distance_t>(fewest > query.length ? fewest - query.length
                                       : query.length > most ? query.length - most
                                                             : 0);
    }

    /** Exact within @p radius, the running score past it; the lanes of a group return the same. */
    static __device__ __forceinline__ distance_t score(query_t const& query, basic::vram_row_t row, distance_t radius,
                                                       u32_t lane) noexcept {
        if (!query.length || !row.bytes) return static_cast<distance_t>(query.length + row.bytes);
        sz_ssize_t const limit = radius < 0x1p62f ? static_cast<sz_ssize_t>(radius) : STRINGZILLA_SSIZE_MAX;
        if constexpr (lanes_per_row_k == 1)
            return score_ladder_(query, row, limit, std::make_integer_sequence<u32_t, words_> {});
        else return score_split_(query, row, limit, lane);
    }

  private:

    /** Steps the candidate at the query's exact word count, one rung of the ladder up to @c words_.
     *  The host never launches a kit narrower than its longest query. */
    template <u32_t... rungs_>
    static __device__ __forceinline__ distance_t score_ladder_(query_t const& query, basic::vram_row_t row,
                                                               sz_ssize_t limit,
                                                               std::integer_sequence<u32_t, rungs_...>) noexcept {
        distance_t distance = std::numeric_limits<distance_t>::infinity();
        [[maybe_unused]] bool const stepped =
            ((query.stride == rungs_ + 1 && (distance = score_words_<rungs_ + 1>(query, row, limit), true)) || ...);
        return distance;
    }

    template <u32_t words_count_>
    static __device__ __forceinline__ distance_t score_words_(query_t const& shared_query, basic::vram_row_t row,
                                                              sz_ssize_t limit) noexcept {
        // A local copy: no store through the verticals aliases it, so its fields stay in registers.
        query_t const query = shared_query;
        sz_levenshtein_u64x1_state_serial_t state;
        sz_levenshtein_u64x1_vertical_serial_t verticals[words_count_];
        sz_levenshtein_u64x1_init_serial(&state, verticals, words_count_, &query);
        if constexpr (symbol_ == sz_levenshtein_runes_k) {
            sz_cptr_t const text = reinterpret_cast<sz_cptr_t>(row.data);
            sz_size_t position = 0;
            // The bytes left bound the runes left, so stopping on them stays sound.
            for (u32_t runes = 0; position < row.bytes; ++runes) {
                if (runes % 8 == 0 &&
                    sz_levenshtein_u64x1_any_active_serial(&state, row.bytes, position, limit) == sz_false_k)
                    break;
                sz_rune_t const rune = sz_utf8_next_rune_(text, row.bytes, &position);
                sz_levenshtein_u64x1_step_serial(&state, verticals, words_count_, &query,
                                                 sz_levenshtein_utf8_class(&query, rune));
            }
            return static_cast<distance_t>(sz_levenshtein_u64x1_score_serial(&state, 0));
        }
        // Aligned words, so every load is whole and the tape's margin covers the last.
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(row.data);
        u64_t const* word_pointer = reinterpret_cast<u64_t const*>(address & ~std::uintptr_t {7});
        u32_t skipped = static_cast<u32_t>(address & 7u);
        for (u32_t position = 0; position < row.bytes; ++word_pointer, skipped = 0) {
            if (sz_levenshtein_u64x1_any_active_serial(&state, row.bytes, position, limit) == sz_false_k) break;
            u64_t bytes = *word_pointer >> (skipped * 8);
            u32_t const available = min_of(8u - skipped, row.bytes - position);
#pragma unroll
            for (u32_t byte = 0; byte != 8; ++byte, bytes >>= 8)
                if (byte < available)
                    sz_levenshtein_u64x1_step_serial(&state, verticals, words_count_, &query,
                                                     query.byte_to_class[bytes & 0xFFu]);
            position += available;
        }
        return static_cast<distance_t>(sz_levenshtein_u64x1_score_serial(&state, 0));
    }

    /** Steps the candidate with the calling lane's group, each lane owning its share of the query's
     *  words; the lane owning the last word keeps the score and decides, every eight steps, whether
     *  the group stops. Words past the query's are never stepped. */
    static __device__ __forceinline__ distance_t score_split_(query_t const& shared_query, basic::vram_row_t row,
                                                              sz_ssize_t limit, u32_t lane) noexcept {
        constexpr u32_t lane_words_k = words_ / lanes_per_row_k;
        query_t const query = shared_query;
        u32_t const group_lane = lane % lanes_per_row_k;
        u32_t const group_mask = ((1u << lanes_per_row_k) - 1u) << (lane - group_lane);
        u32_t const words = static_cast<u32_t>(query.stride);
        u32_t const lane_words = divide_round_up(words, lanes_per_row_k);
        u32_t const first_word = group_lane * lane_words;
        u32_t const score_lane = (words - 1) / lane_words;
        u64_t const last_symbol_bit = u64_t {1} << ((query.length - 1) & 63u);
        sz_levenshtein_u64x1_state_serial_t state;
        sz_levenshtein_u64x1_vertical_serial_t verticals[lane_words_k];
        sz_levenshtein_u64x1_init_serial(&state, verticals, lane_words_k, &query);
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(row.data);
        sz_cptr_t const text = reinterpret_cast<sz_cptr_t>(row.data);
        u32_t symbols = row.bytes;
        if constexpr (symbol_ == sz_levenshtein_runes_k) {
            symbols = 0;
            for (sz_size_t position = 0; position < row.bytes; ++symbols)
                sz_utf8_next_rune_(text, row.bytes, &position);
        }
        sz_size_t cursor = 0;
        u64_t bytes = 0;
        u32_t carry = 0;
        for (u32_t step = 0; step != symbols + score_lane; ++step) {
            // Horizontal deltas out of the lane below: positive in bit zero, negative in bit one.
            u32_t const received = __shfl_up_sync(group_mask, carry, 1, lanes_per_row_k);
            if (step % 8 == 0) {
                bool active = true;
                if (group_lane == score_lane && step > score_lane)
                    active = sz_levenshtein_u64x1_any_active_serial(&state, symbols, step - score_lane, limit) ==
                             sz_true_k;
                if (!__shfl_sync(group_mask, active, score_lane, lanes_per_row_k)) break;
            }
            u32_t const position = step - group_lane;
            if (group_lane > score_lane || step < group_lane || position >= symbols) continue;
            u32_t class_id;
            if constexpr (symbol_ == sz_levenshtein_runes_k)
                class_id = sz_levenshtein_utf8_class(&query, sz_utf8_next_rune_(text, row.bytes, &cursor));
            else {
                std::uintptr_t const byte_address = address + position;
                if (position == 0 || (byte_address & 7u) == 0)
                    bytes = *reinterpret_cast<u64_t const*>(byte_address & ~std::uintptr_t {7});
                class_id = query.byte_to_class[(bytes >> ((byte_address & 7u) * 8)) & 0xFFu];
            }
            u64_t const* const masks = query.masks + class_id * query.stride + first_word;
            // The row above the query's first word is one edit higher than the cell left of it.
            u64_t positive_carry = group_lane ? received & 1u : 1u;
            u64_t negative_carry = group_lane ? (received >> 1) & 1u : 0u;
#pragma unroll
            for (u32_t word = 0; word != lane_words_k; ++word) {
                if (word >= lane_words || first_word + word >= words) continue;
                sz_levenshtein_u64x1_vertical_serial_t& vertical = verticals[word];
                u64_t const equality = masks[word];
                u64_t const vertical_carry = equality | vertical.negative;
                u64_t const matched = equality | negative_carry;
                u64_t const diagonal = (((matched & vertical.positive) + vertical.positive) ^ vertical.positive) |
                                       matched;
                u64_t horizontal_positive = vertical.negative | ~(diagonal | vertical.positive);
                u64_t horizontal_negative = vertical.positive & diagonal;
                if (first_word + word + 1 == words) {
                    state.score += (horizontal_positive & last_symbol_bit) != 0;
                    state.score -= (horizontal_negative & last_symbol_bit) != 0;
                }
                u64_t const next_positive_carry = horizontal_positive >> 63;
                u64_t const next_negative_carry = horizontal_negative >> 63;
                horizontal_positive = (horizontal_positive << 1) | positive_carry;
                horizontal_negative = (horizontal_negative << 1) | negative_carry;
                positive_carry = next_positive_carry, negative_carry = next_negative_carry;
                vertical.positive = horizontal_negative | ~(vertical_carry | horizontal_positive);
                vertical.negative = horizontal_positive & vertical_carry;
            }
            carry = static_cast<u32_t>(positive_carry | (negative_carry << 1));
        }
        return static_cast<distance_t>(__shfl_sync(group_mask, state.score, score_lane, lanes_per_row_k));
    }
#endif
};

} // namespace unum::usearch::sequence
