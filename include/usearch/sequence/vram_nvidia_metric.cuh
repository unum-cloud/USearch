/**
 *  @file include/usearch/sequence/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device kits for the sequence VRAM indexes, the GPU twin of @c ram_metric.hpp, their
 *      ladder, the kernels, and the members of @c vram_nvidia_index.hpp reading them.
 *
 *  A kit binds one sequence distance to one GPU generation, named like the dense kits; the
 *  generation slot names the oldest tier whose instructions it needs. Byte Levenshtein needs
 *  nothing past the @c ampere floor: StringZilla's serial Myers steps run on the device unchanged
 *  through @c --expt-relaxed-constexpr, so the distances are the same integers the CPU kits score.
 *
 *  A kit scores on a rung, the Myers words of the longest query its kernels serve, which each
 *  kernel takes beside the ladder's inputs, so the verticals stay as narrow as the queries allow.
 *  Only the modality's NVIDIA unit compiles this file, so dispatch never names a kit.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - a row's alignment within its word

#include <bit>      // `std::bit_ceil`, `std::popcount` - a row's distinct pages and bytes
#include <iterator> // `std::size` - the rungs
#include <limits>   // `std::numeric_limits` - a ladder no rung of which serves the query
#include <utility>  // `std::make_integer_sequence` - the word-count ladder, every rung's kernels

#include <numkong/types.h>                  // `NUMKONG_ARCH_CUDA_AMPERE_` - the kits this unit carries
#include <stringzilla/levenshtein/serial.h> // `sz_levenshtein_u64x1_step_serial` & kin - the device-callable steps
#include <stringzilla/levenshtein/simt.cuh> // `sz_levenshtein_query_prepare_simt_`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"        // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.cuh" // `basic::cuda_kernel_body`
#include "usearch/sequence/ram_metric.hpp"     // `levenshtein_ladder`
#include "usearch/sequence/vram_nvidia_index.hpp"

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

#pragma region Kits

/** Lanes scoring one candidate for queries of up to @p words Myers words: one up to sixteen,
 *  which a lane's registers hold and where a lane outruns a split group, and a group of four past
 *  them. */
constexpr u32_t cuda_levenshtein_lanes_per_row_ampere(u32_t words) noexcept { return words > 16 ? 4 : 1; }

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

/** Steps the candidate over @p symbol_ at exactly @p words_ words, through a local copy of the
 *  query, so no store through the verticals aliases it and its fields stay in registers. */
template <sz_levenshtein_symbol_t symbol_, u32_t words_>
__device__ __forceinline__ distance_t cuda_levenshtein_words_ampere(sz_levenshtein_query_t const& shared_query,
                                                                    basic::vram_row_t row, sz_ssize_t limit) noexcept {
    sz_levenshtein_query_t const query = shared_query;
    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_vertical_serial_t verticals[words_];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words_, &query);
    if constexpr (symbol_ == sz_levenshtein_runes_k) {
        sz_cptr_t const text = reinterpret_cast<sz_cptr_t>(row.data);
        sz_size_t position = 0;
        // The bytes left bound the runes left, so stopping on them stays sound.
        for (u32_t runes = 0; position < row.bytes; ++runes) {
            if (runes % 8 == 0 &&
                sz_levenshtein_u64x1_any_active_serial(&state, row.bytes, position, limit) == sz_false_k)
                break;
            sz_rune_t const rune = sz_utf8_next_rune_(text, row.bytes, &position);
            sz_levenshtein_u64x1_step_serial(&state, verticals, words_, &query,
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
                sz_levenshtein_u64x1_step_serial(&state, verticals, words_, &query, query.byte_to_class[bytes & 0xFFu]);
        position += available;
    }
    return static_cast<distance_t>(sz_levenshtein_u64x1_score_serial(&state, 0));
}

/** Steps the candidate at the query's exact word count, one rung of the compile-time ladder up to
 *  the kernel's words; the host never launches a rung narrower than its longest query. */
template <sz_levenshtein_symbol_t symbol_, u32_t... rungs_>
__device__ __forceinline__ distance_t cuda_levenshtein_ladder_ampere(sz_levenshtein_query_t const& query,
                                                                     basic::vram_row_t row, sz_ssize_t limit,
                                                                     std::integer_sequence<u32_t, rungs_...>) noexcept {
    distance_t distance = std::numeric_limits<distance_t>::infinity();
    [[maybe_unused]] bool const stepped =
        ((query.stride == rungs_ + 1 &&
          (distance = cuda_levenshtein_words_ampere<symbol_, rungs_ + 1>(query, row, limit), true)) ||
         ...);
    return distance;
}

/** Steps the candidate over @p symbol_ with the calling lane's group for queries of up to @p words_
 *  words, each lane owning its share of the query's words; the lane owning the last word keeps the
 *  score and decides, every eight steps, whether the group stops. Words past the query's are never
 *  stepped. */
template <sz_levenshtein_symbol_t symbol_, u32_t words_>
__device__ __forceinline__ distance_t cuda_levenshtein_split_ampere(sz_levenshtein_query_t const& shared_query,
                                                                    basic::vram_row_t row, sz_ssize_t limit,
                                                                    u32_t lane) noexcept {
    constexpr u32_t lanes_per_row_k = cuda_levenshtein_lanes_per_row_ampere(words_);
    constexpr u32_t lane_words_k = words_ / lanes_per_row_k;
    sz_levenshtein_query_t const query = shared_query;
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
        for (sz_size_t position = 0; position < row.bytes; ++symbols) sz_utf8_next_rune_(text, row.bytes, &position);
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
                active = sz_levenshtein_u64x1_any_active_serial(&state, symbols, step - score_lane, limit) == sz_true_k;
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
            u64_t const diagonal = (((matched & vertical.positive) + vertical.positive) ^ vertical.positive) | matched;
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

/**
 *  @brief Scores over @p symbol_ for queries of up to @p words_ Myers words, exact within @p radius
 *      and the running score past it; the lanes of a group return the same.
 *
 *  Up to sixteen words a lane scores a candidate alone, stepping it at the query's exact word count
 *  from a compile-time ladder, so the verticals stay in registers. Past them the four lanes of a
 *  group split a candidate's words, each a character behind the lane below it, so the carry out of
 *  the lane below's top word arrives through one shuffle, and a block's four warps score a hop's
 *  candidates together. A candidate streams through by aligned words of its bytes, or rune by rune,
 *  and stops once its running score less the symbols left exceeds the radius.
 */
template <sz_levenshtein_symbol_t symbol_, u32_t words_>
__device__ __forceinline__ distance_t cuda_levenshtein_score_ampere(sz_levenshtein_query_t const& query,
                                                                    basic::vram_row_t row, distance_t radius,
                                                                    u32_t lane) noexcept {
    if (!query.length || !row.bytes) return static_cast<distance_t>(query.length + row.bytes);
    sz_ssize_t const limit = radius < 0x1p62f ? static_cast<sz_ssize_t>(radius) : STRINGZILLA_SSIZE_MAX;
    if constexpr (cuda_levenshtein_lanes_per_row_ampere(words_) == 1)
        return cuda_levenshtein_ladder_ampere<symbol_>(query, row, limit, std::make_integer_sequence<u32_t, words_> {});
    else return cuda_levenshtein_split_ampere<symbol_, words_>(query, row, limit, lane);
}

/** Levenshtein over bytes on the Ampere floor, through StringZilla's serial steps, the query's
 *  classes and masks in the span its walk prepares. Serves queries of up to 64 symbols a word. */
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

#pragma endregion Kits

#pragma region Face

template <sz_levenshtein_symbol_t symbol_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
std::size_t vram_nvidia_index<symbol_, node_id_type_, capability_, kernels_type_>::rung_serving_(
    std::size_t longest) noexcept {
    using kit_t = typename levenshtein_ladder<symbol_, capability_>::kit_t;
    std::size_t rung = 0;
    while (rung != rungs_k && !kit_t::serves(vram_levenshtein_words_k[rung], longest)) ++rung;
    return rung;
}

template <sz_levenshtein_symbol_t symbol_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
u32_t vram_nvidia_index<symbol_, node_id_type_, capability_, kernels_type_>::lanes_per_row_(std::size_t rung) noexcept {
    using kit_t = typename levenshtein_ladder<symbol_, capability_>::kit_t;
    return kit_t::lanes_per_row(vram_levenshtein_words_k[rung]);
}

template <sz_levenshtein_symbol_t symbol_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
std::size_t vram_nvidia_index<symbol_, node_id_type_, capability_, kernels_type_>::span_bytes_(
    byte_t const* data, std::size_t bytes) noexcept {
    using kit_t = typename levenshtein_ladder<symbol_, capability_>::kit_t;
    return kit_t::span_bytes(data, bytes);
}

template <sz_levenshtein_symbol_t symbol_, typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
status_t vram_nvidia_index<symbol_, node_id_type_, capability_, kernels_type_>::load_kernels_(int device) noexcept {
    auto [module, status] = cuda_module_t::load(device, kernels_type_::bytes());
    if (failed(status)) return status;
    module_ = std::move(module);
    vram_kernel_symbol_t symbol;
    for (std::size_t rung = 0; rung != rungs_k; ++rung)
        for (basic::cuda_rung_t const instantiated : basic::cuda_rungs) {
            vram_kernel_symbol(symbol, "sequence", "cuda_kernel", {"vram_arguments", 4, itanium_scope_t::basic_k},
                               instantiated.kernel, capability_, symbol_, vram_levenshtein_words_k[rung],
                               itanium_type<node_id_type_> {}, instantiated.slots);
            cuda_function_handle_t& function =
                kernels_[rung][static_cast<u32_t>(instantiated.kernel)][gpu_rung_index(instantiated.slots)];
            if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
        }
    return status_t::success_k;
}

#pragma endregion Face

#pragma region Kernels

/** Every NVIDIA sequence kernel: @p kernel_ picks the walk, @p capability_ and @p symbol_ the kit
 *  through @c levenshtein_ladder, @p words_ its rung, @p node_id_type_ the IDs of the records, and
 *  @p slots_ the register rung. */
template <vram_kernel_t kernel_, sz_capability_t capability_, sz_levenshtein_symbol_t symbol_, u32_t words_,
          typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, basic::cuda_resident_blocks(kernel_, slots_))
    cuda_kernel(basic::vram_arguments<node_id_type_> arguments) {
    using kit_t = typename levenshtein_ladder<symbol_, capability_>::kit_t;
    basic::cuda_kernel_body<kernel_, kit_t, words_, slots_>(arguments);
}

template <sz_capability_t capability_, sz_levenshtein_symbol_t symbol_, u32_t words_, typename node_id_type_,
          std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<basic::cuda_rungs[rungs_].kernel, capability_, symbol_, words_, node_id_type_,
                                    basic::cuda_rungs[rungs_].slots>),
     ...);
}

template <sz_capability_t capability_, sz_levenshtein_symbol_t symbol_, typename node_id_type_, std::size_t... words_>
__device__ void cuda_instantiate_words(std::index_sequence<words_...>) noexcept {
    (cuda_instantiate_rungs<capability_, symbol_, vram_levenshtein_words_k[words_], node_id_type_>(
         std::make_index_sequence<std::size(basic::cuda_rungs)> {}),
     ...);
}

/** Instantiates every kernel of every rung over @p symbol_ under @p capability_ with
 *  @p node_id_type_ IDs; the unit compiling the fatbin names it once per symbol and node-ID type.
 *  Taking the kernels' addresses in device code makes NVCC emit them, while a host-only pass emits
 *  no launch stub. */
template <sz_capability_t capability_, sz_levenshtein_symbol_t symbol_, typename node_id_type_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_words<capability_, symbol_, node_id_type_>(
        std::make_index_sequence<std::size(vram_levenshtein_words_k)> {});
}

#pragma endregion Kernels

} // namespace unum::usearch::sequence
