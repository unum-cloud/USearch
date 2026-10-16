/**
 *  @file include/usearch/utilities/vram_walks.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The host half of what every GPU vendor's walks share: the arena slice a walking warp
 *      carves, the register ladder's rungs, and the bitsets the graph passes mark.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <bit> // `std::bit_ceil`, `std::bit_width`, `std::countr_zero`

#include "usearch/utilities/types.hpp" // `vram_widest_pool_slots_k`, `divide_round_up`

namespace unum::usearch::utilities {

/** Bytes of one packed traversal candidate on the device, the kernels' scored node. */
inline constexpr std::size_t gpu_scored_node_bytes_k = sizeof(u64_t);

/** Layout of one warp's slice of the traversal arena, computed on the host: the exact visited set,
 *  the wide frontier's spill tail when it has one, then a builder's scratch when shared memory
 *  cannot seat it. The kernels carry it as their own @c vram_arena_shape_t. */
struct gpu_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;
    u32_t cache_bytes;

    /** A multiple of 16, as the kernels' twin rounds it. */
    constexpr std::size_t slice_bytes() const noexcept {
        std::size_t const bytes = visited_slots * sizeof(u64_t) + tail_entries * gpu_scored_node_bytes_k + cache_bytes;
        return (bytes + 15) & ~std::size_t {15};
    }
};

/** The slice of a walk expanding @p expansion candidates: the exact visited set, sized well past
 *  the expected visit count so probing stays short, then @p tail_entries of spill tail and
 *  @p cache_bytes of builder scratch. */
constexpr gpu_arena_shape_t gpu_arena_shape(u32_t expansion, u32_t tail_entries, u32_t cache_bytes) noexcept {
    u32_t const visited_slots = std::bit_ceil(64 * expansion);
    return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)), tail_entries, cache_bytes};
}

/** Rounds @p expansion to the register ladder's rungs of 2 to @c vram_widest_pool_slots_k slots
 *  of @p lanes ranks each, or to zero past them, where the arena frontier walks the expansion
 *  itself; @p default_slots when it asks for none. */
constexpr u32_t gpu_pool_slots(u32_t expansion, u32_t default_slots, u32_t lanes = 32) noexcept {
    if (!expansion) return default_slots;
    u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion, lanes));
    return slots > vram_widest_pool_slots_k ? 0 : max_of(slots, 2u);
}

/** Rungs per walking kernel: the arena frontier's zero, then 2, 4, 8 and 16 slots. */
inline constexpr std::size_t gpu_rungs_per_kernel_k = 5;

/** The rung of a walk over @p slots slots, as @ref gpu_pool_slots rounds them, in ladder order. */
constexpr std::size_t gpu_rung_index(u32_t slots) noexcept {
    return slots ? static_cast<std::size_t>(std::countr_zero(slots)) : 0;
}

/** One bit per row over device-resident words, marked concurrently by a whole grid. */
struct vram_bitset_t {
    u32_t* words = nullptr;
};

} // namespace unum::usearch::utilities
