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

#include "usearch/utilities/types.hpp" // `vram_widest_candidate_slots_k`, `divide_round_up`

namespace unum::usearch::utilities {

/** Bytes of one packed traversal candidate on the device, the kernels' scored node: one word for
 *  @c u32_t node IDs, two for @c u64_t. */
template <typename node_id_type_>
inline constexpr std::size_t gpu_scored_node_bytes = sizeof(node_id_type_) == sizeof(u32_t) ? sizeof(u64_t)
                                                                                            : 2 * sizeof(u64_t);

/** Bytes of one @b {epoch,ID} slot of the device's visited set: one word for @c u32_t node IDs,
 *  two for @c u64_t. */
template <typename node_id_type_>
inline constexpr std::size_t gpu_visited_slot_bytes = sizeof(node_id_type_) == sizeof(u32_t) ? sizeof(u64_t)
                                                                                             : 2 * sizeof(u64_t);

/** Layout of one warp's slice of the traversal arena, computed on the host: the exact visited set,
 *  the wide frontier's spill tail when it has one, then a builder's scratch when shared memory
 *  cannot fit it. The kernels carry it as their own @c vram_arena_shape_t. */
struct gpu_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;
    u32_t scratch_bytes;

    /** A multiple of 16, as the kernels' twin rounds it, for a walk over @p node_id_type_ IDs. */
    template <typename node_id_type_>
    constexpr std::size_t slice_bytes() const noexcept {
        std::size_t const bytes = visited_slots * gpu_visited_slot_bytes<node_id_type_> +
                                  tail_entries * gpu_scored_node_bytes<node_id_type_> + scratch_bytes;
        return (bytes + 15) & ~std::size_t {15};
    }
};

/** The breadth a walk asked for @p expansion walks when it reports @p count matches: never fewer
 *  candidates than it reports. */
constexpr u32_t gpu_walk_breadth(u32_t expansion, u32_t count) noexcept { return max_of(expansion, count); }

/** IDs a converged walk keeping @p candidate_nodes candidates claims: a row of @p connectivity
 *  neighbors or a draw of @p lanes landmarks per expansion, one expansion per rank, then a
 *  re-selecting walk's own node and row. A walk past it re-scores, see
 *  @ref unum::usearch::utilities::vram_visited_set. */
constexpr u32_t gpu_walk_visits(u32_t candidate_nodes, u32_t connectivity, u32_t lanes) noexcept {
    return (candidate_nodes + 1) * max_of(connectivity, lanes) + connectivity + 1;
}

/** Entries an arena frontier's spill tail holds: what @p expansion expansions stage, a row or a
 *  landmark draw each, and a draw per restart, which precedes an expansion. */
constexpr u32_t gpu_walk_tail_entries(u32_t expansion, u32_t connectivity, u32_t lanes) noexcept {
    return expansion * (max_of(connectivity, lanes) + lanes);
}

/** The slice of a walk claiming at most @p visits IDs: the exact visited set, at most half full,
 *  the load its linear probing is designed for, then @p tail_entries of spill tail, rounded to an
 *  even count so the builder scratch past it starts on a 16-byte boundary for either entry width,
 *  then @p scratch_bytes of that scratch. */
constexpr gpu_arena_shape_t gpu_arena_shape(u32_t visits, u32_t tail_entries, u32_t scratch_bytes) noexcept {
    u32_t const visited_slots = std::bit_ceil(2 * visits);
    return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)), (tail_entries + 1) & ~1u,
            scratch_bytes};
}

/** Rounds @p expansion to the register ladder's rungs of 2 to @c vram_widest_candidate_slots_k
 *  slots of @p lanes ranks each, or to zero past them, where the arena frontier walks the expansion
 *  itself; @p default_slots when it asks for none. */
constexpr u32_t gpu_candidate_slots(u32_t expansion, u32_t default_slots, u32_t lanes = 32) noexcept {
    if (!expansion) return default_slots;
    u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion, lanes));
    return slots > vram_widest_candidate_slots_k ? 0 : max_of(slots, 2u);
}

/** Rungs per walking kernel: the arena frontier's zero, then 2, 4, 8 and 16 slots. */
inline constexpr std::size_t gpu_rungs_per_kernel_k = 5;

/** The rung of a walk over @p slots slots, as @ref gpu_candidate_slots rounds them, in ladder
 *  order. */
constexpr std::size_t gpu_rung_index(u32_t slots) noexcept {
    return slots ? static_cast<std::size_t>(std::countr_zero(slots)) : 0;
}

/** One bit per node over device-resident words, written concurrently by a whole grid. */
struct vram_bitset_t {
    u32_t* words = nullptr;
};

} // namespace unum::usearch::utilities
