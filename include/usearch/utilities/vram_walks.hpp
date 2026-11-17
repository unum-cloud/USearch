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

/** What a health check and a round copy into one page-locked slot, in the device's order: the
 *  check's counters first, which every fresh sweep zeroes, then the round's, which a round's heal
 *  zeroes; and the claim counter of the launch in flight. */
struct vram_round_counters_t {

    /** Entries the last two depths reached and the last two depths or rechecks marked a way to the
     *  root for, by parity; a pass whose predecessor counted zero was queued past convergence. */
    u32_t reached[2], marked[2];

    /** Island labels the last two passes over the islands lowered or marked, by parity. */
    u32_t relabeled[2];

    /** Depths from the root that reached an entry, and rechecks that marked one. */
    u32_t depth_from_root, depth_to_root;

    /** Entries outside the root's component, and those no component claims yet after a settle. */
    u32_t islands, unassigned;

    /** The population counts @c reachability_t reports. */
    u32_t unlinked, underlinked, beyond_most_populated_depth;
    u32_t unreached_islands, stranded_islands, components;

    /** The largest island component's entries above the complement of its least node, whose
     *  maximum names it. */
    u64_t largest_island;

    /** Entries the sweeps reached plus entries they marked a way to the root for. */
    u64_t swept;

    /** Links the heal wrote, rows whose link set changed, and links the round added and dropped. */
    u32_t healed, changed, added_links, dropped_links;

    /** Items or neighborhoods the blocks of the launch in flight claimed so far. */
    u32_t claimed;
};

/** One pass of a health check over the neighbors matrix, the columns the RAM health check keeps,
 *  on the device: depths both ways in bytes, the in-link and shallower in-link counts, the
 *  reached, reaching-root and two-ways bits, the islands' ranked nodes, components, parents,
 *  labels, entries and roles, the rows a round counted as changed, and the counters. */
struct vram_health_t {
    u32_t* neighbors = nullptr;
    u32_t rows_count = 0;
    u32_t connectivity = 0;

    /** The depth, recheck or pass of the launch, whose parity picks its counters. */
    u32_t depth = 0;

    /** Which step of the island passes a launch of @c island_components_k or @c label_islands_k
     *  runs, a @c vram_island_pass_t. */
    u32_t pass = 0;
    u8_t* depths_from_root = nullptr;
    u8_t* depths_to_root = nullptr;
    u32_t* in_links = nullptr;
    u32_t* shallower_in_links = nullptr;

    /** Entries each depth reached, the deepest recorded one saturating. */
    u32_t* depth_entries = nullptr;
    vram_bitset_t reached, reaches_root, two_ways_to_root, changed;
    u32_t* island_nodes = nullptr;
    u32_t* island_components = nullptr;
    u32_t* island_parents = nullptr;

    /** Per island node a color while its components close, then its weak label; one more slot
     *  stands for the root's component. */
    u32_t* island_labels = nullptr;
    u64_t* island_entries = nullptr;
    u32_t* island_roles = nullptr;
    vram_round_counters_t* counters = nullptr;
};

/** One run of a gather: @c width entries from position @c first of the entries in neighborhood
 *  order, or from ID @c first in the first gather, filed among @c hubs hubs from @c first_hub. */
struct vram_neighborhood_run_t {
    u32_t first = 0, width = 0, first_hub = 0, hubs = 0;
};
static_assert(sizeof(vram_neighborhood_run_t) == 16, "A run is one hot sixteen-byte entry");

/** Which step of the island passes a launch takes: the colors running forward along the links,
 *  the marks running back to each color's least rank, the settle that closes the marked
 *  components, the largest one's tally; then the roles' start, the roles, their spread over each
 *  component's members, the entries, the weak labels and their count. */
enum class vram_island_pass_t : u32_t {
    forward_k,
    backward_k,
    settle_k,
    largest_k,
    roles_start_k,
    roles_k,
    roles_spread_k,
    entries_k,
    weak_k,
    weak_count_k,
};

/** Device columns one round carves once from a @p traversal_context_type_ and its phases reuse:
 *  the health check's, then the round's, which alias the islands', since only the checks and the
 *  heal read those, then each resident block's slice of a neighborhood capacity of mates. */
template <typename traversal_context_type_>
struct vram_round_scratch {
    typename traversal_context_type_::bytes_ptr_t bytes;

    /** The check's columns and counters, as @c vram_health_t names them. */
    vram_health_t health;

    /** Each entry's neighborhood, the entries in neighborhood order, the starts with room for
     *  splits, the hubs of every gather run, the runs, and the items before each run. */
    u16_t* entry_neighborhoods = nullptr;
    u32_t* neighborhood_entries = nullptr;
    u32_t* neighborhood_offsets = nullptr;
    u32_t* hub_nodes = nullptr;
    vram_neighborhood_run_t* runs = nullptr;
    u32_t* run_items = nullptr;
    byte_t* mates = nullptr;

    /** Words of one bitset over the entries the scratch was carved for. */
    std::size_t words = 0;
};

} // namespace unum::usearch::utilities
