/**
 *  @file include/usearch/basic/vram_apple_index.metal
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Metal walks, insertion, re-selection, back-link merge, search, round and graph
 *      kernels of the basic Apple index - the MSL port of @c vram_index.cuh over any basic Metal
 *      kit - and the launch words its host binds them by.
 *
 *  The words lead, in the C++ that MSL shares with every host compiler: the launch scalars and
 *  buffer slots of the kit kernels and of the graph kernels, the round's counter words, the depths
 *  a health check records and the roles of its islands. The host includes this file for them alone,
 *  as Metal defines @c __METAL_VERSION__ only on the device side, and embeds it whole last, so the
 *  two sides read one definition of every word.
 *
 *  One 32-wide simdgroup walks one query or inserts one node over the shared walk state of
 *  @c vram_queue.metal, its visited set, landmark draw and expansion round, as the dense Apple walk
 *  does, and scores whole entries through the kit the host's prelude names as @c metal_kit_t. A kit
 *  whose lane scores an entry alone walks a query per simdgroup; a kit splitting an entry over
 *  several lanes walks one per threadgroup, simdgroup zero owning the frontier and the visited set
 *  and posting each round's entries in threadgroup memory for all four simdgroups to score
 *  together. Every scratch span lives in the walker's device slice.
 *
 *  Insertion publishes forward rows alone and threads its picks onto per-node chains, which the
 *  back-link launch behind each wave merges, re-selecting a full row over the union; a wave's nodes
 *  walk again behind those back-links and re-select. A round's gather files every entry under its
 *  nearest hub, and its restitch keeps the diverse links among each entry's own and its nearest
 *  neighborhood-mates, the health check's reach rule on top. The graph kernels, a cut's partition
 *  growth, the health check, the island components and the heal, read the neighbors matrix alone,
 *  as the CUDA basic passes do. Loads copy a run's bytes as they were saved. The host splices, in
 *  order, the contexts' source, the face's kits, and its prelude naming the kit and defining
 *  @c USEARCH_METAL_KIT_RUNGS, ahead of this file.
 */
#pragma once

namespace unum::usearch::basic {

using namespace unum::usearch::utilities;

#pragma region Launch Words

/** The buffer slots a kit's kernels take their bindings at: the first seven are common to every
 *  kernel reading values, and the slots of the round's columns are the graph kernels' ones, since a
 *  gather and a restitch bind the health check's columns beside them. An insert, a re-selection and
 *  its back-link bind the chains they thread; a search its query starts and its outputs; a load its
 *  run's saved values, their element sums and planned starts and the widest span it counts. The
 *  kit's parameters bind past every round column. */
enum metal_basic_buffer_index_t : unsigned int {
    metal_basic_arguments_buffer_k = 0,
    metal_basic_values_buffer_k = 1,
    metal_basic_offsets_buffer_k = 2,
    metal_basic_neighbors_buffer_k = 3,
    metal_basic_staging_buffer_k = 4,
    metal_basic_arena_buffer_k = 5,
    metal_basic_counters_buffer_k = 6,

    metal_basic_link_heads_buffer_k = 7,
    metal_basic_link_next_buffer_k = 8,
    metal_basic_touched_ids_buffer_k = 9,
    metal_basic_touched_count_buffer_k = 10,

    metal_basic_query_offsets_buffer_k = 7,
    metal_basic_ids_buffer_k = 8,
    metal_basic_distances_buffer_k = 9,

    metal_basic_saved_buffer_k = 7,
    metal_basic_saved_offsets_buffer_k = 8,
    metal_basic_stored_offsets_buffer_k = 9,
    metal_basic_span_words_buffer_k = 10,

    metal_basic_round_counters_buffer_k = 7,
    metal_basic_changed_buffer_k = 15,
    metal_basic_entry_neighborhoods_buffer_k = 16,
    metal_basic_neighborhood_entries_buffer_k = 17,
    metal_basic_neighborhood_offsets_buffer_k = 18,
    metal_basic_hub_nodes_buffer_k = 19,
    metal_basic_runs_buffer_k = 20,
    metal_basic_run_items_buffer_k = 21,

    metal_basic_parameters_buffer_k = 23,
};

/** Every scalar a kit's launch passes, bound at @c metal_basic_arguments_buffer_k, the CUDA
 *  records' scalars field by field: the storage's connectivity and the raw @c nk_kernel_kind_t its
 *  kits score under, a wave's or a walk's bounds and spans, the arena slice and its epochs, a
 *  search's breadth and outputs, a load's run, and the runs, the neighborhoods and the gather pass
 *  of a round. */
struct metal_basic_arguments_t {
    unsigned int connectivity, metric, first_id, end_id;
    unsigned int sampling_bound, span_bytes, scratch_bytes_per_warp, selection_nodes;
    unsigned int expansion, epoch_base, visited_slots, hash_shift;
    unsigned int tail_entries, slice_bytes, concurrent_walks, filling_nodes;
    unsigned int queries_count, count, runs_count, neighborhoods;
    unsigned int pass;
};

/** Every scalar a graph kernel reads, bound at @c metal_graph_arguments_buffer_k: the neighbors
 *  matrix's connectivity and its visible nodes, a health check's depth, its islands and its pass,
 *  and a cut's part capacity and growth round. */
struct metal_graph_arguments_t {
    unsigned int connectivity, end_id, depth, islands;
    unsigned int pass, part_capacity, round;
};

/** The buffer slots the graph kernels take their bindings at: the scalars and the neighbors matrix,
 *  then a split's partition its node parts, proposals, two frontiers, the parts' sizes and round
 *  counts and the placed counts, or a health check and a round the round's counters and the check's
 *  columns, then the columns of the islands. */
enum metal_graph_buffer_index_t : unsigned int {
    metal_graph_arguments_buffer_k = 0,
    metal_graph_neighbors_buffer_k = 3,

    metal_node_parts_buffer_k = 7,
    metal_proposals_buffer_k = 8,
    metal_frontier_buffer_k = 9,
    metal_next_frontier_buffer_k = 10,
    metal_part_sizes_buffer_k = 11,
    metal_round_counts_buffer_k = 12,
    metal_placed_buffer_k = 13,

    metal_round_counters_buffer_k = 7,
    metal_depths_from_root_buffer_k = 8,
    metal_depths_to_root_buffer_k = 9,
    metal_in_links_buffer_k = 10,
    metal_shallower_in_links_buffer_k = 11,
    metal_reached_buffer_k = 12,
    metal_reaches_root_buffer_k = 13,
    metal_two_ways_to_root_buffer_k = 14,
    metal_changed_buffer_k = 15,

    metal_island_nodes_buffer_k = 16,
    metal_island_components_buffer_k = 17,
    metal_island_labels_buffer_k = 18,
    metal_island_parents_buffer_k = 19,
    metal_island_entries_buffer_k = 20,
    metal_island_roles_buffer_k = 21,
};

/** The words of a health check's and a round's counters, which the host zeroes before the pass that
 *  counts into them: the entries a depth reached and marked; the island labels a components pass
 *  lowered or settled; the population counts a check reports; the islands it ordered; the links the
 *  heal wrote; the rows whose link set changed; the next item or neighborhood a threadgroup claims;
 *  and the links a restitch added and dropped, each carrying into its high word. */
enum metal_round_counter_t : unsigned int {
    metal_round_reached_k = 0,
    metal_round_marked_k = 1,
    metal_round_relabeled_k = 2,
    metal_round_unlinked_k = 3,
    metal_round_underlinked_k = 4,
    metal_round_unreached_islands_k = 5,
    metal_round_stranded_islands_k = 6,
    metal_round_components_k = 7,
    metal_round_islands_k = 8,
    metal_round_healed_k = 9,
    metal_round_changed_k = 10,
    metal_round_claimed_k = 11,
    metal_round_added_k = 12,
    metal_round_dropped_k = 14,
    metal_round_counter_words_k = 16,
};

/** The depths a health check records in a byte: 255 for an entry no sweep reached, and the deepest
 *  one it records, past which depths saturate, as on RAM. */
enum metal_depth_t : unsigned int {
    metal_unknown_depth_k = 255,
    metal_deepest_depth_k = 254,
};

/** An island's roles, the bits of its word, the first two its strong component's, as on RAM. */
enum metal_island_role_t : unsigned int {
    metal_unreached_island_k = 1,
    metal_stranded_island_k = 2,
    metal_links_to_component_k = 4,
    metal_linked_from_component_k = 8,
};

#pragma endregion Launch Words

} // namespace unum::usearch::basic

#if defined(__METAL_VERSION__)

namespace unum::usearch::basic {

#pragma region Partitions

/** Whether bit @p position of @p words is set. */
inline bool metal_bitset_contains(device uint const* words, uint position) {
    return (words[position / 32] >> (position % 32)) & 1u;
}

/** Sets bit @p position of @p words, which other lanes set beside it. */
inline void metal_bitset_insert(device uint* words, uint position) {
    atomic_fetch_or_explicit((device atomic_uint*)(words + position / 32), 1u << (position % 32), memory_order_relaxed);
}

/** Whether the round before the bound one placed nothing, which ends the growth: a round queued
 *  past convergence leaves at entry, as every later pass of it does; a first round, having none
 *  before it, never ends the growth. */
inline bool metal_converged(constant metal_graph_arguments_t& arguments, device uint const* placed) {
    return arguments.round > 1 && placed[(arguments.round & 1u) ^ 1u] == 0;
}

/** A growth round's first pass, one simdgroup per frontier row: a row of a part with room proposes
 *  itself to every unplaced row it links to, and each keeps its lowest-ID proposer, so no
 *  simdgroup order sways the round, as the CUDA pass does. */
[[host_name("metal_partition_propose_kernel")]] kernel void metal_partition_propose_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint const* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint const* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device atomic_uint* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint const* frontier [[buffer(metal_frontier_buffer_k)]],
    device uint const* part_sizes [[buffer(metal_part_sizes_buffer_k)]],
    device uint const* placed [[buffer(metal_placed_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (metal_converged(arguments, placed)) return;
    uint const slices = groups * metal_simdgroups_k, nodes = arguments.end_id;
    for (uint node = group * metal_simdgroups_k + simdgroup; node < nodes; node += slices) {
        if (!metal_bitset_contains(frontier, node) || part_sizes[node_parts[node]] >= arguments.part_capacity) continue;
        device uint const* const links = metal_row(neighbors, node, arguments.connectivity);
        for (uint slot = lane; slot < arguments.connectivity; slot += metal_simdgroup_width_k) {
            uint const target = links[slot];
            if (target == metal_unknown_k || target == node || target >= nodes || node_parts[target] != metal_unknown_k)
                continue;
            uint held = atomic_load_explicit(proposals + target, memory_order_relaxed);
            while (node < held && !atomic_compare_exchange_weak_explicit(proposals + target, &held, node,
                                                                         memory_order_relaxed, memory_order_relaxed)) {}
        }
    }
}

/** A growth round's second pass, one thread per row: counts the rows proposed to each part, and
 *  empties the next frontier and the round's placed word for the passes after it, since the host
 *  writes nothing between the round's command buffers. */
[[host_name("metal_partition_count_kernel")]] kernel void metal_partition_count_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint const* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device uint const* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint* next_frontier [[buffer(metal_next_frontier_buffer_k)]],
    device atomic_uint* round_counts [[buffer(metal_round_counts_buffer_k)]],
    device uint* placed [[buffer(metal_placed_buffer_k)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    if (metal_converged(arguments, placed)) return;
    for (uint word = position; word < divide_round_up(arguments.end_id, 32u); word += threads) next_frontier[word] = 0;
    if (position == 0) placed[arguments.round & 1u] = 0;
    for (uint node = position; node < arguments.end_id; node += threads) {
        uint const proposal = proposals[node];
        if (proposal != metal_unknown_k)
            atomic_fetch_add_explicit(round_counts + node_parts[proposal], 1u, memory_order_relaxed);
    }
}

/** A growth round's third pass, one thread per row: places every row proposed to a part with room
 *  for the round's whole count in the next frontier, counting them in the round's placed word; the
 *  rows a round count would overflow stay proposed for the fill. Proposers were placed in earlier
 *  rounds, so no thread rewrites a part another one reads. */
[[host_name("metal_partition_accept_kernel")]] kernel void metal_partition_accept_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device uint* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint* next_frontier [[buffer(metal_next_frontier_buffer_k)]],
    device uint const* part_sizes [[buffer(metal_part_sizes_buffer_k)]],
    device uint const* round_counts [[buffer(metal_round_counts_buffer_k)]],
    device uint* placed [[buffer(metal_placed_buffer_k)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    if (metal_converged(arguments, placed)) return;
    for (uint node = position; node < arguments.end_id; node += threads) {
        uint const proposal = proposals[node];
        if (proposal == metal_unknown_k) continue;
        uint const part = node_parts[proposal];
        if (round_counts[part] > arguments.part_capacity - part_sizes[part]) continue;
        node_parts[node] = part;
        proposals[node] = metal_unknown_k;
        metal_bitset_insert(next_frontier, node);
        atomic_fetch_add_explicit((device atomic_uint*)(placed + (arguments.round & 1u)), 1u, memory_order_relaxed);
    }
}

/** A growth round's last pass, one threadgroup per part: a part whose round count fit takes it into
 *  its size, and one the round count overflows places its lowest-ID proposed rows up to its room in
 *  the next frontier, and fills up. Either way the part's proposals and round count clear for the
 *  round that follows. */
[[host_name("metal_partition_fill_kernel")]] kernel void metal_partition_fill_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* node_parts [[buffer(metal_node_parts_buffer_k)]],
    device uint* proposals [[buffer(metal_proposals_buffer_k)]],
    device uint* next_frontier [[buffer(metal_next_frontier_buffer_k)]],
    device uint* part_sizes [[buffer(metal_part_sizes_buffer_k)]],
    device uint* round_counts [[buffer(metal_round_counts_buffer_k)]],
    device uint* placed [[buffer(metal_placed_buffer_k)]], uint part [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]], uint group_threads [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simdgroup_counts[metal_simdgroups_k];
    if (metal_converged(arguments, placed)) return;
    uint const wanted = round_counts[part], room = arguments.part_capacity - part_sizes[part];
    threadgroup_barrier(mem_flags::mem_device);
    for (uint first = 0, taken = 0; wanted > room && first < arguments.end_id; first += group_threads) {
        uint const node = first + thread_index;
        uint const proposal = node < arguments.end_id ? proposals[node] : metal_unknown_k;
        bool const proposed_here = proposal != metal_unknown_k && node_parts[proposal] == part;
        uint const voted = metal_warp_t::ballot(proposed_here);
        if (lane == 0) simdgroup_counts[simdgroup] = popcount(voted);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint rank = taken + popcount(voted & metal_warp_t::below(lane)), total = 0;
        for (uint other = 0; other != metal_simdgroups_k; ++other) {
            if (other < simdgroup) rank += simdgroup_counts[other];
            total += simdgroup_counts[other];
        }
        if (proposed_here && rank < room) {
            node_parts[node] = part;
            metal_bitset_insert(next_frontier, node);
        }
        if (proposed_here) proposals[node] = metal_unknown_k;
        taken += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index != 0) return;
    part_sizes[part] += wanted > room ? room : wanted;
    round_counts[part] = 0;
    if (wanted > room)
        atomic_fetch_add_explicit((device atomic_uint*)(placed + (arguments.round & 1u)), room, memory_order_relaxed);
}

#pragma endregion Partitions

#pragma region Health

/** Whether bit @p position of @p words is set, read through the word other lanes set bits in. */
inline bool metal_bit(device uint* words, uint position) {
    return (metal_warp_t::load_relaxed(words + position / 32) >> (position % 32)) & 1u;
}

/** Sets bit @p position of @p words, answering whether it was set already. */
inline bool metal_bitset_claim(device uint* words, uint position) {
    uint const bit = 1u << (position % 32);
    return (metal_warp_t::fetch_or_relaxed(words + position / 32, bit) & bit) != 0;
}

/** Clears bit @p position of @p words, answering whether it was set. */
inline bool metal_bitset_release(device uint* words, uint position) {
    uint const bit = 1u << (position % 32);
    return (metal_warp_t::fetch_and_relaxed(words + position / 32, ~bit) & bit) != 0;
}

/** Adds @p value to the tally at @p counter, carrying into its high word. */
inline void metal_tally_add(device uint* counters, uint counter, uint value) {
    if (!value) return;
    uint const low = metal_warp_t::fetch_add_relaxed(counters + counter, value);
    if (low + value < low) metal_warp_t::fetch_add_relaxed(counters + counter + 1, 1u);
}

/** The columns of one health check, as the RAM audit names them, beside the round's counters:
 *  depths both ways a byte each, the exact in-link and shallower in-link counts, and the reached,
 *  reaching-root and two-ways bits. */
struct metal_health_t {
    device uint* neighbors;
    device uint* counters;
    device uchar* depths_from_root;
    device uchar* depths_to_root;
    device uint* in_links;
    device uint* shallower_in_links;
    device uint* reached;
    device uint* reaches_root;
    device uint* two_ways_to_root;
    uint rows_count, connectivity;

    device uint* links_of(uint node) { return metal_row(neighbors, node, connectivity); }
    uint depth_from_root(uint node) { return metal_warp_t::load_relaxed(depths_from_root, node); }
    uint depth_to_root(uint node) { return metal_warp_t::load_relaxed(depths_to_root, node); }
    bool in_component(uint node) { return metal_bit(reached, node) && metal_bit(reaches_root, node); }

    /** The in-links of @p node read saturated at 255, so every decision matches the RAM audit's
     *  byte. */
    uint in_links_of(uint node) { return min(metal_warp_t::load_relaxed(in_links + node), 255u); }

    /** Marks @p node 's way to the root one past @p lowest, and its two ways when two links met it;
     *  answers whether it marked. */
    uint mark_reaches_root(uint node, uint lowest, uint at_lowest) {
        if (lowest == metal_unknown_depth_k) return 0;
        uint const depth = min(lowest + 1u, uint(metal_deepest_depth_k));
        depths_to_root[node] = uchar(depth);
        metal_bitset_claim(reaches_root, node);
        if (at_lowest > 1 && lowest < depth) metal_bitset_claim(two_ways_to_root, node);
        return 1;
    }

    /** Takes one of @p node 's links from the depth before it, answering whether another
     *  remains. */
    bool take_shallower_in_link_if_spare(uint node) {
        uint observed = metal_warp_t::load_relaxed(shallower_in_links + node);
        while (observed > 1 &&
               !metal_warp_t::compare_exchange_relaxed(shallower_in_links + node, observed, observed - 1))
            observed = metal_warp_t::load_relaxed(shallower_in_links + node);
        return observed > 1;
    }

    /** Takes one of @p node 's in-links for a swap, answering whether another remains. */
    bool take_in_link_if_spare(uint node) {
        uint observed = metal_warp_t::load_relaxed(in_links + node);
        while (observed && !metal_warp_t::compare_exchange_relaxed(in_links + node, observed, observed - 1))
            observed = metal_warp_t::load_relaxed(in_links + node);
        return observed > 1;
    }

    /** Takes the link from @p owner to @p resident for a swap: one not from the depth before the
     *  resident goes freely, a shallower in-link only while another remains. */
    bool take_link_for_swap_if_spare(uint owner, uint resident) {
        uint const owner_depth = depth_from_root(owner);
        uint const child = owner_depth < metal_deepest_depth_k ? owner_depth + 1 : uint(metal_unknown_depth_k);
        return child == metal_unknown_depth_k || depth_from_root(resident) != child ||
               take_shallower_in_link_if_spare(resident);
    }

    /** Takes that link when the reach rule releases it: never the resident's last shallower
     *  in-link, and a link toward the root only while the owner keeps another as close to it. */
    bool take_link_under_lock_if_spare(uint owner, uint resident) {
        bool const way_to_root = metal_bit(reaches_root, owner) && depth_to_root(owner) != 0 &&
                                 depth_to_root(resident) < depth_to_root(owner);
        if (way_to_root && !metal_bit(two_ways_to_root, owner)) return false;
        if (!take_link_for_swap_if_spare(owner, resident)) return false;
        if (way_to_root) metal_bitset_release(two_ways_to_root, owner);
        return true;
    }

    /** Files a heal's link from @p owner to island entry @p member in the columns the reach rule
     *  keeps, as the RAM audit's @c record_heal_link does. */
    void record_heal_link(uint owner, uint member) {
        uint const owner_depth = depth_from_root(owner);
        if (depth_from_root(member) == metal_unknown_depth_k && owner_depth < metal_deepest_depth_k) {
            depths_from_root[member] = uchar(owner_depth + 1);
            shallower_in_links[member] = 1;
        }
        uint const member_depth_to_root = depth_to_root(member);
        if (depth_to_root(owner) == metal_unknown_depth_k && member_depth_to_root < metal_deepest_depth_k)
            depths_to_root[owner] = uchar(member_depth_to_root + 1);
    }
};

/** The lowest depth to root among the links one simdgroup read, and how many links met it. */
struct metal_way_to_root_t {
    uint lowest = metal_unknown_depth_k, at_lowest = 0;

    void note(uint depth) {
        if (depth < lowest) lowest = depth, at_lowest = 1;
        else at_lowest += depth == lowest && depth != metal_unknown_depth_k;
    }

    /** The lowest depth every lane met, and the links meeting it, on every lane. */
    metal_way_to_root_t reduced() const {
        uint const least = metal_warp_t::reduce_min(lowest);
        metal_way_to_root_t way;
        way.lowest = least, way.at_lowest = metal_warp_t::reduce_add(lowest == least ? at_lowest : 0u);
        return way;
    }
};

/** Calls @p visit on the lanes holding a link of @p node 's row below the rows but its own, the
 *  links past the first sentinel ignored; a row whose slot zero is the sentinel or the node itself
 *  holds none. Every lane calls it; @p visit runs on some. */
template <typename visit_type_>
inline void metal_for_each_link(thread metal_health_t& health, uint node, uint lane, visit_type_ visit) {
    device uint* const links = health.links_of(node);
    uint const first = metal_warp_t::shuffle(lane == 0 ? metal_warp_t::load_acquire(links) : 0u, 0);
    if (first == metal_unknown_k || first == node) return;
    for (uint base = 0; base < health.connectivity; base += metal_warp_t::width_k) {
        uint const slot = base + lane;
        uint const target = slot >= health.connectivity ? metal_unknown_k
                            : slot                      ? metal_warp_t::load_relaxed(links + slot)
                                                        : first;
        uint const ends = metal_warp_t::ballot(target == metal_unknown_k);
        bool const listed = !ends || lane < ctz(ends);
        if (listed && target < health.rows_count && target != node) visit(target);
        if (ends) return;
    }
}

/** The rank of @p node among the @p islands ascending island nodes, or @p islands when none. */
inline uint metal_island_rank(device uint* island_nodes, uint islands, uint node) {
    uint low = 0, high = islands;
    while (low < high) {
        uint const middle = (low + high) / 2;
        if (island_nodes[middle] < node) low = middle + 1;
        else high = middle;
    }
    return low < islands && island_nodes[low] == node ? low : islands;
}

/** Calls @p visit with the rank of every island node that island node @p node links to, the islands
 *  as ordered, so a way to the root marked meanwhile changes no component's links. */
template <typename visit_type_>
inline void metal_for_each_island_link(thread metal_health_t& health, device uint* island_nodes, uint islands,
                                       uint node, uint lane, visit_type_ visit) {
    metal_for_each_link(health, node, lane, [&](uint target) {
        uint const rank = metal_island_rank(island_nodes, islands, target);
        if (rank != islands) visit(rank);
    });
}

/** Files @p key of island member @p rank under its @p component in two 32-bit minimums, the key's
 *  in the high word on the first pass and the least rank holding it in the low word on the second:
 *  64-bit atomic minimums exist only from Apple GPU family 9, and one source serves them all. */
inline void metal_claim_entry(device uint* entries, uint component, uint key) {
    metal_warp_t::fetch_min_relaxed(entries + 2 * component + 1, key);
}
inline void metal_settle_entry(device uint* entries, uint component, uint key, uint rank) {
    if (entries[2 * component + 1] == key) metal_warp_t::fetch_min_relaxed(entries + 2 * component, rank);
}

/** The key an island member stands for its component by, the least first: a member a link crosses
 *  through, then the fewest in-links, ties by a keyed draw, as the RAM audit orders them. */
inline uint metal_entry_key(thread metal_health_t& health, uint node, uint parent, uint roles) {
    bool const crosses = parent != metal_unknown_k || (roles & metal_links_to_component_k) != 0;
    return (crosses ? 0u : 1u << 31) | health.in_links_of(node) << 23 | uint(metal_mix_bits(node) >> 41);
}

/** The root of @p rank 's set among the union-find @p parents, halving its path. */
inline uint metal_find(device uint* parents, uint rank) {
    for (;;) {
        uint const parent = metal_warp_t::load_relaxed(parents + rank);
        if (parent == rank) return rank;
        uint const grand = metal_warp_t::load_relaxed(parents + parent);
        if (grand != parent) metal_warp_t::compare_exchange_relaxed(parents + rank, parent, grand);
        rank = grand;
    }
}

/** Joins the sets of @p first and @p second, the higher root hooked under the lower one. */
inline void metal_unite(device uint* parents, uint first, uint second) {
    for (;;) {
        first = metal_find(parents, first), second = metal_find(parents, second);
        if (first == second) return;
        if (metal_warp_t::compare_exchange_relaxed(parents + max(first, second), max(first, second),
                                                   min(first, second)))
            return;
    }
}

/** The health columns bound at their slots, every check's kernel taking them alike. */
inline metal_health_t metal_health(uint rows_count, uint connectivity, device uint* neighbors, device uint* counters,
                                   device uchar* depths_from_root, device uchar* depths_to_root, device uint* in_links,
                                   device uint* shallower_in_links, device uint* reached, device uint* reaches_root,
                                   device uint* two_ways_to_root) {
    metal_health_t health;
    health.neighbors = neighbors, health.counters = counters;
    health.depths_from_root = depths_from_root, health.depths_to_root = depths_to_root;
    health.in_links = in_links, health.shallower_in_links = shallower_in_links;
    health.reached = reached, health.reaches_root = reaches_root, health.two_ways_to_root = two_ways_to_root;
    health.rows_count = rows_count, health.connectivity = connectivity;
    return health;
}

/** One depth of a health check from its root, one simdgroup an entry: every entry at @c depth
 *  counts an in-link into each target, reaches the targets no entry reached before one depth
 *  deeper, counts a shallower in-link into every target at that depth or still being stored, and
 *  marks its own way to the root in the same read, as the RAM audit's @c expand_ does. */
[[host_name("metal_health_expand_kernel")]] kernel void metal_health_expand_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    uint const depth = arguments.depth;
    bool const saturated = depth >= metal_deepest_depth_k;
    uint const own = min(depth, uint(metal_deepest_depth_k)), child = min(depth + 1u, uint(metal_deepest_depth_k));
    uint reached_count = 0, marked_count = 0;
    for (uint node = group * metal_simdgroups_k + simdgroup; node < health.rows_count;
         node += groups * metal_simdgroups_k) {
        bool expands = lane == 0 && metal_bit(reached, node) && health.depth_from_root(node) == own;
        // Past the deepest recorded depth an entry marks its expansion in its shallower in-links.
        if (expands && saturated) expands = !metal_warp_t::exchange_relaxed(shallower_in_links + node, 1u);
        if (!metal_warp_t::shuffle(uint(expands), 0)) continue;
        metal_way_to_root_t way;
        metal_for_each_link(health, node, lane, [&](uint target) {
            if (!metal_warp_t::fetch_add_relaxed(in_links + target, 1u) && !metal_bitset_claim(reached, target)) {
                depths_from_root[target] = uchar(child);
                ++reached_count;
            }
            // A depth still unknown is one another simdgroup reached this depth and is storing.
            uint const target_depth = health.depth_from_root(target);
            if (child != metal_deepest_depth_k && (target_depth == child || target_depth == metal_unknown_depth_k))
                metal_warp_t::fetch_add_relaxed(shallower_in_links + target, 1u);
            way.note(health.depth_to_root(target));
        });
        way = way.reduced();
        if (lane == 0 && health.depth_to_root(node) == metal_unknown_depth_k)
            marked_count += health.mark_reaches_root(node, way.lowest, way.at_lowest);
    }
    reached_count = metal_warp_t::reduce_add(reached_count);
    if (lane == 0 && reached_count) metal_warp_t::fetch_add_relaxed(counters + metal_round_reached_k, reached_count);
    if (lane == 0 && marked_count) metal_warp_t::fetch_add_relaxed(counters + metal_round_marked_k, marked_count);
}

/** Re-reads the reached entries not yet marked, one simdgroup an entry, marking each that now links
 *  to an entry reaching the root, as the RAM audit's @c recheck_reaches_root_ does. */
[[host_name("metal_health_recheck_kernel")]] kernel void metal_health_recheck_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    uint marked_count = 0;
    for (uint node = group * metal_simdgroups_k + simdgroup; node < health.rows_count;
         node += groups * metal_simdgroups_k) {
        bool const rechecks = lane == 0 && metal_bit(reached, node) && !metal_bit(reaches_root, node);
        if (!metal_warp_t::shuffle(uint(rechecks), 0)) continue;
        metal_way_to_root_t way;
        metal_for_each_link(health, node, lane, [&](uint target) { way.note(health.depth_to_root(target)); });
        way = way.reduced();
        if (lane == 0) marked_count += health.mark_reaches_root(node, way.lowest, way.at_lowest);
    }
    if (lane == 0 && marked_count) metal_warp_t::fetch_add_relaxed(counters + metal_round_marked_k, marked_count);
}

/** Writes the entries outside the root's component in ascending order on one threadgroup, a word of
 *  entries a thread and a threadgroup's words at a time, so the ranks match the RAM audit's. */
[[host_name("metal_order_islands_kernel")]] kernel void metal_order_islands_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    uint thread_index [[thread_position_in_threadgroup]], uint group_threads [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint simdgroup_counts[metal_simdgroups_k];
    uint const nodes = arguments.end_id, words = divide_round_up(nodes, 32);
    uint written = 0;
    for (uint first = 0; first < words; first += group_threads) {
        uint const word = first + thread_index;
        uint lanes = 0;
        if (word < words) {
            uint const visible = nodes - word * 32;
            lanes = ~(reached[word] & reaches_root[word]) & (visible >= 32 ? ~0u : (1u << visible) - 1u);
        }
        uint const count = popcount(lanes), before = simd_prefix_exclusive_sum(count);
        if (lane == metal_warp_t::width_k - 1) simdgroup_counts[simdgroup] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint offset = written + before, total = 0;
        for (uint other = 0; other != metal_simdgroups_k; ++other) {
            if (other < simdgroup) offset += simdgroup_counts[other];
            total += simdgroup_counts[other];
        }
        for (; lanes; lanes &= lanes - 1) island_nodes[offset++] = word * 32 + ctz(lanes);
        written += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0) counters[metal_round_islands_k] = written;
}

/** Opens every island's columns, one simdgroup an island: marks those linking into the root's
 *  component and their ways to the root, and starts its labels from its own rank, as the RAM
 *  audit's @c gather_islands_ does. */
[[host_name("metal_health_islands_kernel")]] kernel void metal_health_islands_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_labels [[buffer(metal_island_labels_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]],
    device uint* island_roles [[buffer(metal_island_roles_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    for (uint rank = group * metal_simdgroups_k + simdgroup; rank < arguments.islands;
         rank += groups * metal_simdgroups_k) {
        uint const node = island_nodes[rank];
        bool links = false;
        metal_way_to_root_t way;
        metal_for_each_link(health, node, lane, [&](uint target) {
            if (!health.in_component(target)) return;
            links = true;
            way.note(health.depth_to_root(target));
        });
        links = simd_any(links);
        way = way.reduced();
        if (lane != 0) continue;
        island_roles[rank] = links ? uint(metal_links_to_component_k) : 0u;
        island_parents[rank] = metal_unknown_k;
        island_components[rank] = metal_unknown_k;
        island_labels[2 * rank] = island_labels[2 * rank + 1] = rank;
        // The members a component counts while labeled, until the labels pass files its entry here.
        island_entries[2 * rank] = island_entries[2 * rank + 1] = 0;
        // A reached island never links into the component, or a recheck marked it.
        health.mark_reaches_root(node, way.lowest, way.at_lowest);
    }
}

/** One pass of the island components, the strong components among the islands, each named by its
 *  least rank as the RAM audit's Tarjan walk names it. Pass zero lowers, one simdgroup an island,
 *  the forward label of every unlabeled island it links to and its own backward label to theirs,
 *  and marks its way to the root through the islands it links to; pass one settles, a thread an
 *  island, every one whose labels meet, as both name the least rank it reaches and is reached
 *  from, counting its members, and restarts the rest from their own ranks. */
[[host_name("metal_island_components_kernel")]] kernel void metal_island_components_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_labels [[buffer(metal_island_labels_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    uint const islands = arguments.islands;
    uint changed = 0;
    if (arguments.pass == 1) {
        for (uint rank = position; rank < islands; rank += threads) {
            if (island_components[rank] != metal_unknown_k) continue;
            uint const forward = island_labels[2 * rank], backward = island_labels[2 * rank + 1];
            if (forward == backward) {
                island_components[rank] = forward;
                metal_warp_t::fetch_add_relaxed(island_entries + 2 * forward, 1u);
                ++changed;
            }
            else island_labels[2 * rank] = island_labels[2 * rank + 1] = rank;
        }
    }
    else
        for (uint rank = group * metal_simdgroups_k + simdgroup; rank < islands; rank += groups * metal_simdgroups_k) {
            uint const node = island_nodes[rank];
            bool const open = island_components[rank] == metal_unknown_k;
            bool const unmarked = metal_warp_t::shuffle(
                uint(lane == 0 && health.depth_to_root(node) == metal_unknown_depth_k), 0);
            if (!open && !unmarked) continue;
            uint const forward = open ? metal_warp_t::load_relaxed(island_labels + 2 * rank) : 0u;
            uint lowered = 0;
            metal_way_to_root_t way;
            metal_for_each_island_link(health, island_nodes, islands, node, lane, [&](uint next) {
                if (unmarked) way.note(health.depth_to_root(island_nodes[next]));
                if (!open || island_components[next] != metal_unknown_k) return;
                lowered += metal_warp_t::fetch_min_relaxed(island_labels + 2 * next, forward) > forward;
                uint const backward = metal_warp_t::load_relaxed(island_labels + 2 * next + 1);
                lowered += metal_warp_t::fetch_min_relaxed(island_labels + 2 * rank + 1, backward) > backward;
            });
            way = way.reduced();
            lowered = metal_warp_t::reduce_add(lowered);
            if (lane == 0 && unmarked) lowered += health.mark_reaches_root(node, way.lowest, way.at_lowest);
            if (lane == 0) changed += lowered;
        }
    changed = metal_warp_t::reduce_add(changed);
    if (lane == 0 && changed) metal_warp_t::fetch_add_relaxed(counters + metal_round_relabeled_k, changed);
}

/** Counts the entries no reached entry links to and those linked to by fewer entries than they link
 *  to, one simdgroup an entry, and files the least entry of the root's component linking to each
 *  island as its parent, as the RAM audit's @c count_in_and_out_links_ does. */
[[host_name("metal_health_count_links_kernel")]] kernel void metal_health_count_links_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    uint unlinked = 0, underlinked = 0;
    for (uint node = group * metal_simdgroups_k + simdgroup; node < health.rows_count;
         node += groups * metal_simdgroups_k) {
        bool const member = metal_warp_t::shuffle(uint(lane == 0 && health.in_component(node)), 0);
        uint out_links = 0;
        metal_for_each_link(health, node, lane, [&](uint target) {
            ++out_links;
            if (!member || health.in_component(target)) return;
            uint const rank = metal_island_rank(island_nodes, arguments.islands, target);
            if (rank != arguments.islands) metal_warp_t::fetch_min_relaxed(island_parents + rank, node);
        });
        out_links = metal_warp_t::reduce_add(out_links);
        uint const linked_by = health.in_links_of(node);
        unlinked += linked_by == 0, underlinked += linked_by < out_links;
    }
    if (lane != 0) return;
    if (unlinked) metal_warp_t::fetch_add_relaxed(counters + metal_round_unlinked_k, unlinked);
    if (underlinked) metal_warp_t::fetch_add_relaxed(counters + metal_round_underlinked_k, underlinked);
}

/** One pass of labeling the islands, as the RAM audit's @c label_islands_ does. Pass zero, a thread
 *  an island, marks every component first unreached and stranded, empties the entries and starts
 *  the weak components' union-find, the root's component in the last slot. Pass one, a simdgroup an
 *  island, clears the roles its links disprove, files its entry key and joins the weak components
 *  its links join. Pass two, a thread an island, settles the entries, spreads each component's
 *  roles to its members and counts the unreached and stranded ones. Pass three counts the weak
 *  components, the union-find's roots. */
[[host_name("metal_label_islands_kernel")]] kernel void metal_label_islands_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_labels [[buffer(metal_island_labels_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]],
    device uint* island_roles [[buffer(metal_island_roles_buffer_k)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    uint const islands = arguments.islands;
    // The forward and backward labels served the components; the union-find takes their slots.
    device uint* const weak = island_labels;
    uint const component_roles = metal_unreached_island_k | metal_stranded_island_k;
    if (arguments.pass == 0) {
        for (uint rank = position; rank <= islands; rank += threads) {
            weak[rank] = rank;
            if (rank == islands) continue;
            if (island_components[rank] == rank) metal_warp_t::fetch_or_relaxed(island_roles + rank, component_roles);
            island_entries[2 * rank] = island_entries[2 * rank + 1] = metal_unknown_k;
        }
        return;
    }
    if (arguments.pass == 1) {
        for (uint rank = group * metal_simdgroups_k + simdgroup; rank < islands; rank += groups * metal_simdgroups_k) {
            uint const node = island_nodes[rank], component = island_components[rank];
            uint const parent = island_parents[rank], roles = metal_warp_t::load_relaxed(island_roles + rank);
            metal_for_each_island_link(health, island_nodes, islands, node, lane, [&](uint linked) {
                metal_unite(weak, rank, linked);
                uint const next = island_components[linked];
                if (next == component) return;
                metal_warp_t::fetch_and_relaxed(island_roles + component, ~uint(metal_stranded_island_k));
                metal_warp_t::fetch_and_relaxed(island_roles + next, ~uint(metal_unreached_island_k));
            });
            if (lane != 0) continue;
            if (parent != metal_unknown_k) {
                metal_warp_t::fetch_or_relaxed(island_roles + rank, metal_linked_from_component_k);
                metal_warp_t::fetch_and_relaxed(island_roles + component, ~uint(metal_unreached_island_k));
            }
            if (roles & metal_links_to_component_k)
                metal_warp_t::fetch_and_relaxed(island_roles + component, ~uint(metal_stranded_island_k));
            if (parent != metal_unknown_k || (roles & metal_links_to_component_k)) metal_unite(weak, rank, islands);
            metal_claim_entry(island_entries, component, metal_entry_key(health, node, parent, roles));
        }
        return;
    }
    uint counted[2] = {0, 0};
    for (uint rank = position; rank <= islands && arguments.pass == 3; rank += threads)
        counted[0] += metal_find(weak, rank) == rank;
    for (uint rank = position; rank < islands && arguments.pass == 2; rank += threads) {
        uint const node = island_nodes[rank], component = island_components[rank];
        uint const roles = metal_warp_t::load_relaxed(island_roles + rank);
        metal_settle_entry(island_entries, component, metal_entry_key(health, node, island_parents[rank], roles), rank);
        uint const roles_of_component = metal_warp_t::load_relaxed(island_roles + component) & component_roles;
        if (component != rank) metal_warp_t::fetch_or_relaxed(island_roles + rank, roles_of_component);
        else {
            counted[0] += (roles_of_component & metal_unreached_island_k) != 0;
            counted[1] += (roles_of_component & metal_stranded_island_k) != 0;
        }
    }
    counted[0] = metal_warp_t::reduce_add(counted[0]), counted[1] = metal_warp_t::reduce_add(counted[1]);
    if (lane != 0) return;
    if (arguments.pass == 3 && counted[0])
        metal_warp_t::fetch_add_relaxed(counters + metal_round_components_k, counted[0]);
    if (arguments.pass == 2 && counted[0])
        metal_warp_t::fetch_add_relaxed(counters + metal_round_unreached_islands_k, counted[0]);
    if (arguments.pass == 2 && counted[1])
        metal_warp_t::fetch_add_relaxed(counters + metal_round_stranded_islands_k, counted[1]);
}

/** Links @p member from @p owner 's row, as the RAM round's @c add_or_swap_link_ does keeping none:
 *  a free slot first; else, inside the root's component, the slot whose resident is linked to by
 *  the most entries, when it stays linked to by more than @p member and the reach rule releases it.
 *  The heal runs alone, so it writes rows with plain stores; answers whether it wrote. */
inline bool metal_add_or_swap_link(thread metal_health_t& health, device uint* changed, uint owner, uint member,
                                   thread uint& changed_rows) {
    if (owner == member || owner >= health.rows_count || member >= health.rows_count) return false;
    device uint* const links = health.links_of(owner);
    uint const connectivity = health.connectivity, first = links[0];
    bool written = false;
    if (first == owner) links[0] = member, written = true;
    else {
        bool present = first == member;
        uint free_slot = connectivity, richest = connectivity;
        for (uint slot = 1; slot != connectivity && !present; ++slot) {
            uint const resident = links[slot];
            present = resident == member;
            if (resident == metal_unknown_k) {
                free_slot = slot;
                break;
            }
        }
        uint richest_links = health.in_links_of(member);
        for (uint slot = 0; !present && free_slot == connectivity && slot != connectivity && health.in_component(owner);
             ++slot) {
            uint const resident = slot ? links[slot] : first;
            if (resident >= health.rows_count || health.in_links_of(resident) <= richest_links + 1u) continue;
            richest = slot, richest_links = health.in_links_of(resident);
        }
        uint const displaced = richest == connectivity ? metal_unknown_k : richest ? links[richest] : first;
        if (present) {}
        else if (free_slot != connectivity) links[free_slot] = member, written = true;
        else if (richest != connectivity && health.take_link_under_lock_if_spare(owner, displaced)) {
            links[richest] = member;
            health.take_in_link_if_spare(displaced);
            written = true;
        }
    }
    if (!written) return false;
    metal_warp_t::fetch_add_relaxed(health.in_links + member, 1u);
    changed_rows += !metal_bitset_claim(changed, owner);
    return true;
}

/** The first of @p entry 's own links, in slot order up to its first sentinel, that lies inside the
 *  root's component, or the sentinel for none: where a heal links an unreached island from. */
inline uint metal_first_link_into_component(thread metal_health_t& health, uint entry) {
    device uint* const links = health.links_of(entry);
    for (uint slot = 0; slot != health.connectivity; ++slot) {
        uint const link = links[slot];
        if (link == metal_unknown_k) return metal_unknown_k;
        if (link < health.rows_count && link != entry && health.in_component(link)) return link;
    }
    return metal_unknown_k;
}

/** Links every unreached or stranded island component through the root's component with no
 *  distance, in rank order on one lane, as the RAM round's @c heal_ does: an unreached one's entry
 *  from the first of its own links inside the component, a stranded one's entry to the entry of the
 *  component linking to it. */
[[host_name("metal_heal_islands_kernel")]] kernel void metal_heal_islands_kernel(
    constant metal_graph_arguments_t& arguments [[buffer(metal_graph_arguments_buffer_k)]],
    device uint* neighbors [[buffer(metal_graph_neighbors_buffer_k)]],
    device uint* counters [[buffer(metal_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* changed [[buffer(metal_changed_buffer_k)]],
    device uint* island_nodes [[buffer(metal_island_nodes_buffer_k)]],
    device uint* island_components [[buffer(metal_island_components_buffer_k)]],
    device uint* island_parents [[buffer(metal_island_parents_buffer_k)]],
    device uint* island_entries [[buffer(metal_island_entries_buffer_k)]],
    device uint* island_roles [[buffer(metal_island_roles_buffer_k)]], uint position [[thread_position_in_grid]]) {
    if (position != 0) return;
    metal_health_t health = metal_health(arguments.end_id, arguments.connectivity, neighbors, counters,
                                         depths_from_root, depths_to_root, in_links, shallower_in_links, reached,
                                         reaches_root, two_ways_to_root);
    uint healed = 0, changed_rows = 0;
    for (uint rank = 0; rank != arguments.islands; ++rank) {
        if (island_components[rank] != rank) continue;
        uint const roles = island_roles[rank], member = island_entries[2 * rank], entry = island_nodes[member];
        uint const owner = roles & metal_unreached_island_k ? metal_first_link_into_component(health, entry)
                                                            : metal_unknown_k;
        if (owner != metal_unknown_k && metal_add_or_swap_link(health, changed, owner, entry, changed_rows))
            health.record_heal_link(owner, entry), ++healed;
        uint const parent = island_parents[member];
        if ((roles & metal_stranded_island_k) && parent != metal_unknown_k &&
            metal_add_or_swap_link(health, changed, entry, parent, changed_rows))
            health.record_heal_link(entry, parent), ++healed;
    }
    counters[metal_round_healed_k] += healed;
    counters[metal_round_changed_k] += changed_rows;
}

/** One simdgroup's slice of a restitch's scratch: its candidates as scored nodes of their
 *  positions, ascending, two rows' worth; the entry's nearest neighborhood-mates, a row's worth;
 *  the candidates' IDs, the live links first, and distances, two rows' worth each; and the
 *  positions it keeps, a row's worth. */
struct metal_stitch_t {
    device ulong* sorted;
    device ulong* mates;
    device uint* ids;
    device float* distances;
    device uint* kept;
};

/** Puts back what the kept candidates dropped of the @p live_count live links that reachability
 *  needs, exactly as the RAM index's @c keep_reach_ does, on one lane: a member whose last
 *  shallower in-link @p node is, and, when no pick stands closer to the root, the nearest live
 *  member that does. Answers the kept count. */
inline uint metal_keep_reach(thread metal_health_t& health, uint node, metal_stitch_t stitch, uint live_count,
                             uint kept_count, uint connectivity) {
    uint const depth = health.depth_from_root(node), depth_to_root = health.depth_to_root(node);
    // The saturated depth counts no shallower in-links, so its children are never released.
    uint const child = depth < metal_deepest_depth_k ? depth + 1 : uint(metal_unknown_depth_k);
    auto holds = [&](uint candidate) {
        for (uint rank = 0; rank != kept_count; ++rank)
            if (stitch.kept[rank] == candidate) return true;
        return false;
    };
    auto releasable = [&](uint candidate) {
        uint const member = stitch.ids[candidate];
        return child == metal_unknown_depth_k || health.depth_from_root(member) != child || candidate >= live_count ||
               health.take_shallower_in_link_if_spare(member);
    };
    // The kept members gather at the tail, ahead of which the farthest releasable pick gives way.
    uint tail_kept = 0;
    auto keep = [&](uint candidate) {
        if (holds(candidate)) return;
        if (kept_count < connectivity) {
            stitch.kept[kept_count++] = candidate;
            ++tail_kept;
            return;
        }
        uint const tail = kept_count - tail_kept;
        uint at = tail;
        while (at && !releasable(stitch.kept[at - 1])) --at;
        if (!at) return;
        stitch.kept[at - 1] = stitch.kept[tail - 1];
        stitch.kept[tail - 1] = candidate;
        ++tail_kept;
    };
    if (child != metal_unknown_depth_k)
        for (uint candidate = 0; candidate != live_count; ++candidate) {
            uint const member = stitch.ids[candidate];
            if (health.depth_from_root(member) != child || holds(candidate)) continue;
            if (!health.take_shallower_in_link_if_spare(member)) keep(candidate);
        }
    if (depth_to_root != metal_unknown_depth_k && depth_to_root != 0) {
        uint standing = 0;
        while (standing != kept_count && health.depth_to_root(stitch.ids[stitch.kept[standing]]) >= depth_to_root)
            ++standing;
        uint way = standing != kept_count ? stitch.kept[standing] : metal_unknown_k;
        for (uint candidate = 0; candidate != live_count && standing == kept_count; ++candidate)
            if (health.depth_to_root(stitch.ids[candidate]) < depth_to_root &&
                (way == metal_unknown_k || stitch.distances[candidate] < stitch.distances[way]))
                way = candidate;
        if (way != metal_unknown_k) keep(way);
    }
    return kept_count;
}

#pragma endregion Health

} // namespace unum::usearch::basic

namespace unum::usearch::basic {

using query_t = typename metal_kit_t::query_t;

#pragma region Walk

/** A kit's selector when it names one, which every kit scoring a metric beside its diversity rule
 *  does, else the diversity rule here. */
template <typename type_>
struct metal_voided {
    using type = void;
};
template <typename kit_, typename = void>
struct metal_has_selector {
    static constexpr constant bool value = false;
};
template <typename kit_>
struct metal_has_selector<kit_, typename metal_voided<typename kit_::selector_t>::type> {
    static constexpr constant bool value = true;
};

/** The ragged storage a kit reads: every value back to back, each entry's start, one more than the
 *  nodes, the kit's runtime parameters and the metric it scores under. */
struct metal_ragged_t {
    device uchar const* values;
    device ulong const* offsets;
    device uchar const* parameters;
    uint metric;

    device uchar const* data_of(uint node_id) { return values + offsets[node_id]; }
    uint bytes_of(uint node_id) { return uint(offsets[node_id + 1] - offsets[node_id]); }
};

/** One walker's walk: the storage it scores, the neighbors matrix and the counter words, its
 *  slice's visited set, spill tail, query and pivot spans, staged IDs and selection spans, its
 *  simdgroup and lane, the random stream and the telemetry, as the queue's walk verbs read them. */
struct metal_basic_walk_t {
    metal_ragged_t storage;
    device uint* neighbors;
    device atomic_uint* counters;

    device ulong* visited;
    device ulong* tail;
    device uchar* query_span;
    device uchar* pivot_span;
    device uint* ids;
    device ulong* scored_entries;
    device uint* selected_ids;
    device uint* candidate_ids;
    device uint* kept_positions;
    device float* candidate_distances;

    uint connectivity, tail_entries, expansion, selection_nodes, walk_width;
    uint simdgroup, lane;

    uint hash_shift, epoch, saturations;
    ulong random;
    uint hops, scored_count, restarts, lock_attempts, lock_skips, reselections, locked_drops, tail_compactions;
    uint tail_dropped;

    device uint* neighbors_for(uint node_id) { return metal_row(neighbors, node_id, connectivity); }

    /** Lays the entry of @p data and @p bytes out as the kit's query in @p span. */
    query_t prepare(device uchar const* data, uint bytes, device uchar* span) {
        return metal_kit_t::prepare(storage.parameters, data, bytes, span, lane);
    }
    query_t prepare_node(uint node_id, device uchar* span) {
        return prepare(storage.data_of(node_id), storage.bytes_of(node_id), span);
    }
};

/** The distance on rung @p rung_ from @p query to the entry of @p node_id, exact within @p radius
 *  and above it past; infinite for the sentinel. Every lane of the entry's group calls it. */
template <uint rung_>
inline float metal_score_entry(thread metal_ragged_t& storage, thread query_t const& query, uint node_id, float radius,
                               uint lane, uint simdgroup) {
    if (node_id == metal_unknown_k) return numeric_limits<float>::infinity();
    device uchar const* const data = storage.data_of(node_id);
    uint const bytes = storage.bytes_of(node_id);
    float const lower_bound = metal_kit_t::lower_bound(storage.metric, query, data, bytes);
    if (lower_bound > radius) return lower_bound;
    return metal_kit_t::template score<rung_>(storage.metric, query, data, bytes, radius, lane, simdgroup);
}

/** The radius a kit may prune past at a scored node: its packed distance, as the CUDA walks read
 *  @c worst().distance(), so both backends bound every score alike; none for the empty slot. */
inline float metal_radius_of(ulong node) {
    return node == metal_sentinel_k ? numeric_limits<float>::infinity() : metal_distance_of(node);
}
template <uint slots_>
inline float metal_radius_of(thread metal_sorted_candidates<slots_>& frontier) {
    return metal_radius_of(metal_warp_t::shuffle(frontier.entries[slots_ - 1], metal_warp_t::width_k - 1));
}
inline float metal_radius_of(thread metal_arena_frontier&) { return numeric_limits<float>::infinity(); }

/**
 *  @brief Scores each round of a walk's entries with the whole threadgroup: simdgroup zero posts
 *      the query and an entry per lane, and every simdgroup scores its share on @p rung_.
 *
 *  Simdgroups score @p lanes_per_entry_ lanes to an entry, in turns when an entry takes more lanes
 *  than the threadgroup's simdgroups split evenly. Only simdgroup zero walks, so the frontier, the
 *  visited set and the query's span stay its own; the others @ref serve rounds until it
 *  @ref finish es. Two barriers bracket every round, so the posted entries are read between them
 *  alone, as the CUDA and HIP block schedules do.
 */
template <uint rung_, uint lanes_per_entry_>
struct metal_schedule {
    static constexpr constant uint entries_per_simdgroup_k = metal_warp_t::width_k / lanes_per_entry_;
    static constexpr constant uint entries_per_group_k = entries_per_simdgroup_k * metal_simdgroups_k;
    static_assert(metal_warp_t::width_k % entries_per_group_k == 0, "Four simdgroups split 32 entries evenly");

    /** The round simdgroup zero posts and the distances the threadgroup writes for it. */
    struct shared_t {
        query_t query;
        uint ids[metal_simdgroup_width_k];
        float distances[metal_simdgroup_width_k];
        float radius;
        uint entries, finished;
    };

    threadgroup shared_t* posted;
    metal_ragged_t storage;
    uint simdgroup, lane;

    void open() {
        if (simdgroup == 0 && lane == 0) posted->finished = 0;
    }

    /** The distance from @p query to the entry this lane of simdgroup zero names. */
    float score(thread query_t const& query, uint node_id, float radius) {
        posted->ids[lane] = node_id;
        if (lane == 0) posted->query = query, posted->radius = radius, posted->entries = metal_warp_t::width_k;
        barrier_();
        score_share_();
        barrier_();
        return posted->distances[lane];
    }

    /** Scores the rounds simdgroup zero posts until it finishes. */
    void serve() {
        while (true) {
            barrier_();
            if (posted->finished) return;
            score_share_();
            barrier_();
        }
    }

    /** Releases the serving simdgroups once simdgroup zero walked its last entry. */
    void finish() {
        if (lane == 0) posted->finished = 1;
        barrier_();
    }

    void score_share_() {
        query_t const query = posted->query;
        float const radius = posted->radius;
        uint const entries = posted->entries;
        for (uint first = simdgroup * entries_per_simdgroup_k; first < entries; first += entries_per_group_k) {
            uint const position = first + lane / lanes_per_entry_;
            float const distance = metal_score_entry<rung_>(storage, query, posted->ids[position], radius, lane,
                                                            simdgroup);
            if (lane % lanes_per_entry_ == 0) posted->distances[position] = distance;
        }
    }

    /** Every thread of the threadgroup meets here, wherever it reaches it from. */
    static void barrier_() { threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); }
};

/** The schedule of every kit whose lane scores an entry alone: each lane scores its own entry, and
 *  a simdgroup walks by itself. */
template <uint rung_>
struct metal_schedule<rung_, 1> {
    struct shared_t {
        uint unused;
    };

    threadgroup shared_t* posted;
    metal_ragged_t storage;
    uint simdgroup, lane;

    void open() {}
    float score(thread query_t const& query, uint node_id, float radius) {
        return metal_score_entry<rung_>(storage, query, node_id, radius, lane, 0);
    }
    void serve() {}
    void finish() {}
};

/** The schedule a walk on kit rung @p rung_index_ scores through. */
template <uint rung_index_>
using metal_schedule_of =
    metal_schedule<metal_kit_t::rungs_k[rung_index_], metal_kit_t::lanes_per_row(metal_kit_t::rungs_k[rung_index_])>;

/** Scores one staged batch of @p count IDs in @c ids against @p query and folds it into the
 *  frontier; past the full candidates' worst rank nothing enters, so each entry is scored within
 *  the distance of that worst rank. */
template <typename frontier_, typename schedule_>
inline void metal_score_into_frontier(thread metal_basic_walk_t& walk, thread frontier_& frontier,
                                      thread schedule_& schedule, thread query_t const& query, uint count) {
    walk.scored_count += count;
    float const radius = metal_radius_of(frontier);
    uint const node_id = walk.lane < count ? walk.ids[walk.lane] : metal_unknown_k;
    float const distance = schedule.score(query, node_id, radius);
    frontier.push_batch(walk.lane < count ? metal_make(distance, node_id) : metal_sentinel_k, walk.lane);
}

/** Continues a greedy best-first walk toward @p query over whatever the frontier holds, as the
 *  CUDA basic walk does: landmarks from @b [first,bound), a transiently locked candidate held
 *  through one other expansion and dropped if still locked then, no neighbor at or past
 *  @p first_invisible followed, and the neighbors staged across hops scored a simdgroup-width at a
 *  time, since a basic kit pays per entry. Read-only on the graph. */
template <typename frontier_, typename schedule_>
inline void metal_search_ahead(thread metal_basic_walk_t& walk, thread frontier_& frontier, thread schedule_& schedule,
                               thread query_t const& query, uint first, uint bound, uint first_invisible) {
    ulong held = metal_sentinel_k;
    uint held_at_hop = 0, pending = 0;
    bool seeding = true;
    auto score = [&](uint count) { metal_score_into_frontier(walk, frontier, schedule, query, count); };
    while (true) {
        if (seeding) {
            uint const count = metal_sample_landmarks(walk, first, bound, frontier);
            // Every published node of the range is already visited.
            if (!count) break;
            score(count);
            seeding = false;
            continue;
        }
        bool const revisiting = held != metal_sentinel_k && walk.hops != held_at_hop;
        ulong const candidate = revisiting ? held : frontier.pop_nearest(walk.lane);
        if (revisiting) held = metal_sentinel_k;
        // Neighbors staged across hops score before the walk stops or reseeds, since they may
        // refill the frontier.
        if (pending && (candidate == metal_sentinel_k || frontier.beyond_radius(candidate))) {
            score(pending);
            pending = 0;
            metal_warp_t::sync();
            continue;
        }
        if (candidate == metal_sentinel_k) {
            if (held != metal_sentinel_k) {
                held_at_hop = ~0u;
                continue;
            }
            // A drained frontier before an exhausted walk re-seeds, until no published node of the
            // range is left unvisited.
            if (frontier.exhausted()) break;
            seeding = true;
            ++walk.restarts;
            continue;
        }
        if (frontier.beyond_radius(candidate)) break;
        ++walk.hops;
        device uint* const links = walk.neighbors_for(metal_id_of(candidate));
        uint const first_value = metal_warp_t::shuffle(
            walk.lane == 0 ? metal_warp_t::load_acquire(links) : metal_unknown_k, 0);
        // Every lane reads the candidate's links, so every lane fences behind the leader's acquire.
        metal_warp_t::fence();
        if (first_value == metal_unknown_k) {
            if (revisiting || held != metal_sentinel_k) ++walk.locked_drops;
            else held = candidate, held_at_hop = walk.hops;
            continue;
        }
        metal_expand_round(walk, frontier, candidate, first_value, first_invisible, held, held_at_hop, pending, score);
    }
    walk.locked_drops += held != metal_sentinel_k;
}

/** Keeps a candidate only when it is closer to the node than to every kept one, as the CPU's
 *  @c keep_diverse_t does, over @p candidates_count candidates ascending by distance, preparing
 *  each as the pivot in the walker's second span; writes the kept positions and answers how many
 *  it kept of them. */
template <typename schedule_>
inline uint metal_keep_diverse(thread metal_basic_walk_t& walk, thread schedule_& schedule,
                               device uint const* candidate_ids, device float const* candidate_distances,
                               uint candidates_count, device uint* kept_positions) {
    uint kept_count = 0;
#pragma unroll 1
    for (uint candidate = 0; candidate < candidates_count && kept_count < walk.connectivity; ++candidate) {
        float const to_query = candidate_distances[candidate];
        query_t const pivot = kept_count ? walk.prepare_node(candidate_ids[candidate], walk.pivot_span) : query_t {};
        bool diverse = true;
        // Simdgroup-uniform bounds.
        for (uint base = 0; base < kept_count && diverse; base += metal_warp_t::width_k) {
            uint const kept_id = base + walk.lane < kept_count ? candidate_ids[kept_positions[base + walk.lane]]
                                                               : metal_unknown_k;
            diverse = simd_all(!(schedule.score(pivot, kept_id, to_query) < to_query));
            walk.scored_count += min(kept_count - base, metal_warp_t::width_k);
        }
        if (!diverse) continue;
        if (walk.lane == 0) kept_positions[kept_count] = candidate;
        ++kept_count;
        metal_warp_t::sync();
    }
    return kept_count;
}

/** Selects among @p candidates_count candidates ascending by distance to the node of @p query by
 *  the kit's selector, or by the diversity rule for a kit naming none; answers the kept count, the
 *  kept positions ascending. */
template <typename kit_, typename schedule_>
inline uint metal_select(thread metal_basic_walk_t& walk, thread schedule_& schedule, thread query_t const& query,
                         device uint const* candidate_ids, device float const* candidate_distances,
                         uint candidates_count, device uint* kept_positions) {
    if constexpr (metal_has_selector<kit_>::value)
        return kit_::selector_t::template select<kit_>(query, walk.storage.values, walk.storage.offsets, candidate_ids,
                                                       candidates_count, walk.connectivity, kept_positions, walk.lane);
    else
        return metal_keep_diverse(walk, schedule, candidate_ids, candidate_distances, candidates_count, kept_positions);
}

/** Fills the slots a selection left empty with the nearest candidates it did not keep, after its
 *  @p kept_count picks, up to the connectivity or the candidates, as the CPU's @c fill_selection
 *  does; the kept prefix is a subsequence of the candidates, so a single cursor tells the pruned
 *  ones from the kept ones. */
inline uint metal_backfill(thread metal_basic_walk_t& walk, uint candidates_count, device uint* kept_positions,
                           uint kept_count) {
    uint const bound = min(walk.connectivity, candidates_count), selected_count = kept_count;
    for (uint candidate = 0, matched = 0; candidate < candidates_count && kept_count < bound; ++candidate) {
        if (matched != selected_count && kept_positions[matched] == candidate) {
            ++matched;
            continue;
        }
        if (walk.lane == 0) kept_positions[kept_count] = candidate;
        ++kept_count;
        metal_warp_t::sync();
    }
    metal_warp_t::sync();
    return kept_count;
}

/** Hands all of a finished walk's candidates toward @p query, its @c selection_nodes ranks
 *  ascending, to the selector, backfills, and writes the kept candidates as node IDs in
 *  @c selected_ids; answers the kept count. */
template <typename frontier_, typename schedule_>
inline uint metal_select_neighbors(thread metal_basic_walk_t& walk, thread frontier_& frontier,
                                   thread schedule_& schedule, thread query_t const& query) {
    uint candidates_count = 0;
    auto stage = [&](uint rank, ulong entry) {
        bool const live = rank < walk.selection_nodes && entry != metal_sentinel_k;
        if (live) {
            walk.candidate_ids[rank] = metal_id_of(entry);
            walk.candidate_distances[rank] = metal_distance_of(entry);
        }
        candidates_count += popcount(metal_warp_t::ballot(live));
    };
    frontier.for_each_rank(walk.selection_nodes, walk.lane, stage);
    metal_warp_t::sync();
    uint selected_count = metal_select<metal_kit_t>(walk, schedule, query, walk.candidate_ids, walk.candidate_distances,
                                                    candidates_count, walk.kept_positions);
    selected_count = metal_backfill(walk, candidates_count, walk.kept_positions, selected_count);
    // The caller links to nodes, not to positions in the candidates.
    for (uint slot = walk.lane; slot < selected_count; slot += metal_warp_t::width_k)
        walk.selected_ids[slot] = walk.candidate_ids[walk.kept_positions[slot]];
    metal_warp_t::sync();
    return selected_count;
}

/** The chains a wave's walkers thread their picks onto for the back-link launch behind them: each
 *  node's last pick, unknown for none, the pick each was threaded before, and the nodes any walker
 *  picked with their count. */
struct metal_back_links_t {
    device uint* heads;
    device uint* next;
    device uint* touched_ids;
    device atomic_uint* touched_count;
};

/** Threads the @p selected_count picks of @p node_id onto their targets' chains, listing each
 *  target the first time any walker of the launch picks it. */
inline void metal_thread_back_links(thread metal_basic_walk_t& walk, metal_back_links_t links, uint node_id,
                                    uint first_id, uint selected_count) {
    for (uint slot = walk.lane; slot < selected_count; slot += metal_warp_t::width_k) {
        uint const target = walk.selected_ids[slot];
        uint const pick = (node_id - first_id) * walk.connectivity + slot;
        uint const previous = metal_warp_t::exchange_acquire(links.heads + target, pick);
        links.next[pick] = previous;
        if (previous == metal_unknown_k)
            links.touched_ids[atomic_fetch_add_explicit(links.touched_count, 1u, memory_order_relaxed)] = target;
    }
    metal_warp_t::sync();
}

/** Publishes @p node_id 's row from the walk's @p selected_count picks: tail slots relaxed, unknown
 *  past the picks, and slot zero last with a release, a self-loop for published-empty, so walks on
 *  other streams read either row whole. */
inline void metal_publish_row(thread metal_basic_walk_t& walk, uint node_id, uint selected_count) {
    device uint* const own = walk.neighbors_for(node_id);
    for (uint slot = walk.lane; slot < walk.connectivity; slot += metal_warp_t::width_k)
        if (slot)
            metal_warp_t::store_relaxed(own + slot, slot < selected_count ? walk.selected_ids[slot] : metal_unknown_k);
    // Every lane wrote slots, so every lane fences ahead of the leader's release.
    metal_warp_t::fence();
    if (walk.lane == 0) metal_warp_t::store_release(own, selected_count ? walk.selected_ids[0] : node_id);
    metal_warp_t::sync();
}

/** Inserts @p node_id, whose bytes and offsets the host already wrote: walks the published graph
 *  below it, selects its neighbors, publishes its row with one release and threads its picks for
 *  the back-link launch behind the wave. */
template <typename frontier_, typename schedule_>
inline void metal_insert_one(thread metal_basic_walk_t& walk, thread schedule_& schedule, metal_back_links_t links,
                             uint node_id, uint first_id, uint end_id) {
    query_t const query = walk.prepare_node(node_id, walk.query_span);
    frontier_ frontier;
    frontier.clear(walk);
    // A virgin index publishes an empty node instead of walking.
    if (node_id != 0) metal_search_ahead(walk, frontier, schedule, query, 0, node_id, end_id);
    walk.tail_compactions += frontier.compactions();
    walk.tail_dropped += frontier.dropped();
    uint const selected_count = metal_select_neighbors(walk, frontier, schedule, query);
    metal_publish_row(walk, node_id, selected_count);
    if (walk.lane == 0) atomic_fetch_add_explicit(walk.counters + metal_published_k, 1u, memory_order_relaxed);
    metal_thread_back_links(walk, links, node_id, first_id, selected_count);
}

/** Re-selects the published row of @p node_id once its wave's back-links made the wave's nodes
 *  reachable from one another: walks from the row's own neighbors over every node below
 *  @p sampling_bound, selects again, rewrites the row and threads the new picks. */
template <typename frontier_, typename schedule_>
inline void metal_reselect_one(thread metal_basic_walk_t& walk, thread schedule_& schedule, metal_back_links_t links,
                               uint node_id, uint first_id, uint sampling_bound) {
    query_t const query = walk.prepare_node(node_id, walk.query_span);
    frontier_ frontier;
    frontier.clear(walk);
    metal_test_and_set(walk, node_id);
    // The walk starts from the row's own neighbors, which the wave's first walk already found.
    device uint* const own = walk.neighbors_for(node_id);
    for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += metal_warp_t::width_k) {
        uint const slot = slot_base + walk.lane;
        uint const seed_id = slot < walk.connectivity ? metal_warp_t::load_acquire(own + slot) : metal_unknown_k;
        bool const usable = seed_id < sampling_bound && seed_id != node_id;
        bool const fresh = metal_test_and_set(walk, usable ? seed_id : metal_unknown_k);
        uint const fresh_mask = metal_warp_t::ballot(fresh);
        if (fresh) walk.ids[popcount(fresh_mask & metal_warp_t::below(walk.lane))] = seed_id;
        metal_warp_t::sync();
        if (fresh_mask) metal_score_into_frontier(walk, frontier, schedule, query, popcount(fresh_mask));
        metal_warp_t::sync();
    }
    metal_search_ahead(walk, frontier, schedule, query, 0, sampling_bound, sampling_bound);
    walk.tail_compactions += frontier.compactions();
    walk.tail_dropped += frontier.dropped();
    uint const selected_count = metal_select_neighbors(walk, frontier, schedule, query);
    metal_publish_row(walk, node_id, selected_count);
    metal_thread_back_links(walk, links, node_id, first_id, selected_count);
}

/** Gathers @p node_id 's live links, other than itself and the sentinels, into @c candidate_ids
 *  from position zero, below @p rows_count when it bounds them; answers how many. */
inline uint metal_gather_live(thread metal_basic_walk_t& walk, uint node_id, uint rows_count) {
    device uint* const live_row = walk.neighbors_for(node_id);
    uint live_count = 0;
    for (uint slot_base = 0; slot_base < walk.connectivity; slot_base += metal_warp_t::width_k) {
        uint const slot = slot_base + walk.lane;
        uint const live_id = slot < walk.connectivity ? metal_warp_t::load_acquire(live_row + slot) : metal_unknown_k;
        bool const usable = live_id != metal_unknown_k && live_id != node_id && live_id < rows_count;
        uint const usable_mask = metal_warp_t::ballot(usable);
        if (usable) walk.candidate_ids[live_count + popcount(usable_mask & metal_warp_t::below(walk.lane))] = live_id;
        live_count += popcount(usable_mask);
        metal_warp_t::sync();
    }
    return live_count;
}

/** Compacts the chained IDs from @p live_count up to @p chained_end of @c candidate_ids the live
 *  prefix lacks behind it, dropping @p node_id ; answers how many stay. */
inline uint metal_compact_fresh(thread metal_basic_walk_t& walk, uint node_id, uint live_count, uint chained_end) {
    device uint* const candidate_ids = walk.candidate_ids;
    uint fresh_count = 0;
    for (uint slot_base = live_count; slot_base < chained_end; slot_base += metal_warp_t::width_k) {
        uint const slot = slot_base + walk.lane;
        uint const chained_id = slot < chained_end ? candidate_ids[slot] : metal_unknown_k;
        bool fresh = chained_id != metal_unknown_k && chained_id != node_id;
        for (uint live_index = 0; fresh && live_index != live_count; ++live_index)
            fresh = candidate_ids[live_index] != chained_id;
        uint const fresh_mask = metal_warp_t::ballot(fresh);
        metal_warp_t::sync();
        if (fresh)
            candidate_ids[live_count + fresh_count + popcount(fresh_mask & metal_warp_t::below(walk.lane))] =
                chained_id;
        fresh_count += popcount(fresh_mask);
        metal_warp_t::sync();
    }
    return fresh_count;
}

/** Appends the candidates past @p live_count up to @p candidates_count to @p node_id 's row in
 *  place, tail slots relaxed and slot zero last with a release when the row held no live link. */
inline void metal_append_links(thread metal_basic_walk_t& walk, uint node_id, uint live_count, uint candidates_count) {
    device uint* const live_row = walk.neighbors_for(node_id);
    for (uint slot = max(live_count, 1u) + walk.lane; slot < candidates_count; slot += metal_warp_t::width_k)
        metal_warp_t::store_relaxed(live_row + slot, walk.candidate_ids[slot]);
    metal_warp_t::fence();
    if (walk.lane == 0 && !live_count) metal_warp_t::store_release(live_row, walk.candidate_ids[0]);
    metal_warp_t::sync();
}

/**
 *  @brief Back-links one node the launch's walkers picked, chasing their picks from the chain heads
 *      through @c next in rounds, as the CUDA basic back-linker does.
 *
 *  Each round caches the live row and as many walkers as @c selection_nodes leave room for, appends
 *  them while the row has room, and re-selects over the union otherwise, the next round starting
 *  from the row the last one wrote, so no walker is dropped. The launch hands each node to one
 *  walker, the only writer of its row, while walks on other streams may read it throughout: every
 *  slot changes by one relaxed store and slot zero last, with a release.
 */
template <typename schedule_>
inline void metal_back_link_one(thread metal_basic_walk_t& walk, thread schedule_& schedule, metal_back_links_t links,
                                uint node_id, uint first_id) {
    uint const connectivity = walk.connectivity;
    device uint* const candidate_ids = walk.candidate_ids;
    // A chain is serial, so one lane follows it, and the rounds resume where it stopped; no lane
    // writes the head before the call returns, so every lane reads it.
    uint pick = links.heads[node_id];
    while (pick != metal_unknown_k) {
        uint const live_count = metal_gather_live(walk, node_id, metal_unknown_k);
        // The chain's next walkers fill the slots past the live row.
        uint chained_end = live_count;
        if (walk.lane == 0)
            for (; pick != metal_unknown_k && chained_end != walk.selection_nodes; pick = links.next[pick])
                candidate_ids[chained_end++] = first_id + pick / connectivity;
        chained_end = metal_warp_t::shuffle(chained_end, 0);
        pick = metal_warp_t::shuffle(pick, 0);
        metal_warp_t::sync();
        uint const fresh_count = metal_compact_fresh(walk, node_id, live_count, chained_end);
        if (!fresh_count) continue;
        uint const candidates_count = live_count + fresh_count;

        // Room for every walker of the round: appended in place, no distance needed.
        if (candidates_count <= connectivity) {
            metal_append_links(walk, node_id, live_count, candidates_count);
            continue;
        }

        query_t const query = walk.prepare_node(node_id, walk.query_span);
        metal_sorted_buffer sorted_candidates {walk.scored_entries, 0, walk.selection_nodes};
        for (uint slot_base = 0; slot_base < candidates_count; slot_base += metal_warp_t::width_k) {
            uint const slot = slot_base + walk.lane;
            uint const candidate_id = slot < candidates_count ? candidate_ids[slot] : metal_unknown_k;
            float const distance = schedule.score(query, candidate_id, numeric_limits<float>::infinity());
            ulong const scored = candidate_id != metal_unknown_k ? metal_make(distance, candidate_id)
                                                                 : metal_sentinel_k;
            sorted_candidates.insert_batch(metal_sort_lanes(scored, walk.lane),
                                           min(candidates_count - slot_base, metal_warp_t::width_k), walk.lane);
        }
        walk.scored_count += candidates_count;
        ++walk.reselections;

        // The union is scored, so its IDs take their places, ascending.
        for (uint position = walk.lane; position < sorted_candidates.size; position += metal_warp_t::width_k) {
            candidate_ids[position] = metal_id_of(sorted_candidates.at(position));
            walk.candidate_distances[position] = metal_distance_of(sorted_candidates.at(position));
        }
        metal_warp_t::sync();
        uint refined_count = metal_select<metal_kit_t>(walk, schedule, query, candidate_ids, walk.candidate_distances,
                                                       sorted_candidates.size, walk.kept_positions);
        refined_count = metal_backfill(walk, sorted_candidates.size, walk.kept_positions, refined_count);
        for (uint slot = walk.lane; slot < refined_count; slot += metal_warp_t::width_k)
            walk.selected_ids[slot] = candidate_ids[walk.kept_positions[slot]];
        metal_warp_t::sync();
        metal_publish_row(walk, node_id, refined_count);
    }
}

/** One query's full walk: prepares the query into its span, walks the graph, and writes its nearest
 *  @p count IDs and distances ascending, sentinels past the live prefix. Safe against concurrent
 *  insertions: the walk gates on every slot-zero word and never writes the graph. */
template <typename frontier_, typename schedule_>
inline void metal_walk_one(thread metal_basic_walk_t& walk, thread schedule_& schedule,
                           constant metal_basic_arguments_t& arguments, device uchar const* queries,
                           device ulong const* query_offsets, device uint* result_ids, device float* result_distances,
                           uint query_index) {
    ulong const origin = query_offsets[0], first = query_offsets[query_index] - origin;
    query_t const query = walk.prepare(queries + first, uint(query_offsets[query_index + 1] - origin - first),
                                       walk.query_span);
    // Concurrent walks the batch leaves idle widen this one, never past its filling nodes.
    uint live = walk.lane == 0 ? atomic_fetch_add_explicit(walk.counters + metal_live_k, 1u, memory_order_relaxed) + 1
                               : 0;
    live = metal_warp_t::shuffle(live, 0);
    walk.walk_width = walk_width(arguments.concurrent_walks, live, arguments.queries_count - query_index,
                                 arguments.filling_nodes);
    frontier_ frontier;
    frontier.clear(walk);
    metal_search_ahead(walk, frontier, schedule, query, 0, arguments.sampling_bound, arguments.sampling_bound);
    if (walk.lane == 0) atomic_fetch_sub_explicit(walk.counters + metal_live_k, 1u, memory_order_relaxed);
    walk.tail_compactions += frontier.compactions();
    walk.tail_dropped += frontier.dropped();

    // The rank extraction shuffles, so every lane takes part and only in-range ones store.
    uint const count = arguments.count;
    device uint* const ids_row = metal_row(result_ids, query_index, count);
    device float* const distances_row = metal_row(result_distances, query_index, count);
    auto write_rank = [&](uint rank, ulong entry) {
        if (rank >= count) return;
        ids_row[rank] = metal_id_of(entry);
        distances_row[rank] = metal_distance_of(entry);
    };
    frontier.for_each_rank(count, walk.lane, write_rank);
}

#pragma endregion Walk

#pragma region Rounds

/** The span among @p count ascending @p starts holding @p position, the last whose start is at or
 *  before it: a gather's run, or a restitch's neighborhood. */
inline uint metal_span_of(device uint const* starts, uint count, uint position) {
    uint found = 0;
    for (uint remaining = count; remaining;) {
        uint const half = remaining / 2;
        if (starts[found + half + 1] <= position) found += half + 1, remaining -= half + 1;
        else remaining = half;
    }
    return found;
}

/** The runs a gather files and the neighborhoods a restitch reads, as the host binds them. */
struct metal_round_t {
    device ushort* entry_neighborhoods;
    device uint const* neighborhood_entries;
    device uint const* neighborhood_offsets;
    device uint const* hub_nodes;
    device uint4 const* runs;
    device uint const* run_entries;
    uint runs_count, neighborhoods, pass;
};

/**
 *  @brief Files the entry at @p position of a gather's runs under its run's nearest hub, the lower
 *      one among equals, as RAM's gather files them: the entry prepares as the query, and every
 *      lane scores a hub within the nearest so far.
 *
 *  A run is its first entry, its width, its first hub and its hubs; a first gather's run spans IDs,
 *  a later one positions among the entries in neighborhood order. An entry that no distance reached
 *  joins its run's first hub.
 */
template <typename schedule_>
inline void metal_gather_one(thread metal_basic_walk_t& walk, thread schedule_& schedule, metal_round_t round,
                             uint position) {
    uint const run_index = metal_span_of(round.run_entries, round.runs_count, position);
    uint4 const run = round.runs[run_index];
    uint const offset = position - round.run_entries[run_index];
    uint const node_id = round.pass ? round.neighborhood_entries[run.x + offset] : run.x + offset;
    query_t const query = walk.prepare_node(node_id, walk.query_span);
    device uint const* const hubs = round.hub_nodes + run.z;
    ulong nearest = metal_sentinel_k;
    float radius = numeric_limits<float>::infinity();
    // Simdgroup-uniform bounds.
    for (uint base = 0; base < run.w; base += metal_warp_t::width_k) {
        uint const hub = base + walk.lane;
        float const distance = schedule.score(query, hub < run.w ? hubs[hub] : metal_unknown_k, radius);
        // Past the radius a score is a bound, never a nearer hub.
        ulong const scored = hub < run.w && !(distance > radius) ? metal_make(distance, hub) : metal_sentinel_k;
        nearest = metal_warp_t::reduce_min(scored < nearest ? scored : nearest);
        radius = metal_radius_of(nearest);
    }
    walk.scored_count += run.w;
    if (walk.lane == 0)
        round.entry_neighborhoods[node_id] = ushort(nearest == metal_sentinel_k ? 0u : metal_id_of(nearest));
    metal_warp_t::sync();
}

/** The links one restitch added, dropped and the rows it changed, identical on every lane. */
struct metal_restitch_tally_t {
    uint added, dropped, changed;
};

/** Scores into @c candidate_ids past the @p live_count live links the nearest of the
 *  neighborhood's @p count @p entries no live link names, kept ascending, each within the farthest
 *  kept once a row's worth is full; answers how many. */
template <typename schedule_>
inline uint metal_nearest_mates(thread metal_basic_walk_t& walk, thread schedule_& schedule,
                                thread query_t const& query, uint node_id, device uint const* entries, uint count,
                                uint live_count) {
    uint const connectivity = walk.connectivity;
    device uint* const candidate_ids = walk.candidate_ids;
    metal_sorted_buffer nearest {walk.scored_entries, 0, connectivity};
    // Simdgroup-uniform bounds.
    for (uint base = 0; base < count; base += metal_warp_t::width_k) {
        uint const at = base + walk.lane;
        uint mate_id = at < count ? entries[at] : metal_unknown_k;
        bool fresh = mate_id != metal_unknown_k && mate_id != node_id;
        for (uint live = 0; fresh && live != live_count; ++live) fresh = candidate_ids[live] != mate_id;
        if (!fresh) mate_id = metal_unknown_k;
        float const radius = nearest.size == connectivity ? metal_radius_of(nearest.at(connectivity - 1))
                                                          : numeric_limits<float>::infinity();
        float const distance = schedule.score(query, mate_id, radius);
        uint const scored_count = popcount(metal_warp_t::ballot(fresh));
        nearest.insert_batch(metal_sort_lanes(fresh ? metal_make(distance, mate_id) : metal_sentinel_k, walk.lane),
                             scored_count, walk.lane);
        walk.scored_count += scored_count;
    }
    for (uint mate = walk.lane; mate < nearest.size; mate += metal_warp_t::width_k) {
        candidate_ids[live_count + mate] = metal_id_of(nearest.at(mate));
        walk.candidate_distances[live_count + mate] = metal_distance_of(nearest.at(mate));
    }
    metal_warp_t::sync();
    return nearest.size;
}

/**
 *  @brief Re-stitches the entry at @p position of the neighborhoods: its live links and its
 *      nearest neighborhood-mates no live link names compete, ascending; only those the selector
 *      keeps stay, with no refill, then those the health check's reach rule needs.
 *
 *  The live links lead the first half of the selection's candidates and the mates follow them, as
 *  @c metal_keep_reach reads them; the second half holds them ascending, as the selector reads
 *  them. The row is rewritten only where it changed, tail slots relaxed and slot zero last with a
 *  release, and counted once in the changed bits.
 */
template <typename schedule_>
inline void metal_restitch_one(thread metal_basic_walk_t& walk, thread schedule_& schedule,
                               thread metal_health_t& health, device uint* changed, metal_round_t round, uint position,
                               thread metal_restitch_tally_t& tally) {
    uint const connectivity = walk.connectivity, half = walk.selection_nodes / 2;
    uint const neighborhood = metal_span_of(round.neighborhood_offsets, round.neighborhoods, position);
    uint const first = round.neighborhood_offsets[neighborhood];
    uint const count = round.neighborhood_offsets[neighborhood + 1] - first;
    uint const node_id = round.neighborhood_entries[position];
    device uint* const candidate_ids = walk.candidate_ids;
    device float* const candidate_distances = walk.candidate_distances;
    device uint* const kept_positions = walk.kept_positions;
    device uint* const live_row = walk.neighbors_for(node_id);

    // The live links first; a self-loop in slot zero means none.
    uint const live_count = metal_gather_live(walk, node_id, health.rows_count);
    query_t const query = walk.prepare_node(node_id, walk.query_span);
    uint const mates_count = metal_nearest_mates(walk, schedule, query, node_id, round.neighborhood_entries + first,
                                                 count, live_count);
    // Simdgroup-uniform bounds.
    for (uint base = 0; base < live_count; base += metal_warp_t::width_k) {
        uint const at = base + walk.lane;
        float const distance = schedule.score(query, at < live_count ? candidate_ids[at] : metal_unknown_k,
                                              numeric_limits<float>::infinity());
        if (at < live_count) candidate_distances[at] = distance;
    }
    walk.scored_count += live_count;
    metal_warp_t::sync();
    uint const candidates_count = live_count + mates_count;
    if (!candidates_count) return;

    // Ascending by distance, positions breaking ties, into the second half, as the selector reads.
    metal_sorted_buffer ascending {walk.scored_entries, 0, candidates_count};
    for (uint base = 0; base < candidates_count; base += metal_warp_t::width_k) {
        uint const at = base + walk.lane;
        ulong const scored = at < candidates_count ? metal_make(candidate_distances[at], at) : metal_sentinel_k;
        ascending.insert_batch(metal_sort_lanes(scored, walk.lane), min(candidates_count - base, metal_warp_t::width_k),
                               walk.lane);
    }
    device uint* const sorted_ids = candidate_ids + half;
    device float* const sorted_distances = candidate_distances + half;
    for (uint rank = walk.lane; rank < candidates_count; rank += metal_warp_t::width_k) {
        sorted_ids[rank] = candidate_ids[metal_id_of(ascending.at(rank))];
        sorted_distances[rank] = metal_distance_of(ascending.at(rank));
    }
    metal_warp_t::sync();
    uint kept = metal_select<metal_kit_t>(walk, schedule, query, sorted_ids, sorted_distances, candidates_count,
                                          kept_positions);
    // Back to the candidates' own positions, which the reach rule tells live links by.
    for (uint slot = walk.lane; slot < kept; slot += metal_warp_t::width_k)
        kept_positions[slot] = metal_id_of(ascending.at(kept_positions[slot]));
    metal_warp_t::sync();
    metal_stitch_t stitch;
    stitch.sorted = walk.scored_entries, stitch.mates = walk.scored_entries;
    stitch.ids = candidate_ids, stitch.distances = candidate_distances, stitch.kept = kept_positions;
    kept = metal_warp_t::shuffle(
        walk.lane == 0 ? metal_keep_reach(health, node_id, stitch, live_count, kept, connectivity) : 0u, 0);
    metal_warp_t::sync();

    // Positions below the live count are live links, so the kept ones among them stay.
    uint in_live = 0;
    for (uint slot = walk.lane; slot < kept; slot += metal_warp_t::width_k)
        in_live += kept_positions[slot] < live_count;
    in_live = metal_warp_t::reduce_add(in_live);
    tally.added += kept - in_live, tally.dropped += live_count - in_live;
    if (in_live == kept && in_live == live_count) return;
    // Simdgroup-uniform bounds.
    for (uint slot_base = 0; slot_base < connectivity; slot_base += metal_warp_t::width_k) {
        uint const slot = slot_base + walk.lane;
        uint const value = slot < kept ? candidate_ids[kept_positions[slot]] : metal_unknown_k;
        if (slot && slot < connectivity && metal_warp_t::load_relaxed(live_row + slot) != value)
            metal_warp_t::store_relaxed(live_row + slot, value);
    }
    metal_warp_t::fence();
    if (walk.lane == 0) {
        metal_warp_t::store_release(live_row, kept ? candidate_ids[kept_positions[0]] : node_id);
        tally.changed += !metal_bitset_claim(changed, node_id);
    }
    metal_warp_t::sync();
}

#pragma endregion Rounds

#pragma region Kernels

/** Opens the walk of the walker at @p slice over its arena slice, as the host's arena shape lays
 *  it out: the visited set, the spill tail, then the scratch's query and pivot spans, the staged
 *  IDs and the selection spans, with fresh telemetry. */
inline metal_basic_walk_t metal_open(constant metal_basic_arguments_t& arguments, metal_ragged_t storage,
                                     device uint* neighbors, device uchar* arena, device atomic_uint* counters,
                                     uint slice, uint simdgroup, uint lane) {
    metal_basic_walk_t walk;
    device uchar* const slice_base = metal_row(arena, slice, arguments.slice_bytes);
    uint offset = 0;
    auto const carve = [&](uint bytes) {
        device uchar* const carved = slice_base + offset;
        offset += bytes;
        return carved;
    };
    uint const selection_nodes = arguments.selection_nodes;
    walk.storage = storage, walk.neighbors = neighbors, walk.counters = counters;
    walk.visited = (device ulong*)carve(arguments.visited_slots * 8);
    walk.tail = (device ulong*)carve(arguments.tail_entries * 8);
    walk.query_span = carve(arguments.span_bytes);
    walk.pivot_span = carve(arguments.span_bytes);
    walk.ids = (device uint*)carve(3 * metal_simdgroup_width_k * 4);
    walk.scored_entries = (device ulong*)carve(selection_nodes * 8);
    walk.selected_ids = (device uint*)carve(selection_nodes * 4);
    walk.candidate_ids = (device uint*)carve(selection_nodes * 4);
    walk.kept_positions = (device uint*)carve(selection_nodes * 4);
    walk.candidate_distances = (device float*)carve(selection_nodes * 4);
    walk.connectivity = arguments.connectivity, walk.tail_entries = arguments.tail_entries;
    walk.expansion = arguments.expansion, walk.selection_nodes = selection_nodes, walk.walk_width = 1;
    walk.lane = lane, walk.simdgroup = simdgroup;
    walk.hash_shift = arguments.hash_shift, walk.epoch = 0, walk.saturations = 0, walk.random = 0;
    walk.hops = 0, walk.scored_count = 0, walk.restarts = 0, walk.lock_attempts = 0, walk.lock_skips = 0;
    walk.reselections = 0, walk.locked_drops = 0, walk.tail_compactions = 0, walk.tail_dropped = 0;
    return walk;
}

/** Hands @p walk every index in @b [first,end) the calling walker assigns itself from @p assigned:
 *  every simdgroup walks under a lane schedule, simdgroup zero alone under a threadgroup one, the
 *  others serving its rounds until it runs out. */
template <typename schedule_, typename walk_type_>
inline void metal_for_each_assigned(thread schedule_& schedule, device uint* assigned, uint first, uint end,
                                    thread walk_type_& walk) {
    if (schedule.posted && schedule.simdgroup != 0) return schedule.serve();
    while (true) {
        uint offset = schedule.lane == 0 ? metal_warp_t::fetch_add_relaxed(assigned, 1u) : 0;
        offset = first + metal_warp_t::shuffle(offset, 0);
        if (offset >= end) break;
        walk(offset);
    }
    schedule.finish();
}

/** Which walk a build launch takes over its wave: the insertion, or the re-selection behind the
 *  wave's back-links. */
enum metal_build_walk_t : uint {
    metal_build_insert_k = 0,
    metal_build_refine_k = 1,
};

/** The walkers of one launch: a simdgroup each under a lane schedule, a threadgroup each under a
 *  threadgroup one, whose slices and schedule state the kernel opens. */
template <uint rung_index_>
struct metal_walkers {
    using schedule_t = metal_schedule_of<rung_index_>;
    static constexpr constant bool alone_k = metal_kit_t::lanes_per_row(metal_kit_t::rungs_k[rung_index_]) == 1;

    static uint slice_of(uint group, uint simdgroup) {
        return alone_k ? group * metal_simdgroups_k + simdgroup : group;
    }
    static schedule_t schedule(threadgroup typename schedule_t::shared_t* posted, metal_ragged_t storage,
                               uint simdgroup, uint lane) {
        schedule_t schedule;
        schedule.posted = alone_k ? nullptr : posted;
        schedule.storage = storage, schedule.simdgroup = simdgroup, schedule.lane = lane;
        schedule.open();
        // The posted round's first word reaches every simdgroup before any serves.
        if (!alone_k) threadgroup_barrier(mem_flags::mem_threadgroup);
        return schedule;
    }
};

/** One walker inserts or re-selects one node of a wave at a time, assigning itself the wave's
 *  offsets from the launch counter, and threads its picks for the back-link launch behind it. */
template <uint rung_index_, uint slots_, metal_build_walk_t walk_kind_>
inline void metal_build(constant metal_basic_arguments_t& arguments, metal_ragged_t storage, device uint* neighbors,
                        device uchar* arena, device atomic_uint* counters, metal_back_links_t links,
                        threadgroup typename metal_walkers<rung_index_>::schedule_t::shared_t* posted, uint group,
                        uint simdgroup, uint lane) {
    using walkers_t = metal_walkers<rung_index_>;
    using frontier_t = typename metal_frontier_of<slots_>::type;
    metal_basic_walk_t walk = metal_open(arguments, storage, neighbors, arena, counters,
                                         walkers_t::slice_of(group, simdgroup), simdgroup, lane);
    auto schedule = walkers_t::schedule(posted, storage, simdgroup, lane);
    auto build = [&](uint node_id) {
        // A wave's offsets fit the epoch's word, as a wave never outgrows the epoch's range.
        walk.epoch = arguments.epoch_base + (node_id - arguments.first_id) + 1;
        walk.random = metal_lane_generator(node_id, lane);
        if (walk_kind_ == metal_build_refine_k)
            metal_reselect_one<frontier_t>(walk, schedule, links, node_id, arguments.first_id,
                                           arguments.sampling_bound);
        else metal_insert_one<frontier_t>(walk, schedule, links, node_id, arguments.first_id, arguments.end_id);
    };
    metal_for_each_assigned(schedule, (device uint*)(counters + metal_assigned_k), arguments.first_id, arguments.end_id,
                            build);
    if (schedule.posted && simdgroup != 0) return;
    metal_flush(walk);
}

/*  The entry points, one instantiation per kit rung the prelude counts and register rung the
 *  host's ladder names: 2, 4, 8 and 16 slots of 32 lanes, and slot zero's arena frontier past the
 *  widest, each under the name the host looks it up by. */

/** One walker per node of a wave on the register rung of @p slots_, inserting it. */
template <uint rung_index_, uint slots_>
kernel void metal_insert_kernel(constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
                                device uchar const* values [[buffer(metal_basic_values_buffer_k)]],
                                device ulong const* offsets [[buffer(metal_basic_offsets_buffer_k)]],
                                device uint* neighbors [[buffer(metal_basic_neighbors_buffer_k)]],
                                device uchar* arena [[buffer(metal_basic_arena_buffer_k)]],
                                device atomic_uint* counters [[buffer(metal_basic_counters_buffer_k)]],
                                device uint* link_heads [[buffer(metal_basic_link_heads_buffer_k)]],
                                device uint* link_next [[buffer(metal_basic_link_next_buffer_k)]],
                                device uint* touched_ids [[buffer(metal_basic_touched_ids_buffer_k)]],
                                device atomic_uint* touched_count [[buffer(metal_basic_touched_count_buffer_k)]],
                                device uchar const* parameters [[buffer(metal_basic_parameters_buffer_k)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup typename metal_walkers<rung_index_>::schedule_t::shared_t posted;
    metal_build<rung_index_, slots_, metal_build_insert_k>(
        arguments, {values, offsets, parameters, arguments.metric}, neighbors, arena, counters,
        {link_heads, link_next, touched_ids, touched_count}, &posted, group, simdgroup, lane);
}

/** One walker per node of a wave on the register rung of @p slots_, re-selecting its row once the
 *  wave's back-links merged. */
template <uint rung_index_, uint slots_>
kernel void metal_refine_kernel(constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
                                device uchar const* values [[buffer(metal_basic_values_buffer_k)]],
                                device ulong const* offsets [[buffer(metal_basic_offsets_buffer_k)]],
                                device uint* neighbors [[buffer(metal_basic_neighbors_buffer_k)]],
                                device uchar* arena [[buffer(metal_basic_arena_buffer_k)]],
                                device atomic_uint* counters [[buffer(metal_basic_counters_buffer_k)]],
                                device uint* link_heads [[buffer(metal_basic_link_heads_buffer_k)]],
                                device uint* link_next [[buffer(metal_basic_link_next_buffer_k)]],
                                device uint* touched_ids [[buffer(metal_basic_touched_ids_buffer_k)]],
                                device atomic_uint* touched_count [[buffer(metal_basic_touched_count_buffer_k)]],
                                device uchar const* parameters [[buffer(metal_basic_parameters_buffer_k)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup typename metal_walkers<rung_index_>::schedule_t::shared_t posted;
    metal_build<rung_index_, slots_, metal_build_refine_k>(
        arguments, {values, offsets, parameters, arguments.metric}, neighbors, arena, counters,
        {link_heads, link_next, touched_ids, touched_count}, &posted, group, simdgroup, lane);
}

/** One walker back-links one node the walk launch before it touched, assigning itself the touched
 *  list's offsets from the launch counter, and resets the node's chain for the next walk launch. */
template <uint rung_index_>
kernel void metal_back_link_kernel(constant metal_basic_arguments_t& arguments
                                   [[buffer(metal_basic_arguments_buffer_k)]],
                                   device uchar const* values [[buffer(metal_basic_values_buffer_k)]],
                                   device ulong const* offsets [[buffer(metal_basic_offsets_buffer_k)]],
                                   device uint* neighbors [[buffer(metal_basic_neighbors_buffer_k)]],
                                   device uchar* arena [[buffer(metal_basic_arena_buffer_k)]],
                                   device atomic_uint* counters [[buffer(metal_basic_counters_buffer_k)]],
                                   device uint* link_heads [[buffer(metal_basic_link_heads_buffer_k)]],
                                   device uint* link_next [[buffer(metal_basic_link_next_buffer_k)]],
                                   device uint* touched_ids [[buffer(metal_basic_touched_ids_buffer_k)]],
                                   device atomic_uint* touched_count [[buffer(metal_basic_touched_count_buffer_k)]],
                                   device uchar const* parameters [[buffer(metal_basic_parameters_buffer_k)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint simdgroup [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    using walkers_t = metal_walkers<rung_index_>;
    threadgroup typename walkers_t::schedule_t::shared_t posted;
    metal_ragged_t const storage {values, offsets, parameters, arguments.metric};
    metal_back_links_t const links {link_heads, link_next, touched_ids, touched_count};
    metal_basic_walk_t walk = metal_open(arguments, storage, neighbors, arena, counters,
                                         walkers_t::slice_of(group, simdgroup), simdgroup, lane);
    auto schedule = walkers_t::schedule(&posted, storage, simdgroup, lane);
    uint const touched = atomic_load_explicit(touched_count, memory_order_relaxed);
    auto back_link = [&](uint offset) {
        uint const node_id = touched_ids[offset];
        metal_back_link_one(walk, schedule, links, node_id, arguments.first_id);
        if (lane == 0) link_heads[node_id] = metal_unknown_k;
    };
    metal_for_each_assigned(schedule, (device uint*)(counters + metal_assigned_k), 0, touched, back_link);
    if (schedule.posted && simdgroup != 0) return;
    metal_flush(walk);
}

/** One walker per query on the register rung of @p slots_, the arena frontier at zero; walkers take
 *  queries from the batch counter, as builders take entries. */
template <uint rung_index_, uint slots_>
kernel void metal_search_kernel(constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
                                device uchar const* values [[buffer(metal_basic_values_buffer_k)]],
                                device ulong const* offsets [[buffer(metal_basic_offsets_buffer_k)]],
                                device uint* neighbors [[buffer(metal_basic_neighbors_buffer_k)]],
                                device uchar const* queries [[buffer(metal_basic_staging_buffer_k)]],
                                device uchar* arena [[buffer(metal_basic_arena_buffer_k)]],
                                device atomic_uint* counters [[buffer(metal_basic_counters_buffer_k)]],
                                device ulong const* query_offsets [[buffer(metal_basic_query_offsets_buffer_k)]],
                                device uint* ids [[buffer(metal_basic_ids_buffer_k)]],
                                device float* distances [[buffer(metal_basic_distances_buffer_k)]],
                                device uchar const* parameters [[buffer(metal_basic_parameters_buffer_k)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    using walkers_t = metal_walkers<rung_index_>;
    threadgroup typename walkers_t::schedule_t::shared_t posted;
    metal_ragged_t const storage {values, offsets, parameters, arguments.metric};
    metal_basic_walk_t walk = metal_open(arguments, storage, neighbors, arena, counters,
                                         walkers_t::slice_of(group, simdgroup), simdgroup, lane);
    auto schedule = walkers_t::schedule(&posted, storage, simdgroup, lane);
    auto search = [&](uint query_index) {
        walk.epoch = arguments.epoch_base + query_index + 1;
        walk.random = metal_lane_generator(query_index, lane);
        metal_walk_one<typename metal_frontier_of<slots_>::type>(walk, schedule, arguments, queries, query_offsets, ids,
                                                                 distances, query_index);
    };
    metal_for_each_assigned(schedule, (device uint*)(counters + metal_assigned_k), 0, arguments.queries_count, search);
    if (schedule.posted && simdgroup != 0) return;
    metal_flush(walk);
}

/** One walker files one entry of a gather's runs at a time under its nearest hub, assigning itself
 *  the runs' positions from the round's claim word. */
template <uint rung_index_>
kernel void metal_gather_kernel(constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
                                device uchar const* values [[buffer(metal_basic_values_buffer_k)]],
                                device ulong const* offsets [[buffer(metal_basic_offsets_buffer_k)]],
                                device uint* neighbors [[buffer(metal_basic_neighbors_buffer_k)]],
                                device uchar* arena [[buffer(metal_basic_arena_buffer_k)]],
                                device atomic_uint* counters [[buffer(metal_basic_counters_buffer_k)]],
                                device uint* round_counters [[buffer(metal_basic_round_counters_buffer_k)]],
                                device ushort* entry_neighborhoods [[buffer(metal_basic_entry_neighborhoods_buffer_k)]],
                                device uint const* neighborhood_entries
                                [[buffer(metal_basic_neighborhood_entries_buffer_k)]],
                                device uint const* hub_nodes [[buffer(metal_basic_hub_nodes_buffer_k)]],
                                device uint4 const* runs [[buffer(metal_basic_runs_buffer_k)]],
                                device uint const* run_entries [[buffer(metal_basic_run_items_buffer_k)]],
                                device uchar const* parameters [[buffer(metal_basic_parameters_buffer_k)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    using walkers_t = metal_walkers<rung_index_>;
    threadgroup typename walkers_t::schedule_t::shared_t posted;
    metal_ragged_t const storage {values, offsets, parameters, arguments.metric};
    metal_round_t const round {entry_neighborhoods, neighborhood_entries, nullptr, hub_nodes,     runs,
                               run_entries,         arguments.runs_count, 0,       arguments.pass};
    metal_basic_walk_t walk = metal_open(arguments, storage, neighbors, arena, counters,
                                         walkers_t::slice_of(group, simdgroup), simdgroup, lane);
    auto schedule = walkers_t::schedule(&posted, storage, simdgroup, lane);
    auto gather = [&](uint position) { metal_gather_one(walk, schedule, round, position); };
    metal_for_each_assigned(schedule, round_counters + metal_round_claimed_k, 0, run_entries[arguments.runs_count],
                            gather);
    if (schedule.posted && simdgroup != 0) return;
    metal_flush(walk);
}

/** One walker re-stitches one entry of the neighborhoods at a time, assigning itself their
 *  positions from the round's claim word; an entry sits in one neighborhood and its walker alone
 *  writes its row, so no row is locked, and the links added and dropped and the rows changed add to
 *  the round's counters once per walker. */
template <uint rung_index_>
kernel void metal_restitch_kernel(
    constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
    device uchar const* values [[buffer(metal_basic_values_buffer_k)]],
    device ulong const* offsets [[buffer(metal_basic_offsets_buffer_k)]],
    device uint* neighbors [[buffer(metal_basic_neighbors_buffer_k)]],
    device uchar* arena [[buffer(metal_basic_arena_buffer_k)]],
    device atomic_uint* counters [[buffer(metal_basic_counters_buffer_k)]],
    device uint* round_counters [[buffer(metal_basic_round_counters_buffer_k)]],
    device uchar* depths_from_root [[buffer(metal_depths_from_root_buffer_k)]],
    device uchar* depths_to_root [[buffer(metal_depths_to_root_buffer_k)]],
    device uint* in_links [[buffer(metal_in_links_buffer_k)]],
    device uint* shallower_in_links [[buffer(metal_shallower_in_links_buffer_k)]],
    device uint* reached [[buffer(metal_reached_buffer_k)]],
    device uint* reaches_root [[buffer(metal_reaches_root_buffer_k)]],
    device uint* two_ways_to_root [[buffer(metal_two_ways_to_root_buffer_k)]],
    device uint* changed [[buffer(metal_basic_changed_buffer_k)]],
    device uint const* neighborhood_entries [[buffer(metal_basic_neighborhood_entries_buffer_k)]],
    device uint const* neighborhood_offsets [[buffer(metal_basic_neighborhood_offsets_buffer_k)]],
    device uchar const* parameters [[buffer(metal_basic_parameters_buffer_k)]],
    uint group [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    using walkers_t = metal_walkers<rung_index_>;
    threadgroup typename walkers_t::schedule_t::shared_t posted;
    metal_ragged_t const storage {values, offsets, parameters, arguments.metric};
    metal_round_t const round {
        nullptr, neighborhood_entries, neighborhood_offsets, nullptr, nullptr, nullptr, 0, arguments.neighborhoods, 0};
    metal_health_t health;
    health.neighbors = neighbors, health.counters = round_counters;
    health.depths_from_root = depths_from_root, health.depths_to_root = depths_to_root;
    health.in_links = in_links, health.shallower_in_links = shallower_in_links;
    health.reached = reached, health.reaches_root = reaches_root, health.two_ways_to_root = two_ways_to_root;
    health.rows_count = arguments.end_id, health.connectivity = arguments.connectivity;
    metal_basic_walk_t walk = metal_open(arguments, storage, neighbors, arena, counters,
                                         walkers_t::slice_of(group, simdgroup), simdgroup, lane);
    auto schedule = walkers_t::schedule(&posted, storage, simdgroup, lane);
    metal_restitch_tally_t tally {0, 0, 0};
    auto restitch = [&](uint position) { metal_restitch_one(walk, schedule, health, changed, round, position, tally); };
    metal_for_each_assigned(schedule, round_counters + metal_round_claimed_k, 0,
                            neighborhood_offsets[arguments.neighborhoods], restitch);
    if (schedule.posted && simdgroup != 0) return;
    if (lane == 0) {
        metal_tally_add(round_counters, metal_round_added_k, tally.added);
        metal_tally_add(round_counters, metal_round_dropped_k, tally.dropped);
        if (tally.changed) metal_warp_t::fetch_add_relaxed(round_counters + metal_round_changed_k, tally.changed);
    }
    metal_flush(walk);
}

/** Copies one run of a load's saved values into the tape at its planned starts, a byte a thread,
 *  as every kit stores them: saved and stored alike, back to back. */
[[host_name("metal_load_values_kernel")]] kernel void metal_load_values_kernel(
    constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
    device uchar* tape [[buffer(metal_basic_values_buffer_k)]],
    device uchar const* saved [[buffer(metal_basic_saved_buffer_k)]],
    device ulong const* stored_offsets [[buffer(metal_basic_stored_offsets_buffer_k)]],
    uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    ulong const first = stored_offsets[0], bytes = stored_offsets[arguments.count] - first;
    for (ulong byte = position; byte < bytes; byte += threads) tape[first + byte] = saved[byte];
}

/** Raises the load's span word to the widest stored entry of the run, rounded up to the 16 bytes a
 *  span takes, each simdgroup folding its lanes' spans before one atomic maximum. */
[[host_name("metal_size_spans_kernel")]] kernel void metal_size_spans_kernel(
    constant metal_basic_arguments_t& arguments [[buffer(metal_basic_arguments_buffer_k)]],
    device ulong const* stored_offsets [[buffer(metal_basic_stored_offsets_buffer_k)]],
    device atomic_uint* span_words [[buffer(metal_basic_span_words_buffer_k)]],
    uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint widest = 0;
    for (uint node = position; node < arguments.count; node += threads)
        widest = max(widest, (uint(stored_offsets[node + 1] - stored_offsets[node]) + 15u) & ~15u);
    widest = simd_max(widest);
    if (lane == 0 && widest) atomic_fetch_max_explicit(span_words, widest, memory_order_relaxed);
}

#if !defined(USEARCH_METAL_KIT_RUNGS) || USEARCH_METAL_KIT_RUNGS < 1 || USEARCH_METAL_KIT_RUNGS > 8
#error "The host's prelude counts the kit's rungs, one to eight, in USEARCH_METAL_KIT_RUNGS"
#endif

#define USEARCH_METAL_BUILD                                                                                            \
    constant metal_basic_arguments_t &, device uchar const *, device ulong const *, device uint *, device uchar *,     \
        device atomic_uint *, device uint *, device uint *, device uint *, device atomic_uint *, device uchar const *, \
        uint, uint, uint
#define USEARCH_METAL_SEARCH                                                                             \
    constant metal_basic_arguments_t &, device uchar const *, device ulong const *, device uint *,       \
        device uchar const *, device uchar *, device atomic_uint *, device ulong const *, device uint *, \
        device float *, device uchar const *, uint, uint, uint
#define USEARCH_METAL_GATHER                                                                                       \
    constant metal_basic_arguments_t &, device uchar const *, device ulong const *, device uint *, device uchar *, \
        device atomic_uint *, device uint *, device ushort *, device uint const *, device uint const *,            \
        device uint4 const *, device uint const *, device uchar const *, uint, uint, uint
#define USEARCH_METAL_RESTITCH                                                                                     \
    constant metal_basic_arguments_t &, device uchar const *, device ulong const *, device uint *, device uchar *, \
        device atomic_uint *, device uint *, device uchar *, device uchar *, device uint *, device uint *,         \
        device uint *, device uint *, device uint *, device uint *, device uint const *, device uint const *,      \
        device uchar const *, uint, uint, uint

template [[host_name("metal_insert_kernel_0_0")]] kernel void metal_insert_kernel<0, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_0_2")]] kernel void metal_insert_kernel<0, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_0_4")]] kernel void metal_insert_kernel<0, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_0_8")]] kernel void metal_insert_kernel<0, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_0_16")]] kernel void metal_insert_kernel<0, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_0_0")]] kernel void metal_refine_kernel<0, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_0_2")]] kernel void metal_refine_kernel<0, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_0_4")]] kernel void metal_refine_kernel<0, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_0_8")]] kernel void metal_refine_kernel<0, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_0_16")]] kernel void metal_refine_kernel<0, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_0_0")]] kernel void metal_search_kernel<0, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_0_2")]] kernel void metal_search_kernel<0, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_0_4")]] kernel void metal_search_kernel<0, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_0_8")]] kernel void metal_search_kernel<0, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_0_16")]] kernel void metal_search_kernel<0, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_0")]] kernel void metal_back_link_kernel<0>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_0")]] kernel void metal_gather_kernel<0>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_0")]] kernel void metal_restitch_kernel<0>(USEARCH_METAL_RESTITCH);

#if USEARCH_METAL_KIT_RUNGS > 1
template [[host_name("metal_insert_kernel_1_0")]] kernel void metal_insert_kernel<1, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_1_2")]] kernel void metal_insert_kernel<1, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_1_4")]] kernel void metal_insert_kernel<1, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_1_8")]] kernel void metal_insert_kernel<1, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_1_16")]] kernel void metal_insert_kernel<1, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_1_0")]] kernel void metal_refine_kernel<1, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_1_2")]] kernel void metal_refine_kernel<1, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_1_4")]] kernel void metal_refine_kernel<1, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_1_8")]] kernel void metal_refine_kernel<1, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_1_16")]] kernel void metal_refine_kernel<1, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_1_0")]] kernel void metal_search_kernel<1, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_1_2")]] kernel void metal_search_kernel<1, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_1_4")]] kernel void metal_search_kernel<1, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_1_8")]] kernel void metal_search_kernel<1, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_1_16")]] kernel void metal_search_kernel<1, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_1")]] kernel void metal_back_link_kernel<1>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_1")]] kernel void metal_gather_kernel<1>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_1")]] kernel void metal_restitch_kernel<1>(USEARCH_METAL_RESTITCH);
#endif

#if USEARCH_METAL_KIT_RUNGS > 2
template [[host_name("metal_insert_kernel_2_0")]] kernel void metal_insert_kernel<2, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_2_2")]] kernel void metal_insert_kernel<2, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_2_4")]] kernel void metal_insert_kernel<2, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_2_8")]] kernel void metal_insert_kernel<2, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_2_16")]] kernel void metal_insert_kernel<2, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_2_0")]] kernel void metal_refine_kernel<2, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_2_2")]] kernel void metal_refine_kernel<2, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_2_4")]] kernel void metal_refine_kernel<2, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_2_8")]] kernel void metal_refine_kernel<2, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_2_16")]] kernel void metal_refine_kernel<2, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_2_0")]] kernel void metal_search_kernel<2, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_2_2")]] kernel void metal_search_kernel<2, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_2_4")]] kernel void metal_search_kernel<2, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_2_8")]] kernel void metal_search_kernel<2, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_2_16")]] kernel void metal_search_kernel<2, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_2")]] kernel void metal_back_link_kernel<2>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_2")]] kernel void metal_gather_kernel<2>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_2")]] kernel void metal_restitch_kernel<2>(USEARCH_METAL_RESTITCH);
#endif

#if USEARCH_METAL_KIT_RUNGS > 3
template [[host_name("metal_insert_kernel_3_0")]] kernel void metal_insert_kernel<3, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_3_2")]] kernel void metal_insert_kernel<3, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_3_4")]] kernel void metal_insert_kernel<3, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_3_8")]] kernel void metal_insert_kernel<3, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_3_16")]] kernel void metal_insert_kernel<3, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_3_0")]] kernel void metal_refine_kernel<3, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_3_2")]] kernel void metal_refine_kernel<3, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_3_4")]] kernel void metal_refine_kernel<3, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_3_8")]] kernel void metal_refine_kernel<3, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_3_16")]] kernel void metal_refine_kernel<3, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_3_0")]] kernel void metal_search_kernel<3, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_3_2")]] kernel void metal_search_kernel<3, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_3_4")]] kernel void metal_search_kernel<3, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_3_8")]] kernel void metal_search_kernel<3, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_3_16")]] kernel void metal_search_kernel<3, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_3")]] kernel void metal_back_link_kernel<3>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_3")]] kernel void metal_gather_kernel<3>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_3")]] kernel void metal_restitch_kernel<3>(USEARCH_METAL_RESTITCH);
#endif

#if USEARCH_METAL_KIT_RUNGS > 4
template [[host_name("metal_insert_kernel_4_0")]] kernel void metal_insert_kernel<4, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_4_2")]] kernel void metal_insert_kernel<4, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_4_4")]] kernel void metal_insert_kernel<4, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_4_8")]] kernel void metal_insert_kernel<4, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_4_16")]] kernel void metal_insert_kernel<4, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_4_0")]] kernel void metal_refine_kernel<4, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_4_2")]] kernel void metal_refine_kernel<4, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_4_4")]] kernel void metal_refine_kernel<4, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_4_8")]] kernel void metal_refine_kernel<4, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_4_16")]] kernel void metal_refine_kernel<4, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_4_0")]] kernel void metal_search_kernel<4, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_4_2")]] kernel void metal_search_kernel<4, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_4_4")]] kernel void metal_search_kernel<4, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_4_8")]] kernel void metal_search_kernel<4, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_4_16")]] kernel void metal_search_kernel<4, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_4")]] kernel void metal_back_link_kernel<4>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_4")]] kernel void metal_gather_kernel<4>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_4")]] kernel void metal_restitch_kernel<4>(USEARCH_METAL_RESTITCH);
#endif

#if USEARCH_METAL_KIT_RUNGS > 5
template [[host_name("metal_insert_kernel_5_0")]] kernel void metal_insert_kernel<5, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_5_2")]] kernel void metal_insert_kernel<5, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_5_4")]] kernel void metal_insert_kernel<5, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_5_8")]] kernel void metal_insert_kernel<5, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_5_16")]] kernel void metal_insert_kernel<5, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_5_0")]] kernel void metal_refine_kernel<5, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_5_2")]] kernel void metal_refine_kernel<5, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_5_4")]] kernel void metal_refine_kernel<5, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_5_8")]] kernel void metal_refine_kernel<5, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_5_16")]] kernel void metal_refine_kernel<5, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_5_0")]] kernel void metal_search_kernel<5, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_5_2")]] kernel void metal_search_kernel<5, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_5_4")]] kernel void metal_search_kernel<5, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_5_8")]] kernel void metal_search_kernel<5, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_5_16")]] kernel void metal_search_kernel<5, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_5")]] kernel void metal_back_link_kernel<5>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_5")]] kernel void metal_gather_kernel<5>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_5")]] kernel void metal_restitch_kernel<5>(USEARCH_METAL_RESTITCH);
#endif

#if USEARCH_METAL_KIT_RUNGS > 6
template [[host_name("metal_insert_kernel_6_0")]] kernel void metal_insert_kernel<6, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_6_2")]] kernel void metal_insert_kernel<6, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_6_4")]] kernel void metal_insert_kernel<6, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_6_8")]] kernel void metal_insert_kernel<6, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_6_16")]] kernel void metal_insert_kernel<6, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_6_0")]] kernel void metal_refine_kernel<6, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_6_2")]] kernel void metal_refine_kernel<6, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_6_4")]] kernel void metal_refine_kernel<6, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_6_8")]] kernel void metal_refine_kernel<6, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_6_16")]] kernel void metal_refine_kernel<6, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_6_0")]] kernel void metal_search_kernel<6, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_6_2")]] kernel void metal_search_kernel<6, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_6_4")]] kernel void metal_search_kernel<6, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_6_8")]] kernel void metal_search_kernel<6, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_6_16")]] kernel void metal_search_kernel<6, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_6")]] kernel void metal_back_link_kernel<6>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_6")]] kernel void metal_gather_kernel<6>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_6")]] kernel void metal_restitch_kernel<6>(USEARCH_METAL_RESTITCH);
#endif

#if USEARCH_METAL_KIT_RUNGS > 7
template [[host_name("metal_insert_kernel_7_0")]] kernel void metal_insert_kernel<7, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_7_2")]] kernel void metal_insert_kernel<7, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_7_4")]] kernel void metal_insert_kernel<7, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_7_8")]] kernel void metal_insert_kernel<7, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_insert_kernel_7_16")]] kernel void metal_insert_kernel<7, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_7_0")]] kernel void metal_refine_kernel<7, 0>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_7_2")]] kernel void metal_refine_kernel<7, 2>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_7_4")]] kernel void metal_refine_kernel<7, 4>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_7_8")]] kernel void metal_refine_kernel<7, 8>(USEARCH_METAL_BUILD);
template [[host_name("metal_refine_kernel_7_16")]] kernel void metal_refine_kernel<7, 16>(USEARCH_METAL_BUILD);
template [[host_name("metal_search_kernel_7_0")]] kernel void metal_search_kernel<7, 0>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_7_2")]] kernel void metal_search_kernel<7, 2>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_7_4")]] kernel void metal_search_kernel<7, 4>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_7_8")]] kernel void metal_search_kernel<7, 8>(USEARCH_METAL_SEARCH);
template [[host_name("metal_search_kernel_7_16")]] kernel void metal_search_kernel<7, 16>(USEARCH_METAL_SEARCH);
template [[host_name("metal_back_link_kernel_7")]] kernel void metal_back_link_kernel<7>(USEARCH_METAL_BUILD);
template [[host_name("metal_gather_neighborhoods_kernel_7")]] kernel void metal_gather_kernel<7>(USEARCH_METAL_GATHER);
template [[host_name("metal_restitch_kernel_7")]] kernel void metal_restitch_kernel<7>(USEARCH_METAL_RESTITCH);
#endif
#pragma endregion Kernels

} // namespace unum::usearch::basic

#endif // defined(__METAL_VERSION__)
