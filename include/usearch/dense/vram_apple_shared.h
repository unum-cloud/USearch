/**
 *  @file include/usearch/dense/vram_apple_shared.h
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The vocabulary the dense Apple index and its Metal kernels share: the launch arguments,
 *      the metrics, the buffer slots and the arena spans of a walk.
 *
 *  Written in the C++ that MSL shares with every host compiler, as plain enums and one struct, and
 *  read after @c utilities/vram_apple_shared.h, whose extents and counter words it builds on: the
 *  host includes both in that order, and @c vram_apple_index.hpp embeds both in that order ahead of
 *  the Metal sources, which cannot include, so the two sides read one definition of every word.
 */
#pragma once

namespace unum::usearch::dense {

using namespace unum::usearch::utilities;

/** The width a row's fragments round to, which every Metal kit loads at once. */
enum metal_fragment_t : unsigned int {
    metal_fragment_bytes_k = 16,
};

/** Bytes of the batch spans a walk carves first in its scratch, the CUDA walk's shared
 *  @c vram_gathered: one batch's scored nodes and dots, a simdgroup's width each, its IDs, two
 *  simdgroup widths for a scored batch and the next one's carry, and their sum. */
enum metal_span_bytes_t : unsigned int {
    metal_scored_span_bytes_k = metal_simdgroup_width_k * 8,
    metal_ids_span_bytes_k = metal_simdgroup_width_k * 4 * 2,
    metal_dots_span_bytes_k = metal_simdgroup_width_k * 4,
    metal_batch_spans_bytes_k = metal_scored_span_bytes_k + metal_ids_span_bytes_k + metal_dots_span_bytes_k,
};

/** The metrics the host lowers its kernel kind onto: the walk ranks Euclidean by its square. */
enum metal_metric_t : unsigned int {
    metal_dot_k = 0,
    metal_angular_k = 1,
    metal_sqeuclidean_k = 2,
    metal_euclidean_k = 3,
    metal_hamming_k = 4,
    metal_jaccard_k = 5,
};

/** The buffer slots the kernels take their bindings at: the first seven are common to every walking
 *  kernel, then an insert binds the back-link chains it threads, and a search its outputs, its
 *  indexer heads and its keeps word; a gather binds its listed rows and its outputs, a count sum
 *  the counts, and a split's partition its node parts, proposals, two frontiers, the parts' sizes
 *  and round counts and the placed counts. A health check and a round bind the round's counters and
 *  the check's columns, then the islands' columns, or the round's columns, which alias them. */
enum metal_buffer_index_t : unsigned int {
    metal_arguments_buffer_k = 0,
    metal_vectors_buffer_k = 1,
    metal_norms_buffer_k = 2,
    metal_neighbors_buffer_k = 3,
    metal_staging_buffer_k = 4,
    metal_arena_buffer_k = 5,
    metal_counters_buffer_k = 6,

    metal_link_heads_buffer_k = 7,
    metal_link_next_buffer_k = 8,
    metal_touched_ids_buffer_k = 9,
    metal_touched_count_buffer_k = 10,

    metal_ids_buffer_k = 7,
    metal_distances_buffer_k = 8,
    metal_products_buffer_k = 9,
    metal_scored_counts_buffer_k = 10,
    metal_heads_buffer_k = 11,
    metal_weights_buffer_k = 12,
    metal_keeps_buffer_k = 13,

    metal_gathered_rows_buffer_k = 7,
    metal_gathered_outputs_buffer_k = 8,
    metal_counts_buffer_k = 7,

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

    metal_entry_neighborhoods_buffer_k = 16,
    metal_neighborhood_entries_buffer_k = 17,
    metal_neighborhood_offsets_buffer_k = 18,
    metal_hub_nodes_buffer_k = 19,
    metal_runs_buffer_k = 20,
    metal_run_items_buffer_k = 21,
    metal_stitch_buffer_k = 22,
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

/** The threadgroup memory a gather or a restitch stages rows and norms in: the 32 KB every Apple
 *  family grants a threadgroup, less what the kernel keeps beside them. A neighborhood or a run of
 *  hubs that outgrows it reads its rows in place, as the walks always do. */
enum metal_round_staging_t : unsigned int {
    metal_threadgroup_bytes_k = 32768,
    metal_round_staged_bytes_k = metal_threadgroup_bytes_k - 4096,
};

/** Every scalar a launch carries, bound at @c metal_arguments_buffer_k. */
struct metal_arguments_t {
    unsigned int row_fragments, lanes_per_row, steps_per_row, connectivity;
    unsigned int metric, has_norms, first_id, end_id;
    unsigned int sampling_bound, queries_count, count, epoch_base;
    unsigned int visited_slots, hash_shift, selection_nodes, slice_bytes;
    unsigned int heads_count, row_fragments_magic, expansion, tail_entries;
    unsigned int round, concurrent_walks, filling_nodes, packed_bytes;
    unsigned int part_capacity, depth, islands, pass;
    unsigned int runs_count, neighborhoods;
};

} // namespace unum::usearch::dense
