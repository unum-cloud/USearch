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
 *  kernel, then an insert binds the back-link chains it threads, a refine and its merge the shadow
 *  matrix, the parts' first rows, the walkers' chains and the walked marks, and a search its
 *  outputs, its indexer heads and its keeps word; a gather binds its listed rows and its outputs, a
 *  count sum the counts, and a split's partition its node parts, proposals, two frontiers, the
 *  parts' sizes and round counts, the placed counts and the picked bitset. */
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

    metal_shadow_buffer_k = 7,
    metal_part_bases_buffer_k = 8,
    metal_chain_heads_buffer_k = 9,
    metal_chain_links_buffer_k = 10,
    metal_walked_buffer_k = 11,

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
    metal_picked_buffer_k = 14,
};

/** Every scalar a launch carries, bound at @c metal_arguments_buffer_k. */
struct metal_arguments_t {
    unsigned int row_fragments, lanes_per_row, steps_per_row, connectivity;
    unsigned int metric, has_norms, first_id, end_id;
    unsigned int sampling_bound, queries_count, count, epoch_base;
    unsigned int visited_slots, hash_shift, selection_nodes, slice_bytes;
    unsigned int heads_count, row_fragments_magic, use_node_ids, expansion;
    unsigned int tail_entries, parts, round, concurrent_walks;
    unsigned int filling_nodes, packed_bytes, part_capacity;
};

} // namespace unum::usearch::dense
