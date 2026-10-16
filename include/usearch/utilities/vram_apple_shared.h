/**
 *  @file include/usearch/utilities/vram_apple_shared.h
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The vocabulary every Apple traversal context and its Metal kernels share: the walk's
 *      fixed extents and the counter words a launch lands its telemetry in.
 *
 *  Written in the C++ that MSL shares with every host compiler, as plain enums: the host includes
 *  it, and the dense Apple index embeds it first among the Metal sources, which cannot include, so
 *  the two sides read one definition of every word they exchange. The counters are the Metal
 *  spelling of @c vram_telemetry_block_t, each tally split in two 32-bit words.
 */
#pragma once

namespace unum::usearch::utilities {

/** The fixed extents of every Metal walk. */
enum metal_extent_t : unsigned int {

    /** Simdgroups per threadgroup, one walk each. */
    metal_simdgroups_k = 4,

    /** Lanes per simdgroup, the width every warp collective spans. */
    metal_simdgroup_width_k = 32,

    /** Rows of the per-walk candidate cache; also bounds the diversity selection's prefix and the
     *  staging spans. */
    metal_cache_rows_k = 64,

};

/** The counter words a launch binds: the batch's assigned rows, the published rows, the changed
 *  nodes and the word an attend raises when a node has no value row, then the walks' telemetry,
 *  each tally a low and a high word, so a batch's totals outgrow 32 bits as on CUDA and HIP. */
enum metal_counter_t : unsigned int {
    metal_assigned_k = 0,
    metal_published_k = 1,
    metal_changed_k = 2,
    metal_missed_k = 3,
    metal_hops_k = 4,
    metal_scored_k = 6,
    metal_restarts_k = 8,
    metal_lock_attempts_k = 10,
    metal_lock_skips_k = 12,
    metal_reselections_k = 14,
    metal_locked_drops_k = 16,
    metal_saturations_k = 18,
    metal_tail_drops_k = 20,
    metal_counter_words_k = 24,
};

} // namespace unum::usearch::utilities
