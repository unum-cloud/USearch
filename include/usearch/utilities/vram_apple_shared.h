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

    /** Lanes per simdgroup, the width every warp collective spans; every pipeline must report it
     *  as its @c threadExecutionWidth, or the device refuses it. */
    metal_simdgroup_width_k = 32,

    /** Slot loads one lane keeps in flight across a walk's round, the CUDA and HIP
     *  @c vram_max_gathers_in_flight_k. */
    metal_max_gathers_in_flight_k = 8,

};

/** Rows a walk expands per hop, the twin of the host's @c walk_width: @p concurrent_walks, the
 *  walks the hardware runs at once, divided among the walks that need them, at least one and at
 *  most @p filling_rows. */
constexpr unsigned int walk_width(unsigned int concurrent_walks, unsigned int live_walks, unsigned int unstarted_walks,
                                  unsigned int filling_rows) {
    unsigned int const started = unstarted_walks < concurrent_walks ? unstarted_walks : concurrent_walks;
    unsigned int const walks = live_walks < started ? started : live_walks;
    unsigned int const share = concurrent_walks / walks ? concurrent_walks / walks : 1;
    return share < filling_rows ? share : filling_rows;
}

/** The counter words a launch binds: the batch's assigned rows, the walks still running, the
 *  published rows, the changed nodes and the value-lookup miss word a walk raises at a node with no
 *  value row, then the walks' telemetry, each tally a low and a high word, so a batch's totals
 *  outgrow 32 bits as on CUDA and HIP. */
enum metal_counter_t : unsigned int {
    metal_assigned_k = 0,
    metal_live_k = 1,
    metal_published_k = 2,
    metal_changed_k = 3,
    metal_missed_k = 4,
    metal_hops_k = 5,
    metal_scored_k = 7,
    metal_restarts_k = 9,
    metal_lock_attempts_k = 11,
    metal_lock_skips_k = 13,
    metal_reselections_k = 15,
    metal_locked_drops_k = 17,
    metal_saturations_k = 19,
    metal_tail_compactions_k = 21,
    metal_tail_dropped_k = 23,

    /** The word count, padded to an even one so whatever follows a block in staging stays 8-byte
     *  aligned. */
    metal_counter_words_k = 26,
};

} // namespace unum::usearch::utilities
