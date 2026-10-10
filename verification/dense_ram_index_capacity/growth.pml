/**
 *  @brief A grower elected by its subtract beside a claimer that fits.
 *
 *  Thread 0 claims two rows, which grows the only chunk, and releases them; thread 1 claims one.
 *  `client.cpp` runs the same scenario under GenMC, with its race detector on the chunk directory.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail far deflation_order=order_relaxed: `release_capacity`'s subtract on
 *      `last_allocated_id_` releases, which no reference posts, and `capacity()` reads the release
 *      before the claim, so a seen release carries its deflation; relaxed, the subtract is posted
 *      under Intel's RAO-INT rule and still in flight when the release lands, and a claimer takes
 *      rows past the end of the only chunk
 */
#define thread_count 2
#define location_count 3
#define history_depth 8
#include "protocol.pml"

proctype grower(byte t) {
    int first, whole, old_remaining, reverted, seen_chunks, added;
    claim(t, 2, first);
    release_capacity(t, 2);
    landed(t)
}

proctype claimer(byte t) {
    int first, old_remaining, reverted, seen_chunks, added;
    claim(t, 1, first)
}

init { atomic { run grower(0); run claimer(1) } }
