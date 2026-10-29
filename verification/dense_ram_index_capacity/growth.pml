/**
 *  @file verification/dense_ram_index_capacity/growth.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A grower elected by its subtract beside a claimer that fits.
 *
 *  Thread 0 claims two rows, which grows the only chunk, and releases them; thread 1 claims one.
 *  `client.cpp` runs the same scenario under GenMC, with its race detector on the chunk directory.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc keyed_cover=false: the grower covers the rows alone, and a claimer's rows lie past
 *      the chunks of the key column
 *  @verify fail far deflation_order=order_relaxed: relaxed, the subtract is posted under Intel's
 *      RAO-INT rule and still in flight when the release lands, and a claimer takes rows past the
 *      end of the only chunk
 */
#define thread_count 2
#define location_count 6
#define history_depth 8
#include <dense_ram_index_capacity/protocol.pml>

proctype grower(byte t) {
    byte column;
    int first, whole, old_remaining, reverted, seen_chunks, added;
    claim(t, 2, first);
    release_capacity(t, 2);
    landed(t)
}

proctype claimer(byte t) {
    byte column;
    int first, old_remaining, reverted, seen_chunks, added;
    claim(t, 1, first)
}

init { atomic { run grower(0); run claimer(1) } }
