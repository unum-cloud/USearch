/**
 *  @brief `reserve` called twice on one index, the second time above the physical rows.
 *
 *  The regrower reserves two rows, then three. The second reservation's deficit is one row while
 *  two free rows sit on the counter.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc deficit_growth=false: `reserve` claims from the claimed rows up to the request,
 *      strictly more than the counter holds, so the subtract crosses zero, elects a grower, and the
 *      loop ends in one round; claiming the deficit alone, the second reservation fits in the free
 *      rows, is granted without a growth, released, and read back unchanged, and the rounds
 *      over-run
 */
#define thread_count 1
#define location_count 3
#define history_depth 7
#include "protocol.pml"

proctype regrower(byte t) {
    int claimed_rows, total, deficit, first;
    int whole, old_remaining, reverted, seen_chunks, added, unclaimed;
    byte rounds;
    reserve(t, 2, claimed_rows, total);
    capacity(t, claimed_rows, total);
    assert(total >= 2);
    reserve(t, 3, claimed_rows, total);
    capacity(t, claimed_rows, total);
    assert(total >= 3);
    landed(t)
}

init { atomic { run regrower(0) } }
