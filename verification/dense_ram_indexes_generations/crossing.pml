/**
 *  @file verification/dense_ram_indexes_generations/crossing.pml
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief A one-row planned batch and a single-row insert crossing the bound at once, the insert's
 *      build free to fail, beside a reader touching every shard the count publishes.
 *
 *  The first generation's shard is full, so the planner's `room()` is zero and the insert's probe
 *  is refused, and both open the next generation from the same count. One takes the opening bit and
 *  builds; the other waits on the count, then claims in the new shard. When the insert's build
 *  fails, it puts the count back and the insert is refused, and the planner takes the bit and
 *  builds instead.
 *
 *  - Every insert not refused lands its row, and the count ends past the new generation.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 opening_cap=false: the cap is not what ends the wait: a monitored waiter
 *      re-checks the count after arming and wakes on the store that publishes or restores it
 *  @verify fail sc opening_bit=false: with a plain load of the count instead, both adders read the
 *      generation unopened and both construct its shard
 *  @verify fail rc11 publish_order=order_relaxed: relaxed, the waiter or the reader acquires the new
 *      count and touches a shard whose construction it has not seen
 *  @verify fail rc11 failure_order=order_relaxed: relaxed, the planner claims on a shard whose
 *      construction it has not seen
 *  @verify stuck sc restore_count=false: leaving the bit set, the waiter re-reads the same word
 *      forever and no generation opens
 *  @verify stuck sc opening_cap=false count_wakes=false: an uncapped wait that no store to the count
 *      wakes leaves the waiter armed after the generation opened
 *  @verify stuck sc opening_cap=false recheck=false: an uncapped wait that arms without re-reading
 *      the count waits for a store that already happened
 */
#define thread_count 3
#define location_count 6
#define history_depth 5
#include <dense_ram_indexes_generations/protocol.pml>

proctype planner(byte t) {
    int count, seen, watermark, sequence;
    byte split, shard;
    bool won, granted, opened;
    plan_batch(t, 1)
}

proctype inserter(byte t) {
    int count, seen, id, watermark;
    byte shard;
    bool won, granted, opened;
    insert_row(t, true)
}

// a snapshot's shards: every one `size()` counts, touched while a generation may be opening
proctype reader(byte t) {
    int count, seen;
    byte shard;
    published_shards(t);
    do
    :: shard == count -> break
    :: else -> touch(t, shard); shard++
    od
}

// once both adders are done: the generation open and built once, and every claim landed
proctype checker() {
    (done == 2);
    assert(newest_value(shards_count) == 2 && constructed[1]);
    assert(claimed[1] == dispensed[1] && newest_value(last_visible_id(1)) == dispensed[1])
}

init {
    atomic {
        seed(shards_count, 1);
        seed(built(0), 1);
        seed(last_visible_id(0), row_limit);
        claimed[0] = row_limit;
        dispensed[0] = row_limit;
        constructed[0] = true;
        run planner(0);
        run inserter(1);
        run reader(2);
        run checker()
    }
}
