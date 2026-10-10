/**
 *  @brief The monolith save: one `plan_save` per shard beside a batch adder and a searcher.
 *
 *  The batch adder lands one row on each shard. The saver runs `plan_save` per shard, then
 *  `finish` per shard: the acquire fence, the relaxed re-check, then the plain key copy of
 *  `keys[0..visible)` at the shard's base. A settled plan copies no unset key below its watermark,
 *  and under `sc` a plan is refused exactly when a row is between its dispense and its
 *  publication, or was published between the plan's two loads.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 whole_batches=true: the watermark is per shard and nothing joins the two,
 *      so a settled plan captures a batch's rows on shard zero without those on shard one; the
 *      header does not claim whole batches, and a plan that does fails under every model
 */
#define thread_count 3
#define location_count 8
#define history_depth 2
#include "protocol.pml"

proctype batch_adder(byte t) {
    int id;
    land_row(t, 0, id);
    land_row(t, 1, id)
}

proctype saver(byte t) {
    byte shard, id;
    int visible[shards], dispensed_now, again, copy;
    // plan_save per shard: the watermark, refused while an insert is mid-flight
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, visible[shard]);
        atomic {
            load(t, next_dispensed_id(shard), order_acquire, dispensed_now);
            assert(memory != sc || (visible[shard] != dispensed_now) ==
                   (newest_value(next_dispensed_id(shard)) != newest_value(last_visible_id(shard))
                    || newest_value(last_visible_id(shard)) != visible[shard]))
        };
        if
        :: visible[shard] != dispensed_now -> goto contended
        :: else
        fi
    };
    // finish per shard: the fence, the relaxed re-check, then the plain key copy at the shard's base
    for (shard : 0 .. shards - 1) {
        fence_acquire(t);
        load(t, last_visible_id(shard), order_relaxed, again);
        if
        :: again != visible[shard] -> goto contended
        :: else
        fi;
        load(t, next_dispensed_id(shard), order_relaxed, again);
        if
        :: again != visible[shard] -> goto contended
        :: else
        fi;
        id = 0;
        do
        :: id == visible[shard] -> break
        :: else -> load(t, key(shard, id), order_relaxed, copy); assert(copy != unset); id++
        od
    };
    if
    :: whole_batches -> assert(visible[0] == visible[1])
    :: else
    fi;
contended:
    skip
}

proctype searcher(byte t) { search(t) }

init { atomic { run batch_adder(0); run saver(1); run searcher(2) } }
