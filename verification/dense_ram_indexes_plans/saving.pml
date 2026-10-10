/**
 *  @brief The monolith save: one `plan_save` per shard beside a planned batch and a searcher.
 *
 *  The batch lands one row on each shard as `insert_plan_t` does. The saver opens every shard's
 *  plan, then runs every shard's `finish`. A settled save copies no unset key below its watermark,
 *  and holds the batch on both shards or on neither: the batch dispenses every shard before it
 *  publishes any, so a shard the batch has not reached yet refuses its plan, and since every plan
 *  opens before any re-check, the re-checks hold every shard's watermark across one instant.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 plans_before_finish=false: `plan_save` opens every shard's plan before
 *      `finish` re-checks any; planning and finishing one shard at a time, the batch publishes
 *      between them, and a settled save holds its row on shard one without the one on shard zero
 */
#define thread_count 3
#define location_count 9
#define history_depth 3
#include "protocol.pml"

proctype batch_adder(byte t) { insert_batch(t) }

proctype saver(byte t) {
    byte shard;
    int visible[shards];
    for (shard : 0 .. shards - 1) {
        plan_shard(t, shard, visible[shard]);
        if
        :: !plans_before_finish -> finish_shard(t, shard, visible[shard])
        :: else
        fi
    };
    if
    :: plans_before_finish ->
        for (shard : 0 .. shards - 1) {
            finish_shard(t, shard, visible[shard])
        }
    :: else
    fi;
    assert(visible[0] == visible[1]);
contended:
    skip
}

proctype searcher(byte t) { search(t) }

init { atomic { run batch_adder(0); run saver(1); run searcher(2) } }
