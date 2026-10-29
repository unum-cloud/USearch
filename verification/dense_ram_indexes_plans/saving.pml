/**
 *  @file verification/dense_ram_indexes_plans/saving.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The monolith save: one `plan_save` per shard beside a planned batch and a searcher.
 *
 *  The batch lands one row on each shard as `insert_plan_t` does, filing each shard's key while it
 *  holds that shard's turn. The saver opens every shard's plan, then runs every shard's `finish`;
 *  the searcher counts each shard's map below the watermark it acquired.
 *
 *  - A settled save copies no unset key below its watermark.
 *  - A settled save holds the batch on both shards or on neither.
 *  - A search finds every key below the watermark it acquired, once.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 plans_before_finish=false: planning and finishing one shard at a time, the
 *      batch publishes between them, and a settled save holds its row on shard one without the one
 *      on shard zero
 */
#define thread_count 3
#define location_count 13
#define history_depth 3
#include <dense_ram_indexes_plans/protocol.pml>

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
