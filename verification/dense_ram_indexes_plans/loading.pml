/**
 *  @file verification/dense_ram_indexes_plans/loading.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The sharded load path: every shard's keys landed and filed before `finish`, beside a
 *      searcher.
 *
 *  The loader runs `plan_load` per shard on the still-empty index, never refused there; its steps
 *  land each shard's rows with their keys, and the last shard's may fall short. Its `finish` runs
 *  per shard: the plan's `publish_keys`, the shard map's one writer, then the plan's `finish`, which
 *  refuses a short shard and otherwise bumps the dispenser and releases the watermark.
 *
 *  - A search finds every key below the watermark it acquired, once.
 *  - Once both are done, no node of a refused plan is filed.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 keys_before_finish=false: after them, the searcher reads an unset key below
 *      the watermark it acquired
 *  @verify fail sc keys_once_covered=false: filing before the step tally is checked, a refused
 *      `finish` leaves the short shard's nodes filed above its watermark, where the next rows
 *      dispensed there would meet them
 */
#define thread_count 2
#define location_count 13
#define history_depth 2
#include <dense_ram_indexes_plans/protocol.pml>

byte done; // the threads finished

proctype loader(byte t) {
    byte shard, position;
    int visible[shards], dispensed_now, seen;
    bool whole[shards];
    // plan_load: the watermark, never refused on an idle index
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, visible[shard]);
        load(t, next_dispensed_id(shard), order_acquire, dispensed_now);
        assert(visible[shard] == dispensed_now)
    };
    // the steps, the last shard's possibly short, each landing its rows' keys
    for (shard : 0 .. shards - 1) {
        if
        :: true -> whole[shard] = true
        :: shard == shards - 1 -> whole[shard] = false
        fi;
        if
        :: keys_before_finish -> land_keys(t, shard, visible[shard], (whole[shard] -> row_limit : row_limit - 1))
        :: else
        fi
    };
    // finish: per shard the keys filed, then the plan's own `finish`, refused while a step fell short
    for (shard : 0 .. shards - 1) {
        if
        :: keys_before_finish && (whole[shard] || !keys_once_covered) ->
            publish_keys(t, shard, visible[shard], position, seen)
        :: else
        fi;
        if
        :: !whole[shard] -> goto refused
        :: else
        fi;
        add_no_return(t, next_dispensed_id(shard), order_release, row_limit);
        store(t, last_visible_id(shard), order_release, visible[shard] + row_limit);
        if
        :: !keys_before_finish ->
            land_keys(t, shard, visible[shard], row_limit);
            publish_keys(t, shard, visible[shard], position, seen)
        :: else
        fi
    };
refused:
    done++
}

proctype searcher(byte t) { search(t); done++ }

init {
    atomic { run loader(0); run searcher(1) };
    (done == 2);
    audit_keys()
}
