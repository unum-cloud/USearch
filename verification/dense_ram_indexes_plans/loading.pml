/**
 *  @brief The sharded load path: every shard's keys before `finish`, beside a searcher.
 *
 *  The loader runs `plan_load` per shard on the still-empty index: it reads the shard's watermark
 *  and dispenser, never refused on an idle index, writes the shard's keys at the base, then bumps
 *  the dispenser and releases the watermark.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc,rc11 keys_before_finish=false: the load path lands every shard's keys before
 *      `finish` releases the dispenser and the watermark; after them, the searcher reads an unset
 *      key below the watermark it acquired
 */
#define thread_count 2
#define location_count 8
#define history_depth 2
#include "protocol.pml"

proctype loader(byte t) {
    byte shard;
    int visible, dispensed_now;
    for (shard : 0 .. shards - 1) {
        // plan_load: the watermark, never refused on an idle index
        load(t, last_visible_id(shard), order_acquire, visible);
        load(t, next_dispensed_id(shard), order_acquire, dispensed_now);
        assert(visible == dispensed_now);
        // the keys at the base, then the publication
        if
        :: keys_before_finish -> land_keys(t, shard, visible)
        :: else
        fi;
        add_no_return(t, next_dispensed_id(shard), order_release, row_limit);
        store(t, last_visible_id(shard), order_release, visible + row_limit);
        if
        :: !keys_before_finish -> land_keys(t, shard, visible)
        :: else
        fi
    };
    landed(t)
}

proctype searcher(byte t) { search(t) }

init { atomic { run loader(0); run searcher(1) } }
