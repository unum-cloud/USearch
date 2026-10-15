/**
 *  @brief A load beside one batch and a searcher: the load closes every shard's dispenser first,
 *      checks under the closure that no row is visible, and writes each shard's key before the shard
 *      plan's `finish`.
 *
 *  The loader closes both shards in shard order, refused while the batch is in flight on either,
 *  checks that no row is visible, appends at each shard's watermark: the key, then the shard plan's
 *  `finish`, the turn; and reopens both shards, the last first. A batch meeting a closed shard is
 *  refused, so it never claims a row the load lands.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc load_closes=false: the load closes every shard before it appends at the
 *      watermarks; without the closure, the batch claims the row the load lands, and their keys
 *      collide
 *  @verify fail sc load_key_before_finish=false: the load writes each shard's key before its plan
 *      finishes, the order the RAM twin had already corrected; after it, the searcher reads the
 *      loaded shard's row before its key
 */
#define thread_count 3
#define location_count 7
#define history_depth 3
#define mutators 1
#include "protocol.pml"

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

// the closure, then per shard the key at the watermark and `load_plan_t::finish`, the turn; then the reopening
proctype loader(byte t) {
    byte shard, count;
    int seen, total, first_id, watermark, visible;
    if
    :: load_closes ->
        close_shards(t, count, visible);
        if
        :: count != shards -> goto contended
        :: else
        fi
    :: else
    fi;
    // empty under the closure, as a load requires: `gate_load_` reads `size()`
    load(t, last_visible_id(0), order_acquire, seen);
    load(t, last_visible_id(1), order_acquire, total);
    if
    :: seen + total != 0 -> goto refused
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, first_id);
        atomic { take_row(t, shard, first_id) };
        if
        :: load_key_before_finish ->
            store(t, key(shard, first_id), order_relaxed, key_of(t));
            take_turn(t, shard, first_id, watermark)
        :: else ->
            take_turn(t, shard, first_id, watermark);
            store(t, key(shard, first_id), order_relaxed, key_of(t))
        fi
    };
refused:
    if
    :: load_closes -> count = shards; reopen_shards(count)
    :: else
    fi;
contended:
    skip
}

init { atomic { run mutator(0); run searcher(1); run loader(2) } }
