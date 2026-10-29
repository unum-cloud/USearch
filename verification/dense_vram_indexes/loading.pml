/**
 *  @file verification/dense_vram_indexes/loading.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A load beside one batch and a searcher: the load closes every shard's dispenser first,
 *      checks under the closure that no row is visible, and files each shard's keys before the shard
 *      plan's `finish`.
 *
 *  The loader closes both shards in shard order, refused while the batch is in flight on either,
 *  and checks that no row is visible. Its steps land a row and its key at each shard's watermark,
 *  the last shard's possibly short; then per shard `publish_keys`, the map's one writer, and the
 *  shard plan's `finish`, the turn, refused on a short shard. It reopens both, the last first.
 *
 *  - Once all are done, no row of the refused load is filed.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc load_closes=false: without the closure, the batch claims the row the load lands,
 *      and their keys collide
 *  @verify fail sc load_key_before_finish=false: after it, the searcher reads the loaded shard's row
 *      before its key
 *  @verify fail sc keys_once_covered=false: filing before the step tally is checked, a refused
 *      `finish` leaves the short shard's row filed, and the batch claiming it after the reopening
 *      files it a second time
 */
#define thread_count 3
#define location_count 11
#define history_depth 3
#define mutators 1
#include <dense_vram_indexes/protocol.pml>

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

// the closure, then per shard the key at the watermark and `load_plan_t::finish`, the turn; then the reopening
proctype loader(byte t) {
    byte shard, count, position;
    int seen, total, watermark, visible;
    bool whole;
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
    // the steps: each shard's row at its watermark, which stays where the check read it, with its
    // key, the last shard's step possibly short
    for (shard : 0 .. shards - 1) {
        atomic { take_row(t, shard, 0) };
        if
        :: load_key_before_finish -> store(t, key(shard, 0), order_relaxed, key_of(t))
        :: else
        fi
    };
    if
    :: true -> whole = true
    :: true -> whole = false
    fi;
    // finish: per shard the keys filed, then the turn, refused on the short shard, whose row the
    // dying plan gives back
    for (shard : 0 .. shards - 1) {
        if
        :: load_key_before_finish && (whole || shard != shards - 1 || !keys_once_covered) ->
            publish_key(t, shard, 0, position, seen)
        :: else
        fi;
        if
        :: !whole && shard == shards - 1 -> atomic { owner_of(shard, 0) = 0 }; goto refused
        :: else
        fi;
        take_turn(t, shard, 0, watermark);
        if
        :: !load_key_before_finish ->
            store(t, key(shard, 0), order_relaxed, key_of(t));
            publish_key(t, shard, 0, position, seen)
        :: else
        fi
    };
refused:
    if
    :: load_closes -> count = shards; reopen_shards(count)
    :: else
    fi;
contended:
    done++
}

init {
    atomic { run mutator(0); run searcher(1); run loader(2) };
    (done == 3);
    audit_keys()
}
