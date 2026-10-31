/**
 *  @file verification/dense_vram_indexes/loading.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A load beside one batch and a searcher: the load closes every shard's dispenser first,
 *      checks under the closure that no row is visible, files each shard's keys, and in each shard
 *      plan's `finish` refuses a short shard before it awaits the column copy and turns.
 *
 *  The loader closes both shards in shard order, refused while the batch is in flight on either,
 *  and checks that no row is visible. Its steps take a row and its key at each shard's watermark and
 *  queue the row's columns onto the shard's storage on its stream, which records an event after
 *  them; the last shard's step is possibly short. Then per shard `publish_keys`, the map's one
 *  writer, which files nothing on a short shard, and `finish`: the refusal of a short shard first,
 *  then the wait on the copy's event, then the turn. It reopens both.
 *
 *  - Once all are done, no row of the refused load is filed.
 *  - A searcher past a loaded row's turn reads the row's copied columns.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc load_closes=false: without the closure, the batch claims the row the load copies,
 *      and their keys collide
 *  @verify fail sc load_key_before_finish=false: after it, the searcher reads the loaded shard's row
 *      before its key
 *  @verify fail sc keys_once_covered=false: filing before the step tally is checked, a refused
 *      `finish` leaves the short shard's row filed, and the batch claiming it after the reopening
 *      files it a second time
 *  @verify fail sc copy_drained_before_finish=false: turning without the copy's event, the searcher
 *      reads the loaded row before its columns arrive
 *  @verify fail rc11 record_order=order_relaxed: a relaxed record carries none of the copy, so the
 *      turn after the wait publishes the row with its columns stale
 */
#define thread_count 5
#define location_count 15
#define history_depth 3
#define mutators 1
#define loader_thread 2
#define first_stream 3
#include <dense_vram_indexes/protocol.pml>

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) {
    byte read_shard;
    int read_watermark, read_value;
    search(t);
    for (read_shard : 0 .. shards - 1) {
        read_copied(t, read_shard, loader_thread, read_watermark, read_value)
    }
}

// a shard's stream, running the column copy and the record the loader queued, in order, until none comes
proctype stream(byte t) {
    byte kind, named, value;
    int seen;
end:
    do
    :: stream_queue[t - first_stream] ? kind, named, value ->
        run_operation(t, t - first_stream, kind, named, value, seen)
    od
}

// the closure, then per shard the key at the watermark and the queued copy, and `load_plan_t::finish`,
// the short shard's refusal, the wait on each copy's event and the turn; then the reopening
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
        fi;
        queue_copy(shard)
    };
    if
    :: true -> whole = true
    :: true -> whole = false
    fi;
    // finish: per shard the keys filed, then the short shard refused, its row given back by the dying
    // plan, else the copy awaited and the turn
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
        if
        :: copy_drained_before_finish -> await_event(t, shard, 1, seen)
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
    atomic { run mutator(0); run searcher(1); run loader(2); run stream(3); run stream(4) };
    (done == 3);
    audit_keys()
}
