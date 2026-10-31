/**
 *  @file verification/dense_vram_indexes/saving.pml
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief A save beside one batch: the save closes every shard's dispenser in shard order, refused
 *      while the batch is in flight on either, copies every key below each shard's watermark, and
 *      drains each shard's columns off the device through its pinned double-buffered bounce, the
 *      last half in `finish`.
 *
 *  `plan_save` holds the closure until the plan dies, since the device storage cannot be
 *  settlement-probed, and a batch meeting a closed shard is refused. The key copy stands for the
 *  host key column. Each shard's `save_plan_t` queues one half's copies off the device on the shard's
 *  stream and records that half's event behind them; the host waits on the previous half's event
 *  before reading it into the columns, so one half crosses while the other copies. `finish` waits on
 *  the pending half's event, reads it, and checks that every row arrived before the save returns and
 *  reopens the shards.
 *
 *  - The copy holds the batch on both shards or on neither.
 *  - Every key it copies is its claimer's.
 *  - Once the save returns, the checkpoint holds every shard's columns.
 *  - Once both are done, no row of a failed batch is filed.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc save_closes=false: without the closure, it copies the first shard after the
 *      batch's turn there and the second before it, a checkpoint holding half a batch
 *  @verify fail sc copy_drained_before_finish=false: returning without the pending half's event,
 *      the caller reads a checkpoint whose last half has not arrived
 */
#define thread_count 4
#define location_count 22
#define history_depth 3
#define mutators 1
#define first_stream 2
#include <dense_vram_indexes/protocol.pml>

proctype mutator(byte t) { insert_batch(t) }

// a shard's stream, running each half's copies off the device and its record the saver queued, in
// order, until none comes
proctype stream(byte t) {
    byte kind, named, value;
    int seen;
end:
    do
    :: stream_queue[t - first_stream] ? kind, named, value ->
        run_operation(t, t - first_stream, kind, named, value, seen)
    od
}

// the closure, the copy of every shard below its watermark, each shard's two halves queued with the
// first read behind its event, `finish` awaiting the pending half, the checkpoint as the caller then
// reads it, and the reopening
proctype saver(byte t) {
    byte shard, count, id;
    int seen, visible, watermarks[shards];
    if
    :: save_closes ->
        close_shards(t, count, visible);
        if
        :: count != shards -> goto contended
        :: else
        fi
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermarks[shard]);
        id = 0;
        do
        :: id < watermarks[shard] ->
            load(t, key(shard, id), order_relaxed, seen);
            assert(owner_of(shard, id) != 0 && seen == key_of(owner_of(shard, id) - 1));
            id++
        :: else -> break
        od;
        // `save` per step: the half's copies and their record queued, then the previous half awaited
        // and read while this one copies
        queue_half(shard, 0);
        queue_half(shard, 1);
        await_event(t, shard, 1, seen);
        load(t, bounce_half(shard, 0), order_relaxed, seen);
        assert(seen == copied_value)
    };
    assert(!committing[0] || (watermarks[0] > part_of(0, 0)) == (watermarks[1] > part_of(0, 1)));
    // `finish`: the pending half awaited and read, then the tally checked
    for (shard : 0 .. shards - 1) {
        if
        :: copy_drained_before_finish -> await_event(t, shard, 2, seen)
        :: else
        fi;
        load(t, bounce_half(shard, 1), order_relaxed, seen);
        assert(seen == copied_value)
    };
    if
    :: save_closes -> count = shards; reopen_shards(count)
    :: else
    fi;
contended:
    done++
}

init {
    atomic { run mutator(0); run saver(1); run stream(2); run stream(3) };
    (done == 2);
    audit_keys()
}
