/**
 *  @brief A save beside one batch: the save closes every shard's dispenser in shard order, refused
 *      while the batch is in flight on either, copies every key below each shard's watermark, and
 *      reopens them, the last first.
 *
 *  `plan_save` holds the closure until the plan dies, since the device storage cannot be
 *  settlement-probed, and a batch meeting a closed shard is refused. The copy stands for every
 *  column a shard plan exports: it holds the batch on both shards or on neither, and every key it
 *  copies is its claimer's.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc save_closes=false: the save closes every shard before it copies; without the
 *      closure, it copies the first shard after the batch's turn there and the second before it, a
 *      checkpoint holding half a batch
 */
#define thread_count 2
#define location_count 7
#define history_depth 3
#define mutators 1
#include "protocol.pml"

proctype mutator(byte t) { insert_batch(t) }

// the closure, the copy of every shard below its watermark, and the reopening
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
        od
    };
    assert(!committing[0] || (watermarks[0] > part_of(0, 0)) == (watermarks[1] > part_of(0, 1)));
    if
    :: save_closes -> count = shards; reopen_shards(count)
    :: else
    fi;
contended:
    skip
}

init { atomic { run mutator(0); run saver(1) } }
