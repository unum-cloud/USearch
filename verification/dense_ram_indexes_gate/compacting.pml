/**
 *  @file verification/dense_ram_indexes_gate/compacting.pml
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief An in-place compaction moving a row between shards and swapping its locator, beside a
 *      looker counting the moved key: one `replace` keeps the key in its slot throughout.
 *
 *  Shard 0 holds key A at row 0 and shard 1 key B at row 0, each with one locator slot. The
 *  compactor takes the unique side, closes both shards, and inside its window lands B onto shard 0
 *  at row 1, publishes shard 0's watermark, forgets shard 1 in place, and swaps B's one locator
 *  from {1, 0} to {0, 1} with one store of its slot, whose key half never changes. The looker counts
 *  B's slots, bracketed by the epoch, retrying once and then refusing.
 *
 *  - A validated count sees B exactly once.
 *
 *  @verify pass sc
 *  @verify pass sc count_bracket=false: under `replace`, `count` and `contains` stay exact without
 *      the epoch bracket against a compaction, since a swap moves no key out of its slot
 *  @verify pass sc rewrite=rebuild: the epoch bracket alone already hides a rebuild of the whole map
 *      from `count`, since the rebuild runs inside the window
 *  @verify fail sc rewrite=rebuild count_bracket=false: clearing the map and inserting every row's
 *      locator again leaves B in no slot meanwhile, so an unbracketed count misses the key
 */
#define thread_count 2
#define location_count 8
#define history_depth 1
#include <dense_ram_indexes_gate/protocol.pml>

// The words past the protocol's: shard 0's key at row 1, where B lands, and the two locator slots.
#define landed_key 5
#define slot(index) (6 + (index))

// The values: the keys, and a slot as its key above its locator, zero when empty.
#define key_a 1
#define key_b 2
#define slot_of(key, shard, node) ((key) * 8 + (shard) * 2 + (node) + 1)
#define key_of(slot) ((slot) / 8)

proctype compactor(byte t) {
    int observed, seen, epoch_seen;
    byte shard;
    bool succeeded, refused;
    try_lock(t, gate, observed, succeeded);
    if
    :: succeeded ->
        close_shards(t, 2, refused);
        if
        :: !refused ->
            begin_window(t, epoch_seen);
            store(t, landed_key, order_relaxed, key_b);
            atomic { claimed[0]++; dispensed[0]++ };
            store(t, watermark(0), order_release, 2);
            forget(t, 1);
            if
            :: rewrite == replace -> store(t, slot(1), order_relaxed, slot_of(key_b, 0, 1))
            :: else ->
                store(t, slot(0), order_relaxed, 0);
                store(t, slot(1), order_relaxed, 0);
                store(t, slot(0), order_relaxed, slot_of(key_a, 0, 0));
                store(t, slot(1), order_relaxed, slot_of(key_b, 0, 1))
            fi;
            end_window(t, epoch_seen);
            reopen_shards(2)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

// `count(B)`: every slot probed, bracketed by the epoch or not
proctype looker(byte t) {
    int epoch_seen, again, first, second;
    byte attempt;
    bool valid;
    do
    :: attempt == 2 -> break
    :: else ->
        if
        :: count_bracket -> read_epoch(t, epoch_seen)
        :: else
        fi;
        load(t, slot(0), order_acquire, first);
        load(t, slot(1), order_acquire, second);
        valid = true;
        if
        :: count_bracket -> still_valid(t, epoch_seen, again, valid)
        :: else
        fi;
        if
        :: valid -> assert((key_of(first) == key_b) + (key_of(second) == key_b) == 1); break
        :: else -> attempt++
        fi
    od
}

init {
    atomic {
        seed(watermark(0), 1);
        seed(watermark(1), 1);
        seed(slot(0), slot_of(key_a, 0, 0));
        seed(slot(1), slot_of(key_b, 1, 0));
        claimed[0] = 1;
        dispensed[0] = 1;
        claimed[1] = 1;
        dispensed[1] = 1;
        run compactor(0);
        run looker(1)
    }
}
