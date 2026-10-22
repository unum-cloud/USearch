/**
 *  @brief A cursor reading one row, a clear landing, and the cursor reading the next row.
 *
 *  The index holds two rows on one shard. The cursor records the epoch and the watermark at
 *  construction, then reads one row per call; each read ends with `still_valid` against the epoch it
 *  recorded, and a moved epoch makes the read return zero rows with `contended_k`, then and on every
 *  later read. The clear takes the unique side, closes the shard, and inside its window forgets the
 *  rows in place: the watermark and the keys column go to zero, the memory stays. A row a read
 *  returns carries a live key.
 *
 *  The plan also clamps each read to the shard's current `size()`; the model leaves the clamp out,
 *  since within one epoch a watermark only grows, and a read past the size after a window touches
 *  only memory retention kept, which the exit check then voids. The fences the exit check leans on
 *  are `epoch.pml`'s claim.
 *
 *  @verify pass sc
 *  @verify fail sc revalidate=false: a cursor read re-checks the epoch it recorded at construction;
 *      without it, the read after the clear indexes the forgotten shard with its stale watermark and
 *      returns a row whose key the clear wiped
 */
#define thread_count 2
#define location_count 7
#define history_depth 1
#include "protocol.pml"

// The words past the protocol's: the shard's keys column.
#define key(row) (5 + (row))

proctype cursor(byte t) {
    int recorded, again, visible, seen_key;
    byte node, call;
    bool valid;
    read_epoch(t, recorded);
    load(t, watermark(0), order_acquire, visible);
    do
    :: call == 2 -> break
    :: else ->
        valid = node < visible;
        if
        :: valid -> load(t, key(node), order_relaxed, seen_key)
        :: else
        fi;
        if
        :: valid && revalidate -> still_valid(t, recorded, again, valid)
        :: else
        fi;
        if
        :: valid -> assert(seen_key != 0); node++
        :: else
        fi;
        call++
    od
}

proctype clearer(byte t) {
    int observed, seen, epoch_seen;
    byte shard;
    bool succeeded, refused;
    try_lock(t, gate, observed, succeeded);
    if
    :: succeeded ->
        close_shards(t, 1, refused);
        if
        :: !refused ->
            begin_window(t, epoch_seen);
            forget(t, 0);
            store(t, key(0), order_relaxed, 0);
            store(t, key(1), order_relaxed, 0);
            end_window(t, epoch_seen);
            reopen_shards(1)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

init {
    atomic {
        seed(watermark(0), 2);
        seed(key(0), 1);
        seed(key(1), 2);
        claimed[0] = 2;
        dispensed[0] = 2;
        run cursor(0);
        run clearer(1)
    }
}
