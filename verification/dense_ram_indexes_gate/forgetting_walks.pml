/**
 *  @file verification/dense_ram_indexes_gate/forgetting_walks.pml
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief A search walk beside a `clear`: a row the window forgets reads as locked forever, so a
 *      walk that reaches it must notice the watermark dropped beneath it rather than wait.
 *
 *  Shard 0 holds one published row. The walker takes an even epoch and the watermark snapshot, and,
 *  the row being below it, peeks the row's slot zero, as the walk's slot-zero wait and the landmark
 *  re-roll do. On the sentinel it re-reads the live watermark relaxed and leaves a row at or above
 *  it; otherwise it waits the sentinel out as a lock through the capped waiter. The clearer takes
 *  the unique side or is refused, closes the shard, and inside its window forgets it as `forget_`
 *  does: the watermark to zero, then the row refilled with the sentinel, which wakes no one. The
 *  walker then validates its epoch, retries once, and refuses.
 *
 *  - The walker always finishes: no wait outlives the clear.
 *
 *  @verify pass sc,rc11
 *  @verify stuck sc,rc11 walk_revalidates=false: a walk that took its snapshot before the window
 *      and peeks the row after it waits for an unlock no one will store
 */
#define thread_count 2
#define location_count 6
#define history_depth 3
#include <dense_ram_indexes_gate/protocol.pml>

// The words past the protocol's: the row's slot zero.
#define slot_zero 5

// The values: the row's ID, and the sentinel, the all-ones ID in silicon.
#define row_id 0
#define sentinel 255

// The knob: the walk re-reads the watermark when slot zero reads the sentinel below its snapshot.
#ifndef walk_revalidates
#define walk_revalidates true
#endif

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
            store(t, slot_zero, order_relaxed, sentinel);
            end_window(t, epoch_seen);
            reopen_shards(1)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

// a search under `read_retrying_once`: below its snapshot a sentinel is a lock, unless the live
// watermark dropped to the row since, and the capped wait returns once the next pass can leave
proctype walker(byte t) {
    int epoch_seen, again, snapshot, first, visible;
    byte attempt;
    bool valid;
    do
    :: attempt == 2 -> break
    :: else ->
        read_epoch(t, epoch_seen);
        load(t, watermark(0), order_acquire, snapshot);
        if
        :: row_id < snapshot ->
            do
            :: load(t, slot_zero, order_acquire, first);
                if
                :: first != sentinel -> break
                :: else ->
                    if
                    :: walk_revalidates -> load(t, watermark(0), order_relaxed, visible)
                    :: else -> visible = snapshot
                    fi;
                    if
                    :: row_id >= visible -> break
                    :: else ->
                        wait_on(t, slot_zero, sentinel,
                                newest_value(slot_zero) != sentinel ||
                                    (walk_revalidates && row_id >= newest_value(watermark(0))))
                    fi
                fi
            od
        :: else
        fi;
        still_valid(t, epoch_seen, again, valid);
        if
        :: valid -> break
        :: else -> attempt++
        fi
    od
}

init {
    atomic {
        seed(watermark(0), 1);
        seed(slot_zero, 1);
        claimed[0] = 1;
        dispensed[0] = 1;
        run clearer(0);
        run walker(1)
    }
}
