/**
 *  @file verification/dense_ram_indexes_gate/epoch.pml
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief A reader bracketing its call with the epoch alone beside an in-place writer moving a row
 *      between two shards inside a destructive window.
 *
 *  Each shard holds one row. The writer takes the unique side, makes the epoch odd, moves shard 1's
 *  row onto shard 0 by storing both watermarks, the way in-place compaction places a row and forgets
 *  its source, and makes the epoch even. The reader acquires an even epoch, reads both watermarks,
 *  and validates; a moved epoch retries once, then refuses.
 *
 *  - A validated read counts the row once: both watermarks old or both new.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 exit_fence=false: without it, a reader that read the writer's new shard-1
 *      watermark still re-reads the even epoch it started at and validates the old shard-0
 *      watermark beside it, counting the row on neither shard
 *  @verify fail rc11 window_fence=false: without it, the rewrites carry no view of the odd epoch, so
 *      the reader's acquire fence moves its exit read nowhere and a torn pair validates
 *  @verify fail rc11 epoch_order=order_relaxed: both relaxed, a reader that starts at the window's
 *      even epoch reads a stale watermark beside a new one, and its exit read finds the same epoch
 *  @verify fail rc11 epoch_order=order_release: a relaxed entry load of the closing epoch carries
 *      the window's rewrites only to the exit fence, after the reads it should have covered
 *  @verify fail rc11 epoch_order=order_acquire: the reader acquires the even epoch without the
 *      rewrites before it
 *  @verify fail sc free_in_window=true: a window that frees the shard it empties frees memory a
 *      reader that took its epoch before the window still touches
 */
#define thread_count 2
#define location_count 5
#define history_depth 3
#include <dense_ram_indexes_gate/protocol.pml>

byte object;   // which allocation the shard pointer names, a ghost
bool freed[2]; // an allocation returned to the allocator, a ghost

proctype writer(byte t) {
    int observed, epoch_seen;
    bool succeeded;
    try_lock(t, gate, observed, succeeded);
    if
    :: succeeded ->
        begin_window(t, epoch_seen);
        if
        :: free_in_window -> atomic { freed[object] = true; object = 1 - object }
        :: else
        fi;
        store(t, watermark(0), order_relaxed, 2);
        store(t, watermark(1), order_relaxed, 0);
        end_window(t, epoch_seen);
        unlock(t, gate, observed)
    :: else
    fi
}

proctype reader(byte t) {
    int epoch_seen, again, first, second;
    byte attempt, pointed;
    bool valid;
    do
    :: attempt == 2 -> break
    :: else ->
        read_epoch(t, epoch_seen);
        pointed = object;
        atomic { assert(!freed[pointed]); load(t, watermark(0), order_relaxed, first) };
        atomic { assert(!freed[pointed]); load(t, watermark(1), order_relaxed, second) };
        still_valid(t, epoch_seen, again, valid);
        if
        :: valid -> assert(first + second == 2); break
        :: else -> attempt++
        fi
    od
}

init {
    atomic {
        seed(watermark(0), 1);
        seed(watermark(1), 1);
        run writer(0);
        run reader(1)
    }
}
