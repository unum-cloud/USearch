/**
 *  @brief `dense_ram_cursor` beside two inserters: the constructor's acquire of the watermark,
 *      `seek`, `follow` refused past the snapshot, `refresh`, and the same follow admitted.
 *
 *  `seek` acquires slot zero exactly once per try and waits out a sentinel on the index's monitor,
 *  then reads the tail relaxed. The follower's own seek waits out the stale sentinel a traversal
 *  would trip on, and the row it follows rides the back-link fence into even a relaxed watermark
 *  store, so only the searcher reads a row stale when the watermark's release goes.
 *
 *  A back-link names a row once it is published, but a reader reaches the row's hosting chunk only
 *  behind the ID's dispense, which the watermark above the ID carries and a relaxed tail read does
 *  not; each row here grows a chunk of its own, the case where the followed row opens a fresh
 *  chunk. So `follow` seeks the staged ID only below the snapshot, and `refresh` widens the
 *  snapshot by acquiring the watermark.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: the watermark's release carries every row below
 *      it; without it, the searcher reads a row below the watermark it acquired unpublished
 *  @verify fail rc11 snapshot_check=false: `seek` refuses an ID at or past the cursor's snapshot;
 *      unchecked, the follower follows an ID it read from row 0's tail relaxed and locates a row
 *      whose hosting chunk has not reached it
 *  @verify fail rc11 refresh_order=order_relaxed: `refresh` acquires the watermark, which carries
 *      the dispense of every ID below it; relaxed, the snapshot widens past the staged ID with no
 *      edge to its chunk, and the second follow locates a row whose hosting chunk has not reached it
 */
#define thread_count 4
#define location_count 9
#define history_depth 4
#define named_rows true
#define appended_rows false
#define refined_rows false
#include "protocol.pml"

proctype inserter(byte t) { insert(t) }

proctype searcher(byte t) {
    int watermark;
    search(t, watermark)
}

// a cursor holder: the constructor's snapshot, seek(0), one follow refused, refresh, the follow admitted
proctype follower(byte t) {
    int visible_count, first_value, staged, seen;
    load(t, last_visible_id, order_acquire, visible_count);
    if
    :: visible_count == 0 -> goto past_end // seek(0) is malformed at or past the snapshot
    :: else
    fi;
    // seek: slot zero acquired exactly once per try, a sentinel waited out on the index's monitor, the tail relaxed
seek:
    (newest_value(row_lock(0)) != sentinel);
    load(t, row_lock(0), order_acquire, first_value);
    if
    :: first_value == sentinel -> goto seek
    :: else
    fi;
    load(t, row_tail(0), order_relaxed, staged);
    follow(t, staged, visible_count, seen);
    // refresh: the snapshot widens and never shrinks, and the staged ID may be admitted now
    load(t, last_visible_id, refresh_order, seen);
    assert(seen >= visible_count);
    visible_count = seen;
    follow(t, staged, visible_count, seen);
past_end:
    skip
}

init { atomic { run inserter(0); run inserter(1); run searcher(2); run follower(3) } }
