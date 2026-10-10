/**
 *  @brief `dense_ram_cursor` beside two inserters: the constructor's acquire of the watermark,
 *      `seek`, `follow` refused past the snapshot, `refresh`, and the same follow admitted.
 *
 *  `seek` acquires slot zero exactly once per try and waits out a sentinel on the index's monitor,
 *  then reads the tail relaxed. The follower's own seek waits out the stale sentinel a traversal
 *  would trip on, and the row it follows rides the back-link fence into even a relaxed watermark
 *  store, so only the searcher reads a row stale when the watermark's release goes.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: the watermark's release carries every row below
 *      it; without it, the searcher reads a row below the watermark it acquired unpublished
 */
#define thread_count 4
#define location_count 9
#define history_depth 4
#define named_rows true
#define appended_rows false
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
    // follow: malformed at or past the snapshot, since the row it names may not have reached this cursor
    if
    :: id_of(staged) < visible_count -> read_published(t, id_of(staged), seen)
    :: else
    fi;
    // refresh: the snapshot widens and never shrinks, and the staged ID may be admitted now
    load(t, last_visible_id, order_acquire, seen);
    assert(seen >= visible_count);
    visible_count = seen;
    if
    :: id_of(staged) < visible_count -> read_published(t, id_of(staged), seen)
    :: else
    fi;
past_end:
    skip
}

init { atomic { run inserter(0); run inserter(1); run searcher(2); run follower(3) } }
