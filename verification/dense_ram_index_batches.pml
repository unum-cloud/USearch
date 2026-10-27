/**
 *  @file verification/dense_ram_index_batches.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A batch's commit on the dense index: one add dispenses its whole range, and one watermark
 *      turn makes it visible whole, with no new atomics.
 *
 *  Two commits of two rows each. A commit's leader dispenses, hands the range to a helper as the
 *  executor's fan-out would, each wires one row, the helper joins with a release the leader
 *  acquires, and the leader takes one turn from the range's first ID to its end. A row's wirer
 *  stores its payload relaxed ahead of the release of slot zero that publishes it. The searcher
 *  acquires the watermark, which never lies strictly inside a range, and reads every row below it.
 *
 *  - Every row a reader admits, it reads with its payload whole.
 *  - A back-link into a later commit's rows never leads into them.
 *
 *  @budget 8192: rc11 stores 2.2 GB of states, and three times that needs 8192
 *  @verify pass sc,rc11
 *  @verify stuck sc watermark_wakes=false: with none after the watermark store, the later commit
 *      waits forever for a turn that came
 *  @verify stuck sc recheck=false: without it, a turn that ends between the waiter's load and its
 *      arming wakes nobody, and the later commit blocks forever
 *  @verify fail sc range_turn=false: a turn per row lets the searcher see half a commit
 *  @verify fail sc,rc11 own_range_check=false: unchecked, it reads the sibling's payload before its
 *      wirer published it
 *  @verify fail rc11 admission_order=order_relaxed: relaxed, the walk sees the row published and
 *      reads its payload stale
 *  @verify fail rc11 publish_order=order_relaxed: relaxed, a walk that admits the row through its
 *      slot zero reads the payload stale
 */
#define thread_count 5
#define location_count 13
#define history_depth 7
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for; the four rows keep slot zero alone, but for row
// 0's tail, which the back-links edit, and each row's payload comes last.
#define next_dispensed_id 0
#define last_visible_id 1
#define row_lock(id) (2 + (id))
#define row_tail(id) 6
#define joined(commit) (7 + (commit))
#define payload(id) (9 + (id))
#define range_rows 2
#define rows 4

// The values rows carry: the sentinel is the all-ones ID in silicon, and a slot names an ID.
#define sentinel 0
#define published(id) (10 + (id))
#define edited(id) (20 + (id))
#define id_of(word) ((word) % 10)

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef range_turn
#define range_turn true
#endif
#ifndef own_range_check
#define own_range_check true
#endif
#ifndef admission_order
#define admission_order order_acquire
#endif
#ifndef publish_order
#define publish_order order_release
#endif

byte range_first[2];     // the range each commit dispensed, handed to its helper
bool dispatched[2];      // the fan-out's dispatch, which carries no row
hidden int payload_read; // scratch: a payload a reader read, checked in the same step

// a read of a row a reader admitted: the payload its wirer stored before publishing it
inline read_payload(t, id) {
    atomic {
        load(t, payload(id), order_relaxed, payload_read);
        assert(payload_read == published(id))
    }
}

// a walk's admission of a row past its snapshot: slot zero shows it published, then its payload
inline admit(t, id, seen) {
    atomic {
        load(t, row_lock(id), admission_order, seen);
        if
        :: seen != sentinel -> read_payload(t, id)
        :: else
        fi
    }
}

// one row of a commit: the walk below the snapshot and over the published rows dispensed before it
// began, the payload and slot zero's release, then the back-link into row 0 when the walk reached it
inline wire(t, id, sibling) {
    byte other;
    int first_invisible, first_undispensed, seen, displaced;
    bool linked;
    load(t, last_visible_id, order_acquire, first_invisible);
    load(t, next_dispensed_id, order_acquire, first_undispensed);
    linked = first_invisible > 0;
    if
    :: linked -> read_payload(t, 0)
    :: else
    fi;
    // past the snapshot, the other commit's first row, once dispensed and its slot zero shows it published
    other = (id < range_rows -> range_rows : 0);
    if
    :: !linked && other < first_undispensed ->
        admit(t, other, seen);
        linked = other == 0 && seen != sentinel
    :: else
    fi;
    if
    :: own_range_check -> admit(t, sibling, seen)
    :: else -> read_payload(t, sibling)
    fi;
    if
    :: id == 0 -> store(t, row_tail(0), order_relaxed, published(0))
    :: else
    fi;
    atomic {
        store(t, payload(id), order_relaxed, published(id));
        store(t, row_lock(id), publish_order, published(id))
    };
    if
    :: linked ->
        fence_release(t);
        do
        :: read_modify_write(t, row_lock(0), order_acq_rel, displaced, sentinel);
           if
           :: displaced != sentinel -> break
           :: else -> (newest_value(row_lock(0)) != sentinel)
           fi
        od;
        store(t, row_tail(0), order_relaxed, edited(id));
        store(t, row_lock(0), order_release, edited(id))
    :: else
    fi
}

// a commit's leader: the whole range from one add, its own row, the helper's join, then one turn
proctype committer(byte t; byte commit) {
    int first, joins, watermark;
    read_modify_write(t, next_dispensed_id, order_acq_rel, first, first + range_rows);
    atomic { range_first[commit] = first; dispatched[commit] = true };
    wire(t, first, first + 1);
    if
    :: !range_turn -> take_turn(t, last_visible_id, first, first + 1, order_release, watermark)
    :: else
    fi;
    (newest_value(joined(commit)) != 0);
    do
    :: load(t, joined(commit), order_acquire, joins);
       if
       :: joins != 0 -> break
       :: else
       fi
    od;
    if
    :: range_turn ->
        take_turn(t, last_visible_id, first, first + range_rows, order_release, watermark)
    :: else
    fi
}

// the executor's second worker on a commit's range: the range's second row, then the join
proctype helper(byte t; byte commit) {
    int first, watermark;
    dispatched[commit];
    first = range_first[commit];
    wire(t, first + 1, first);
    if
    :: !range_turn -> take_turn(t, last_visible_id, first + 1, first + 2, order_release, watermark)
    :: else
    fi;
    store(t, joined(commit), order_release, 1)
}

// a search: the watermark acquired, never inside a range, every row below it read, and the walk
// through row 0's tail, which a later commit's back-link may name, filtered by the snapshot
proctype searcher(byte t) {
    byte id;
    int watermark, found;
    load(t, last_visible_id, order_acquire, watermark);
    assert(watermark % range_rows == 0);
    do
    :: id == watermark -> break
    :: else -> read_payload(t, id); id++
    od;
    if
    :: watermark > 0 ->
        load(t, row_tail(0), order_relaxed, found);
        if
        :: id_of(found) < watermark -> read_payload(t, id_of(found))
        :: else
        fi
    :: else
    fi
}

init { atomic { run committer(0, 0); run helper(1, 0); run committer(2, 1); run helper(3, 1); run searcher(4) } }
