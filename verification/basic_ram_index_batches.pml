/**
 *  @brief A basic index batch publishes whole: one add dispenses its range, one turn publishes it.
 *      Other batches' walks admit its rows earlier through one `wired` flag per row.
 *
 *  `basic::ram_index::insert` from `include/usearch/basic/ram_index.hpp`. Two inserters bring two
 *  rows each. One add dispenses an inserter's range, then each row walks, writes its value and
 *  neighbors plain, and stores its `wired` flag with a release; one turn from the range's first ID
 *  to its end publishes both rows, and the keys land after it. A row's walk draws below its own ID.
 *  Its own batch's earlier row it wired itself, so that one needs no acquire; the other batch's
 *  rows, when that batch dispensed first, `dispensed_admission_t` admits below the walk's snapshot
 *  outright, and past it only once an acquire of `wired` shows them wired. The searcher acquires
 *  the watermark, which never lies strictly inside a range, and reads every row below it wired and
 *  whole; under `sc` it also finds no key naming a row at or above the watermark.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 range_turn=false: one turn per range keeps a batch whole; a turn per row
 *      lets the searcher see half a batch
 *  @verify fail sc,rc11 wired_check=false: a walk admits a row past its snapshot only once an
 *      acquire of its `wired` flag shows it wired; unchecked, a later batch's walk reads an earlier
 *      batch's row before its inserter wrote it
 *  @verify fail rc11 wired_order=order_relaxed: `wired` is stored with a release, which carries the
 *      row's plain value and neighbors to the walk that admits it; relaxed, the walk reads the flag
 *      set and the row unwritten
 *  @verify fail rc11 admit_order=order_relaxed: `admits` loads `wired` with an acquire, the other
 *      half of the pair; relaxed, the walk reads the flag set and the row unwritten
 *  @verify fail sc keys_after_turn=false: the keys land after the turn, so a key `find` returns
 *      names a published row; before it, a lookup names a row the watermark does not cover
 */
#define thread_count 3
#define location_count 10
#define history_depth 5
#include "weak_memory.pml"
#include "watermark_turn.pml"

// The words, by the index member each stands for: the two counters, each row's `wired` flag, and
// each row's value and neighbors, written plain before the flag.
#define next_dispensed_id 0
#define last_visible_id 1
#define wired(id) (2 + (id))
#define row(id) (6 + (id))
#define range_rows 2
#define rows 4

// The values: a row holds its moved payload or nothing yet.
#define moved(id) (10 + (id))

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef range_turn
#define range_turn true
#endif
#ifndef wired_check
#define wired_check true
#endif
#ifndef wired_order
#define wired_order order_release
#endif
#ifndef admit_order
#define admit_order order_acquire
#endif
#ifndef keys_after_turn
#define keys_after_turn true
#endif

bool keyed[rows]; // the key map holds the row's key

// a read of a row a walk or a search admitted: its payload, in the reader's view
inline read_row(t, id, seen) {
    load(t, row(id), order_relaxed, seen);
    assert(seen == moved(id))
}

// one row: the walk over the rows below its range, the other batch's, published below the snapshot
// and past it admitted through `wired`, then the row's payload and its `wired` release
inline wire(t, id, range_first, first_invisible, other, seen) {
    load(t, last_visible_id, order_acquire, first_invisible);
    other = 0;
    do
    :: other == range_first -> break
    :: else ->
        if
        :: other < first_invisible -> read_row(t, other, seen)
        :: else ->
            if
            :: wired_check -> load(t, wired(other), admit_order, seen)
            :: else -> seen = 1
            fi;
            if
            :: seen != 0 -> read_row(t, other, seen)
            :: else
            fi
        fi;
        other++
    od;
    store(t, row(id), order_relaxed, moved(id));
    store(t, wired(id), wired_order, 1);
    if
    :: !keys_after_turn -> keyed[id] = true
    :: else
    fi;
    if
    :: !range_turn -> take_turn(t, last_visible_id, id, id + 1, order_release, seen); keyed[id] = true
    :: else
    fi
}

// `insert` of one batch: the range from one add, each row wired, one turn, then the keys
proctype inserter(byte t) {
    byte other;
    int first, first_invisible, seen;
    read_modify_write(t, next_dispensed_id, order_acq_rel, first, first + range_rows);
    wire(t, first, first, first_invisible, other, seen);
    wire(t, first + 1, first, first_invisible, other, seen);
    if
    :: range_turn ->
        take_turn(t, last_visible_id, first, first + range_rows, order_release, seen);
        keyed[first] = true;
        keyed[first + 1] = true
    :: else
    fi
}

// a search: the watermark acquired, never inside a range, every row below it wired and whole, and
// under `sc` no key naming a row the watermark does not cover
proctype searcher(byte t) {
    byte id;
    int watermark, seen;
    atomic {
        load(t, last_visible_id, order_acquire, watermark);
        for (id : 0 .. rows - 1) { assert(memory != sc || !keyed[id] || id < watermark) }
    };
    assert(watermark % range_rows == 0);
    id = 0;
    do
    :: id == watermark -> break
    :: else ->
        load(t, wired(id), order_relaxed, seen);
        assert(seen == 1);
        read_row(t, id, seen);
        id++
    od
}

init { atomic { run inserter(0); run inserter(1); run searcher(2) } }
