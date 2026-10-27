/**
 *  @file verification/basic_ram_index_batches/protocol.pml
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief A basic index batch: one add dispenses its range, its rows land, and one turn publishes it.
 *
 *  From `include/usearch/basic/ram_index.hpp`. A row's walk admits the rows below its snapshot
 *  outright and, past it, only the rows an acquire of their `wired_k` bit shows wired, as
 *  `dispensed_admission_t` does; each row writes its value and neighbors plain, then sets its own
 *  `wired_k` with a release. Two batches of two rows each: `serial.pml` lands a batch on its
 *  inserter's thread, as `insert` does, and `planned.pml` fans one out over steppers.
 */
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for: the two counters, each row's state, wired or not,
// each row's value and neighbors, written plain before the state, and a plan's `landed` tally.
#define next_dispensed_id 0
#define last_visible_id 1
#define wired(id) (2 + (id))
#define row(id) (6 + (id))
#define landed_tally 10
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
#ifndef landed_order
#define landed_order order_release
#endif
#ifndef finish_waits_landed
#define finish_waits_landed true
#endif
#ifndef caller_lands_refused
#define caller_lands_refused true
#endif

bool keyed[rows]; // the key map holds the row's key

// a read of a row a walk or a search admitted: its payload, in the reader's view
inline read_row(t, id, seen) {
    load(t, row(id), order_relaxed, seen);
    assert(seen == moved(id))
}

// one row: the walk over the rows below @p walk_bound, published below the snapshot and past it
// admitted through `wired_k`, then the row's payload and its `wired_k` release
inline wire(t, id, walk_bound, first_invisible, other, seen) {
    load(t, last_visible_id, order_acquire, first_invisible);
    other = 0;
    do
    :: other == walk_bound -> break
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

// `insert` of one batch on its thread: the range from one add, each row wired with its walk below
// the range, since the batch's earlier row it wired itself, one turn, then the keys
inline insert_batch(t) {
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

// `insert_plan_t::insert` of one position: the row wired with its walk below its own ID, where a
// batch-mate another step wired is admitted only through `wired_k`, then the step's count added to
// the tally, waking a `finish` armed on it
inline land_position(t, id, first_invisible, other, seen) {
    wire(t, id, id, first_invisible, other, seen);
    atomic {
        add_no_return(t, landed_tally, landed_order, 1);
        wake(landed_tally)
    }
}

// `insert_plan_t::finish`, retried while it refuses: the tally acquired until every position
// landed, then the range's one turn and the keys
inline finish(t, first, tally, watermark) {
    if
    :: finish_waits_landed ->
        do
        :: load(t, landed_tally, order_acquire, tally);
            if
            :: tally == range_rows -> break
            :: else -> wait_on(t, landed_tally, tally, false)
            fi
        od
    :: else
    fi;
    take_turn(t, last_visible_id, first, first + range_rows, order_release, watermark);
    keyed[first] = true;
    keyed[first + 1] = true
}

// a search: the watermark acquired, never inside a range, every row below it wired and whole, and
// under `sc` no key naming a row the watermark does not cover
inline search(t) {
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
