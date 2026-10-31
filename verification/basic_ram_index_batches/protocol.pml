/**
 *  @file verification/basic_ram_index_batches/protocol.pml
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief A basic index batch: one add dispenses its range, its rows are placed, and one turn publishes it.
 *
 *  From `include/usearch/basic/ram_index.hpp`. A row's walk admits the rows below its snapshot
 *  outright and, past it, only the rows an acquire of their `wired_k` bit shows wired, as
 *  `dispensed_admission_t` does; each row writes its value and neighbors plain, then sets its own
 *  `wired_k` with a release. A range's turn holder files its keys in the map as its one writer,
 *  between the turn's wait and the watermark's release. Two batches of two rows each: `serial.pml`
 *  places a batch on its inserter's thread, as `insert` does, and `planned.pml` fans one out over
 *  steppers.
 */
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for: the two counters, each row's state, wired or not,
// each row's value and neighbors, written plain before the state, a plan's `placed` tally, and each
// row's seat in the key map, filed or not.
#define next_dispensed_id 0
#define last_visible_id 1
#define wired(id) (2 + (id))
#define row(id) (6 + (id))
#define placed_tally 10
#define key_state(id) (11 + (id))
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
#ifndef keys_in_turn
#define keys_in_turn true
#endif
#ifndef placed_order
#define placed_order order_release
#endif
#ifndef finish_waits_placed
#define finish_waits_placed true
#endif
#ifndef caller_copies_refused
#define caller_copies_refused true
#endif

// a read of a row a walk or a search admitted: its payload, in the reader's view
inline read_row(t, id, seen) {
    load(t, row(id), order_relaxed, seen);
    assert(seen == moved(id))
}

// the turn of [@p first, @p bound): the wait, the keys filed by the holder, the map's one writer, a
// release per seat, then the watermark's release; `keys_in_turn=false` files them past the release
inline publish_range(t, first, bound, id, watermark) {
    await_turn(t, last_visible_id, first, watermark);
    if
    :: keys_in_turn ->
        for (id : first .. bound - 1) { store(t, key_state(id), order_release, 1) };
        end_turn(t, last_visible_id, order_release, bound)
    :: else ->
        end_turn(t, last_visible_id, order_release, bound);
        for (id : first .. bound - 1) { store(t, key_state(id), order_release, 1) }
    fi
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
    :: !range_turn -> publish_range(t, id, id + 1, other, seen)
    :: else
    fi
}

// `insert` of one batch on its thread: the range from one add, each row wired with its walk below
// the range, since the batch's earlier row it wired itself, then one turn that files the keys
inline insert_batch(t) {
    byte other;
    int first, first_invisible, seen;
    read_modify_write(t, next_dispensed_id, order_acq_rel, first, first + range_rows);
    wire(t, first, first, first_invisible, other, seen);
    wire(t, first + 1, first, first_invisible, other, seen);
    if
    :: range_turn -> publish_range(t, first, first + range_rows, other, seen)
    :: else
    fi
}

// `insert_plan_t::insert` of one position: the row wired with its walk below its own ID, where a
// batch-mate another step wired is admitted only through `wired_k`, then the step's count added to
// the tally, waking a `finish` armed on it
inline copy_position(t, id, first_invisible, other, seen) {
    wire(t, id, id, first_invisible, other, seen);
    atomic {
        add_no_return(t, placed_tally, placed_order, 1);
        wake(placed_tally)
    }
}

// `insert_plan_t::finish`, retried while it refuses: the tally acquired until every position
// placed, then the range's one turn, which files the keys
inline finish(t, first, id, tally, watermark) {
    if
    :: finish_waits_placed ->
        do
        :: load(t, placed_tally, order_acquire, tally);
            if
            :: tally == range_rows -> break
            :: else -> wait_on(t, placed_tally, tally, false)
            fi
        od
    :: else
    fi;
    publish_range(t, first, first + range_rows, id, watermark)
}

// a search: the watermark acquired, never inside a range, and every row below it wired and whole
inline search(t) {
    byte id;
    int watermark, seen;
    load(t, last_visible_id, order_acquire, watermark);
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

// `count` as `find_each` runs it: the watermark acquired once, then every seat acquired, a hit kept
// only below the watermark; every row below it counts once, and each kept hit's row is whole
inline find_each(t) {
    byte id, found;
    int watermark, seen;
    load(t, last_visible_id, order_acquire, watermark);
    for (id : 0 .. rows - 1) {
        load(t, key_state(id), order_acquire, seen);
        if
        :: seen == 1 && id < watermark -> found++; read_row(t, id, seen)
        :: else
        fi
    };
    assert(found == watermark)
}
