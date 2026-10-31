/**
 *  @file verification/basic_ram_index_batches/planned.pml
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief A planned basic batch publishes whole: one dispense of two rows, two steppers placing one
 *      row each, and one `finish` that takes the range's turn once both placed.
 *
 *  `basic::ram_index::plan_insert` from `include/usearch/basic/ram_index.hpp`. The planner
 *  dispenses a range of two rows and fans it out to two steppers, each placing one position
 *  through `insert_plan_t::insert` and adding it to the plan's `placed` tally; a refused fan-out
 *  places both positions on the planner instead. A peer batch inserts two rows as `insert` does.
 *
 *  - The reader, as a search, reads every row below the watermark it acquired wired and whole.
 *  - The reader, as a count, finds exactly the rows below the watermark it acquired, each one whole.
 *
 *  @budget 2048
 *  @verify pass sc,rc11,far
 *  @verify fail rc11 placed_order=order_relaxed: relaxed, the search acquires the watermark and
 *      reads a row unwritten
 *  @verify fail sc finish_waits_placed=false: taking the turn at once, it publishes a row no stepper
 *      wrote yet
 *  @verify fail sc keys_in_turn=false: filed past the release, the count acquires the watermark
 *      past a row whose key is not filed yet, and counts one short
 *  @verify stuck sc caller_copies_refused=false: dropped, `finish` waits on a tally nobody moves, and
 *      every later turn waits on it
 */
#define thread_count 5
#define location_count 15
#define history_depth 3
#include <basic_ram_index_batches/protocol.pml>

// The fan-out's answer the steppers wait for: none yet, the steppers run, or refused.
#define undecided 0
#define accepted 1
#define refused 2

byte fan_out;    // the planner's fan-out, a ghost the steppers block on as a dispatch does
byte plan_first; // the plan's first ID, handed to the steppers with the dispatch

// `plan_insert`'s dispense, the fan-out or the caller's own placing, then `finish`
proctype planner(byte t) {
    byte other;
    int first, first_invisible, seen, tally;
    read_modify_write(t, next_dispensed_id, order_acq_rel, first, first + range_rows);
    if
    :: atomic { plan_first = first; fan_out = accepted }
    :: atomic { plan_first = first; fan_out = refused };
        if
        :: caller_copies_refused ->
            copy_position(t, first, first_invisible, other, seen);
            copy_position(t, first + 1, first_invisible, other, seen)
        :: else
        fi
    fi;
    finish(t, first, other, tally, seen)
}

// one thread of the fan-out, which places its position once dispatched; it takes nothing from the
// planner's view, since every row it reads it admits through its own acquires
proctype stepper(byte t; byte position) {
    byte other;
    int first_invisible, seen;
    fan_out != undecided;
    if
    :: fan_out == accepted -> copy_position(t, plan_first + position, first_invisible, other, seen)
    :: else
    fi;
    landed(t)
}

proctype inserter(byte t) { insert_batch(t) }

// a search or a count, on one thread, since `far` places the posted adds of five threads at most
proctype reader(byte t) {
    if
    :: search(t)
    :: find_each(t)
    fi
}

init { atomic { run planner(0); run stepper(1, 0); run stepper(2, 1); run inserter(3); run reader(4) } }
