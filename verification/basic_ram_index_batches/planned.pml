/**
 *  @brief A planned basic batch publishes whole: one dispense of two rows, two steppers landing one
 *      row each, and one `finish` that takes the range's turn once both landed.
 *
 *  `basic::ram_index::plan_insert` from `include/usearch/basic/ram_index.hpp`. The planner
 *  dispenses a range of two rows and fans it out to two steppers, each landing one position
 *  through `insert_plan_t::insert` and adding it to the plan's `landed` tally; a refused fan-out
 *  lands both positions on the planner instead. The planner's `finish` acquires the tally,
 *  retrying as a caller retries a refused `finish`, until both positions landed, then takes the
 *  range's one turn and lands the keys. A peer batch inserts two rows as `insert` does, and the
 *  searcher reads every row below the watermark it acquired wired and whole.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail rc11 landed_order=order_relaxed: each step adds its count to the tally with a
 *      release, which the finisher's acquire pairs with, so the turn's release carries every
 *      stepper's rows; relaxed, the searcher acquires the watermark and reads a row unwritten
 *  @verify fail sc finish_waits_landed=false: `finish` refuses until every position landed; taking
 *      the turn at once, it publishes a row no stepper wrote yet
 *  @verify stuck sc caller_lands_refused=false: a refused fan-out lands every position on the
 *      calling thread, since nothing cancels a dispensed range; dropped, `finish` waits on a tally
 *      nobody moves, and every later turn waits on it
 */
#define thread_count 5
#define location_count 11
#define history_depth 3
#include "protocol.pml"

// The fan-out's answer the steppers wait for: none yet, the steppers run, or refused.
#define undecided 0
#define accepted 1
#define refused 2

byte fan_out;    // the planner's fan-out, a ghost the steppers block on as a dispatch does
byte plan_first; // the plan's first ID, handed to the steppers with the dispatch

// `plan_insert`'s dispense, the fan-out or the caller's own landing, then `finish`
proctype planner(byte t) {
    byte other;
    int first, first_invisible, seen, tally;
    read_modify_write(t, next_dispensed_id, order_acq_rel, first, first + range_rows);
    if
    :: atomic { plan_first = first; fan_out = accepted }
    :: atomic { plan_first = first; fan_out = refused };
        if
        :: caller_lands_refused ->
            land_position(t, first, first_invisible, other, seen);
            land_position(t, first + 1, first_invisible, other, seen)
        :: else
        fi
    fi;
    finish(t, first, tally, seen)
}

// one thread of the fan-out, which lands its position once dispatched; it takes nothing from the
// planner's view, since every row it reads it admits through its own acquires
proctype stepper(byte t; byte position) {
    byte other;
    int first_invisible, seen;
    fan_out != undecided;
    if
    :: fan_out == accepted -> land_position(t, plan_first + position, first_invisible, other, seen)
    :: else
    fi;
    landed(t)
}

proctype inserter(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

init { atomic { run planner(0); run stepper(1, 0); run stepper(2, 1); run inserter(3); run searcher(4) } }
