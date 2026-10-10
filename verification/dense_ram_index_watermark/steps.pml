/**
 *  @brief The load path's fan-out: two steps land one row each and tally it on `covered`, and the
 *      finisher publishes once the tally is whole.
 *
 *  `covered` stands for the `covered_vectors` and `covered_neighbors` each `load_plan_t::load` step
 *  adds to. The finisher acquires the tally and, when whole, bumps the dispenser and releases the
 *  watermark.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 covered_order=order_relaxed: the tally's release is what carries the steps'
 *      rows into the watermark; relaxed, the searcher finds a row below the watermark unpublished
 */
#define thread_count 4
#define location_count 10
#define history_depth 3
#define named_rows false
#define appended_rows false
#define refined_rows false
#include "protocol.pml"

// a step lands its row as the loader does, then tallies it on the plan
proctype step(byte t; byte row) {
    land_row(t, row);
    add_no_return(t, covered, covered_order, 1)
}

// the load path's `finish` behind the fan-out: the tally acquired, and when whole, the publication
proctype finisher(byte t) {
    int tally;
    load(t, covered, order_acquire, tally);
    if
    :: tally == 2 -> finish_load(t)
    :: else
    fi
}

proctype searcher(byte t) {
    int watermark;
    search(t, watermark)
}

init { atomic { run step(0, 0); run step(1, 1); run searcher(2); run finisher(3) } }
