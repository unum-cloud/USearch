/**
 *  @brief The dispenser closed around a load or a refinement beside an inserter and a searcher:
 *      every dispense refuses while it is closed, so no insert reuses a loaded ID or loses a
 *      back-link to a rewrite.
 *
 *  `ram_indexes` closes every shard's `next_dispensed_id_` before an absorb lands its rows or a
 *  refine rewrites them, and reopens it at the end. In one run the index starts empty and the
 *  closer loads one row: it closes the dispenser, appends at the watermark, lands the row, runs
 *  `finish`, whose compare-exchange leaves a closed dispenser closed while the watermark moves past
 *  the row, and reopens the dispenser at the watermark. In the other the index rests at one
 *  published row and the closer refines it: it closes the dispenser, stages row 0 as `refine_one_`
 *  does before its walk, rewrites it under its lock from the staged copy, and reopens. Beside
 *  either, the inserter dispenses one ID or is refused, wires its row, back-links into row 0 once
 *  it saw it published, and takes its turn; a searcher reads every row below the watermark it
 *  acquired.
 *
 *  No ID is dispensed twice, every row below the watermark reads published, and a back-link the
 *  inserter appended survives the refinement.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc dispenser_closes=false: a load closes the dispenser; checking only at
 *      `plan_load` that the dispenser stands level with the watermark, it leaves the dispenser where
 *      it was until `finish`, and an insert arriving meanwhile dispenses the ID the load is landing
 *  @verify fail sc refine_closes=false: a refinement closes the dispenser; without it, an insert
 *      appends its back-link into row 0 between the refinement's staging and its rewrite, and the
 *      rewrite from the staged copy drops it, leaving the new row without that in-link
 */
#define thread_count 3
#define location_count 7
#define history_depth 5
#define named_rows false
#define appended_rows false
#define refined_rows false
#include "protocol.pml"

byte done; // the inserter and the closer, once finished

proctype inserter(byte t) {
    insert(t);
    done++
}

proctype searcher(byte t) {
    int watermark;
    search(t, watermark)
}

// `plan_load` of one row at the watermark, the row landed, `finish`, and the reopening
proctype loader(byte t) {
    int visible, dispensed, expected;
    bool closed, moved, reopened;
    if
    :: dispenser_closes ->
        close_dispenser(t, visible, closed);
        if
        :: !closed -> goto refused
        :: else
        fi
    :: else ->
        load(t, last_visible_id, order_acquire, visible);
        load(t, next_dispensed_id, order_acquire, dispensed);
        if
        :: dispensed != visible -> goto refused
        :: else
        fi
    fi;
    atomic {
        assert(!taken[visible]);
        taken[visible] = true
    };
    land_row(t, visible);
    finish_load(t, visible, visible + 1, expected, moved);
    if
    :: dispenser_closes -> reopen_dispenser(t, visible, expected, reopened)
    :: else
    fi;
refused:
    done++
}

// `refine_one_` on row 0 between the closing and the reopening: the row staged before the walk,
// then rewritten under its lock from the staged copy
proctype refiner(byte t) {
    int visible, expected, first, staged, displaced;
    bool closed, reopened;
    if
    :: refine_closes ->
        close_dispenser(t, visible, closed);
        if
        :: !closed -> goto refused
        :: else
        fi
    :: else
    fi;
    // `stage_neighbors_`: slot zero acquired with a lock holder waited out, then the tail relaxed
    do
    :: (newest_value(row_lock(0)) != sentinel);
        load(t, row_lock(0), order_acquire, first);
        if
        :: first != sentinel -> break
        :: else
        fi
    od;
    load(t, row_tail(0), order_relaxed, staged);
    lock(t, 0, displaced);
    store(t, row_tail(0), order_relaxed, staged);
    unlock(t, 0, displaced);
    if
    :: refine_closes -> reopen_dispenser(t, visible, expected, reopened)
    :: else
    fi;
refused:
    done++
}

// once both are done: a back-link the inserter appended into row 0 is still there
proctype checker() {
    (done == 2);
    assert(!linked || newest_value(row_tail(0)) == edited(1))
}

// an empty index beside a load, or one published row beside a refinement
init {
    atomic {
        if
        :: run inserter(0); run loader(1); run searcher(2); run checker()
        :: seed(next_dispensed_id, 1);
            seed(last_visible_id, 1);
            seed(last_allocated_id, 1);
            seed(row_lock(0), published(0));
            seed(row_tail(0), published(0));
            taken[0] = true;
            run inserter(0); run refiner(1); run searcher(2); run checker()
        fi
    }
}
