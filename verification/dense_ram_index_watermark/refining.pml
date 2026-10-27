/**
 *  @file verification/dense_ram_index_watermark/refining.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A refinement beside a concurrent save: `refine_one_` relocks a published row and rewrites
 *      its tail, and moves no counter the save's settlement probe re-reads.
 *
 *  The index rests at two published rows, so the save plans; row 0 carries a second tail slot. The
 *  refiner closes the dispenser, which a save reads as settled, and reopens it at the end; between
 *  the two it runs `refine_one_` on row 0 and back-links into row 1, keeping each row's first
 *  neighbor, so every unlock stores back the slot-zero value its lock displaced. The saver runs
 *  `plan_save` and `finish` under `concurrent_k`, copying each row under the row's lock.
 *
 *  - A settled copy holds no locked row and no tail torn between two rewrites.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 save_locks_rows=false: without the row locks, a save copies row 1 while
 *      `back_link_neighbors` holds it and settles a file holding a locked row
 *  @verify fail sc,rc11 save_locks_rows=false lock_recheck=true: a save re-reading slot zero after
 *      its copies finds it unchanged and settles a copy whose tail holds one slot rewritten and one
 *      not
 */
#define thread_count 2
#define location_count 11
#define history_depth 6
#define named_rows false
#define appended_rows false
#define refined_rows true
#include <dense_ram_index_watermark/protocol.pml>

proctype refiner(byte t) {
    int visible, expected;
    bool closed, reopened;
    if
    :: refine_closes -> close_dispenser(t, visible, closed)
    :: else
    fi;
    refine(t);
    if
    :: refine_closes -> reopen_dispenser(t, visible, expected, reopened)
    :: else
    fi
}

proctype saver(byte t) { save(t) }

// the index at rest, every counter at two and both rows published, before any thread runs
init {
    atomic {
        seed(next_dispensed_id, 2);
        seed(last_visible_id, 2);
        seed(last_allocated_id, 2);
        seed(row_lock(0), published(0));
        seed(row_tail(0), published(0));
        seed(second_tail, published(0));
        seed(row_lock(1), published(1));
        seed(row_tail(1), published(1));
        run refiner(0); run saver(1)
    }
}
