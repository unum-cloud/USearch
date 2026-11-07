/**
 *  @file verification/dense_ram_index_refine/sweep.pml
 *  @author Ash Vardanian
 *  @date November 5, 2026
 *  @brief One sweep tile's two scans, its merge, and a walk reading the merged row.
 *
 *  Two pool threads count their source rows' links, the caller offsets the tile, both scatter onto
 *  the tape, and each merges its target row's run. A walk reads row 0 meanwhile. @c tape_capacity
 *  runs the tape short, and @c offsets_add replays the count's other forms.
 *
 *  - Every link in the tile lands in exactly one tape slot.
 *  - A run holds exactly its target's links, less those past the tape.
 *  - A walk reads each slot old or merged, and acquiring merged slot zero reads the merged tail.
 *
 *  @verify pass sc,rc11,far
 *  @verify pass sc,rc11 tape_capacity=2
 *  @verify pass sc,rc11 tape_capacity=1
 *  @verify fail far offsets_add=add: a relaxed @c add is posted under Intel's RAO-INT rule and
 *      still in flight past the await, so @c offset_back_links_ reads a short count, and a run
 *      comes up short or two links share a slot
 *  @verify fail sc,rc11 offsets_add=plain: two scanners lose a count, so two links share a slot
 *  @verify fail rc11 await_order=order_relaxed: relaxed, @c offset_back_links_ reads counts it was
 *      never told are final, so two links share a slot, and a merge reads a slot still unwritten
 *  @verify fail sc,rc11 clear_offsets=false: the count starts from the previous tile's run ends,
 *      and the tile ends a row early
 *  @verify fail sc,rc11 rows_change_between_scans=true: a row written between the scans overruns
 *      its run into the next one's first slot
 *  @verify fail rc11 peek_order=order_relaxed: relaxed, a walk reads the merged slot zero
 *      and the old tail
 *  @verify fail rc11 unlock_order=order_relaxed: relaxed, the same walk acquires nothing
 *      from the unlock
 *  @verify fail sc,rc11 whole_rows=true: a walk takes no lock and re-reads nothing, so one that
 *      peeked the old slot zero gathers the merged tail
 */
#define thread_count 4
#define location_count 9
#define history_depth 8
#include <dense_ram_index_refine/protocol.pml>

/** A pool thread: source row @p t 's scans, then target row @p t 's merge. */
proctype worker(byte t) {
    int seen, first, bound, position, value;
    byte row = t, sources;
    await_fork(t, 1, seen);
    count_back_links(t, seen);
    arrive(t, seen);
    await_fork(t, 2, seen);
    scatter_back_links(t, row, position, value);
    arrive(t, seen);
    await_fork(t, 3, seen);
    merge_back_links(t, row, first, bound, position, value, sources);
    arrive(t, seen);
    landed(t)
}

/** The thread driving the plan: every fork and await, and @c advance between them. */
proctype caller(byte t) {
    int awaited, rows, offset, count;
    open_sweep(t);
    fork_phase(t, 1);
    await_join(t, 2, awaited);
    offset_back_links(t, rows, offset, count);
    fork_phase(t, 2);
    await_join(t, 4, awaited);
    fork_phase(t, 3);
    await_join(t, 6, awaited)
}

/** A search walking row 0 at any point of the sweep. */
proctype reader(byte t) {
    int head, prefetched, gathered;
    walk_row(t, head, prefetched, gathered)
}

/** The offsets as the previous tile's merge left them, and row 0 before its merge. */
init {
    atomic {
        seed(row_lock, old_first);
        seed(row_tail, old_tail);
        seed(incoming_offsets(0), 1);
        seed(incoming_offsets(1), 1);
        run worker(0); run worker(1); run caller(2); run reader(3)
    }
}
