/**
 *  @file verification/dense_ram_index_refine.pml
 *  @author Ash Vardanian
 *  @date November 5, 2026
 *  @brief A refinement round's restitch rewriting a row while a search walks it.
 *
 *  A round lets searches run throughout, so @c rewrite_row_ in @c ram_index.hpp rewrites row 0
 *  under its lock while @c search_around in @c ram_index.hpp walks it: @c neighbors_ref::peek_first
 *  acquires slot zero, and the relaxed @c load_slot reads the tail once for the prefetch and again
 *  for the gather. The rewrite takes @c neighbors_ref::lock, stores the tail relaxed, and
 *  republishes the row by the release in @c neighbors_ref::unlock.
 *
 *  - A walk reads each slot old or merged, never merged then old.
 *  - A walk that acquires the merged slot zero gathers the merged tail.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 peek_order=order_relaxed: relaxed, a walk reads the merged slot zero
 *      and the old tail
 *  @verify fail rc11 unlock_order=order_relaxed: relaxed, the same walk acquires nothing
 *      from the unlock
 *  @verify fail sc,rc11 whole_rows=true: a walk takes no lock and re-reads nothing, so one that
 *      peeked the old slot zero gathers the merged tail
 */
#define thread_count 2
#define location_count 2
#define history_depth 3
#include <weak_memory.pml>

/** The words: row 0's slot zero and its one tail slot. */
#define row_lock 0
#define row_tail 1

/** The values: row 0 holds its old members until the rewrite, each distinct, with the sentinel the
 *  all-ones ID in silicon. */
#define sentinel 0
#define old_first 1
#define old_tail 2
#define merged_first 3
#define merged_tail 4

/** The knobs: the header's choices, and the stronger claim a walk never makes, each overridden by
 *  a verify line that replays it. */
#ifndef peek_order
#define peek_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif
#ifndef whole_rows
#define whole_rows false
#endif

/** @c rewrite_row_: the lock, the tail, and the unlock that republishes the row. */
inline rewrite_row(t, displaced) {
    read_modify_write(t, row_lock, order_acq_rel, displaced, sentinel);
    store(t, row_tail, order_relaxed, merged_tail);
    store(t, row_lock, unlock_order, merged_first)
}

/** The acquire of slot zero a walk makes, waiting out the lock holder. */
inline peek_first(t, first) {
    do
    :: load(t, row_lock, peek_order, first);
        if
        :: first != sentinel -> break
        :: else -> (newest_value(row_lock) != sentinel)
        fi
    od
}

/** Row 0 as a walk reads it: the peek, then the tail for the prefetch and again for the gather. */
inline walk_row(t, first, prefetched, gathered) {
    peek_first(t, first);
    load(t, row_tail, order_relaxed, prefetched);
    load(t, row_tail, order_relaxed, gathered);
    assert(prefetched == old_tail || prefetched == merged_tail);
    assert(gathered == merged_tail || prefetched == old_tail);
    assert(first != merged_first || gathered == merged_tail);
    assert(!whole_rows || first != old_first || gathered == old_tail)
}

/** A restitch task rewriting row 0. */
proctype restitcher(byte t) {
    int displaced;
    rewrite_row(t, displaced);
    landed(t)
}

/** A search walking row 0 at any point of the rewrite. */
proctype searcher(byte t) {
    int first, prefetched, gathered;
    walk_row(t, first, prefetched, gathered)
}

/** Row 0 published before the round. */
init {
    atomic {
        seed(row_lock, old_first);
        seed(row_tail, old_tail);
        run restitcher(0); run searcher(1)
    }
}
