/**
 *  @file verification/dense_ram_index_watermark/inserters.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Two inserters, a searcher and a concurrent save: publication below the watermark, and the
 *      save's settlement probe.
 *
 *  The inserters dispense IDs 0 and 1. Each writes its own row's tail relaxed and slot zero with a
 *  release; the second, when it saw the first below the watermark, back-links into row 0. The
 *  saver runs `plan_save` and `finish` under `concurrent_k`. The lock's exchange acquires the
 *  back-link's unlock, which carries its dispense into the re-check, so no fence heads
 *  `back_link_neighbors` or the re-check.
 *
 *  - The searcher acquires the watermark and reads every row below it.
 *  - A settled save copied no locked row and no torn edit.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: without it, the searcher reads a row below the
 *      watermark it acquired unpublished
 *  @verify fail rc11 save_locks_rows=false: a save copying without the row locks has no edge to the
 *      back-link's dispense, re-reads the counter stale and settles a row caught locked
 *  @verify pass rc11 save_locks_rows=false back_link_fence=true finish_fence=true: a release fence
 *      heading the back-link and an acquire fence heading the re-check carry the dispense as the
 *      locks do, but `refining.pml` needs the locks anyway, so the header spends no fence
 */
#define thread_count 4
#define location_count 7
#define history_depth 7
#define named_rows false
#define appended_rows false
#define refined_rows false
#include <dense_ram_index_watermark/protocol.pml>

proctype inserter(byte t) { insert(t) }

proctype searcher(byte t) {
    int watermark;
    search(t, watermark)
}

proctype saver(byte t) { save(t) }

init { atomic { run inserter(0); run inserter(1); run searcher(2); run saver(3) } }
