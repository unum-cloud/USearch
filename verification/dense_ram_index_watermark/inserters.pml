/**
 *  @brief Two inserters, a searcher and a concurrent save: publication below the watermark, and the
 *      save's settlement probe.
 *
 *  The inserters dispense IDs 0 and 1. Each writes its own row's tail relaxed and slot zero with a
 *  release, the publication; the second, when it saw the first below the watermark, back-links into
 *  row 0: a release fence, the slot-zero lock exchange, a relaxed tail store, the release unlock.
 *  The searcher acquires the watermark and reads every row below it. The saver runs `plan_save` and
 *  `finish` under `concurrent_k`: three acquire loads, every row copied under its lock, an acquire
 *  fence, and the relaxed re-check of the three counters; a settled save copied no locked row and
 *  no torn edit. The lock's exchange acquires the back-link's unlock, which carries its dispense
 *  into the re-check, so the save rests on the release fence heading `back_link_neighbors` no more.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: the watermark's release carries every row below
 *      it; without it, the searcher reads a row below the watermark it acquired unpublished
 *  @verify pass rc11 back_link_fence=false
 */
#define thread_count 4
#define location_count 7
#define history_depth 7
#define named_rows false
#define appended_rows false
#define refined_rows false
#include "protocol.pml"

proctype inserter(byte t) { insert(t) }

proctype searcher(byte t) {
    int watermark;
    search(t, watermark)
}

proctype saver(byte t) { save(t) }

init { atomic { run inserter(0); run inserter(1); run searcher(2); run saver(3) } }
