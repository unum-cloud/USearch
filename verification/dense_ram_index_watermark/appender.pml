/**
 *  @file verification/dense_ram_index_watermark/appender.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The load path's `finish`: one compare-exchange moving `next_dispensed_id_` from the
 *      load's base past its rows, before the watermark's release store, beside a searcher and a save.
 *
 *  The loader lands both rows relaxed, moves the dispenser and releases the watermark.
 *
 *  - A saver that acquired the watermark must read the dispenser level with it.
 *
 *  @verify pass sc,rc11,far
 *  @verify pass far appender_order=order_relaxed: a compare-exchange is never posted, so even a
 *      relaxed one lands before the watermark's release store that carries it
 *  @verify pass sc spends_plan=false: a second `finish` finds the dispenser past the load's base,
 *      so its compare-exchange moves nothing
 */
#define thread_count 3
#define location_count 7
#define history_depth 4
#define named_rows false
#define appended_rows true
#define refined_rows false
#include <dense_ram_index_watermark/protocol.pml>

proctype loader(byte t) {
    int expected;
    bool moved;
    land_row(t, 0);
    land_row(t, 1);
    finish_load(t, 0, 2, expected, moved);
    if
    :: !spends_plan -> finish_load(t, 0, 2, expected, moved)
    :: else
    fi;
    landed(t)
}

proctype searcher(byte t) {
    int watermark;
    search(t, watermark)
}

proctype saver(byte t) { save(t) }

init { atomic { run loader(0); run searcher(1); run saver(2) } }
