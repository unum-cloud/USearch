/**
 *  @brief The load path's `finish`: one release no-return add on `next_dispensed_id_` before the
 *      watermark's release store, beside a searcher and a save.
 *
 *  The loader lands both rows relaxed, bumps the dispenser and releases the watermark. A saver that
 *  acquired the watermark must read the dispenser level with it.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail far appender_order=order_relaxed: the dispenser's add releases; relaxed, it is still
 *      carried under the C++ model, but under far memory it is posted and may be in flight when the
 *      watermark lands, so the plan reports contention on a settled index
 *  @verify fail sc spends_plan=false: a successful `finish` spends the plan; a second one moves the
 *      dispenser again and leaves the watermark, and the saver finds the dispenser past it for good
 */
#define thread_count 3
#define location_count 7
#define history_depth 2
#define named_rows false
#define appended_rows true
#include "protocol.pml"

proctype loader(byte t) {
    land_row(t, 0);
    land_row(t, 1);
    finish_load(t);
    if
    :: !spends_plan -> finish_load(t)
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
