/**
 *  @brief The basic index's load path: two steps' tally, then `finish` under the unique side.
 *
 *  `load_plan_t` from `include/usearch/basic/ram_index.hpp`. Two steps each store their key and add
 *  their count to `covered` with a release; the finisher, holding the unique side through
 *  `basic_gate.pml`, acquires the tally, reads both keys, stores the claim and identity counters
 *  relaxed and the watermark with a release, and unlocks. A reader runs `watermark_` and `key_of`:
 *  a watermark it acquired at 2 must show the dispenser at 2 and node 1's key.
 *  `basic_ram_index_rows.pml` covers the rows and the gate's shared side.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail rc11,far covered_order=order_relaxed: the steps' tally add releases for the
 *      finisher's acquire, which is what puts the steps' keys behind the watermark; relaxed, the
 *      finisher reads no key, and under `far` the add is posted and lands carrying nothing
 *  @verify fail rc11 counters_before_watermark=false: `finish` stores the claim and identity
 *      counters before the watermark's release, so a reader that acquired the watermark sees the
 *      dispenser with it; reversed, the reader finds the dispenser behind the watermark, and
 *      `watermark_` reports a settled load contended
 */
#define thread_count 4
#define location_count 7
#define history_depth 3
#include "weak_memory.pml"
#include "basic_gate.pml"

// The words, by the index member each stands for.
#define shared_mutex 0
#define covered 1
#define claimed_count 2
#define next_dispensed_id 3
#define last_visible_id 4
#define key(id) (5 + (id))

// The values: a key word holds the record's key or nothing yet, and the steps land one row each.
#define keyed(id) (10 + (id))
#define step_count 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef covered_order
#define covered_order order_release
#endif
#ifndef counters_before_watermark
#define counters_before_watermark true
#endif

// load_plan_t::load: the step's records, then its share of the tally
proctype step(byte t) {
    store(t, key(t), order_relaxed, keyed(t));
    add_no_return(t, covered, covered_order, 1);
    landed(t)
}

// load_plan_t::finish under plan_load's hold: the tally, then claims, identity, watermark last
proctype finisher(byte t) {
    int observed, tally, first, second;
    bool succeeded;
    try_lock(t, shared_mutex, observed, succeeded);
    assert(succeeded);
    (newest_value(covered) == step_count);
    load(t, covered, order_acquire, tally);
    if
    :: tally == step_count ->
        load(t, key(0), order_relaxed, first);
        load(t, key(1), order_relaxed, second);
        assert(first == keyed(0) && second == keyed(1));
        if
        :: !counters_before_watermark -> store(t, last_visible_id, order_release, step_count)
        :: else
        fi;
        store(t, claimed_count, order_relaxed, step_count);
        store(t, next_dispensed_id, order_relaxed, step_count);
        if
        :: counters_before_watermark -> store(t, last_visible_id, order_release, step_count)
        :: else
        fi
    :: else
    fi;
    unlock(t, shared_mutex, observed)
}

// watermark_ and key_of: a watermark acquired at the row count shows the dispenser there and the keys
proctype reader(byte t) {
    int watermark, dispensed, seen;
    load(t, last_visible_id, order_acquire, watermark);
    if
    :: watermark == step_count ->
        load(t, next_dispensed_id, order_acquire, dispensed);
        assert(dispensed == step_count);
        load(t, key(1), order_relaxed, seen);
        assert(seen == keyed(1))
    :: else
    fi
}

init { atomic { run step(0); run step(1); run reader(2); run finisher(3) } }
