/**
 *  @brief A load plan that dies before `finish`, and the `plan_load` after it.
 *
 *  The abandoner claims two rows, as `plan_load` does, and dies; the claim it owns releases them.
 *  The retrier's rows then start at the watermark again, and no second chunk grew.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc release_on_abandon=false: the plan owns its claim, whose death releases the
 *      rows; without the release, `last_allocated_id_` stays past the watermark, and the next
 *      plan's rows start past the leaked ones, where the header answers every later `plan_load`
 *      `contended_k`
 */
#define thread_count 2
#define location_count 3
#define history_depth 5
#include "protocol.pml"

byte abandoned; // the first plan died

// `plan_load`'s claim of two rows, then the plan's death
proctype abandoner(byte t) {
    int first, whole, old_remaining, reverted, seen_chunks, added;
    claim(t, 2, first);
    assert(first == 0);
    if
    :: release_on_abandon -> release_capacity(t, 2)
    :: else
    fi;
    landed(t);
    abandoned = 1
}

// the next `plan_load`: its rows start at the watermark again, and no second chunk grew
proctype retrier(byte t) {
    int first, claimed_rows, total, old_remaining, reverted, seen_chunks, added, unclaimed;
    (abandoned == 1);
    claim(t, 2, first);
    assert(first == 0);
    capacity(t, claimed_rows, total);
    assert(total == rows_per_chunk)
}

init { atomic { run abandoner(0); run retrier(1) } }
