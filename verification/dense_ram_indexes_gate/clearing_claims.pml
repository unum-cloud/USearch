/**
 *  @file verification/dense_ram_indexes_gate/clearing_claims.pml
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief An insert plan that claims, drops the gate and steps later, beside a `clear`: the
 *      claim-aware `close` refuses a claim not yet dispensed.
 *
 *  The index starts empty on one shard of one row. The planner makes two calls, each holding the
 *  gate's shared side: `plan_insert` claims the row, and the step dispenses it. The clearer takes
 *  the unique side or is refused, closes the shard through the claim-aware `close`, and inside its
 *  window forgets the shard in place. The orderings the gate and the window carry are `epoch.pml`'s
 *  and `basic_shared_mutex.pml`'s.
 *
 *  - Every dispense consumes a claim that still stands.
 *
 *  @verify pass sc
 *  @verify fail sc claims_check=false: a clear between `plan_insert` and the step forgets the claim,
 *      and the step dispenses a row no claim covers, the capacity a later claimer takes again
 */
#define thread_count 2
#define location_count 4
#define history_depth 1
#include <dense_ram_indexes_gate/protocol.pml>

proctype planner(byte t) {
    int observed;
    byte id;
    bool granted, refused;
    hold_shared(t, observed);
    claim(0, 1, granted);
    release_shared(t, observed);
    hold_shared(t, observed);
    dispense(0, id, refused);
    if
    :: refused -> release_claim(0, 1)
    :: else
    fi;
    release_shared(t, observed)
}

proctype clearer(byte t) {
    int observed, seen, epoch_seen;
    byte shard;
    bool succeeded, refused;
    try_lock(t, gate, observed, succeeded);
    if
    :: succeeded ->
        close_shards(t, 1, refused);
        if
        :: !refused ->
            begin_window(t, epoch_seen);
            forget(t, 0);
            end_window(t, epoch_seen);
            reopen_shards(1)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

init { atomic { run planner(0); run clearer(1) } }
