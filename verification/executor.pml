/**
 *  @brief `forkunion_executor_t`: the one-caller hold through `claim`, the arena rewind in
 *      `reserve`, and the fan-out no second caller may rewind under.
 *
 *  From `include/usearch/utilities/ram_device.hpp`. `claim` is a `lock_if_free` on a spin mutex
 *  that answers `contended_k` rather than waiting; the holder rewinds every thread's arena in
 *  `reserve` and fans out over the pool in `for_tasks`. Two callers and two workers. A caller
 *  takes the hold with an acquire exchange, or leaves contended having touched nothing; the holder
 *  writes each worker's arena relaxed, then dispatches with a release on the epoch; each worker
 *  acquires the epoch, reads its arena, and counts itself down with `acq_rel`; the holder waits for
 *  the countdown, then drops the hold with a release. The fan-out is the pool's protocol, which
 *  ForkUnion's `flat_pool` models prove; here it is only the edge the arenas travel on.
 *
 *  A worker reads the arena its holder wrote, never one rewound by another caller, no arena is
 *  rewound while a fan-out is live, and a refused caller writes nothing.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc hold=false: `claim` holds one caller at a time and answers `contended_k` to a
 *      second; without the hold, the second caller rewinds the arenas under the first one's live
 *      fan-out
 */
#define thread_count 4
#define location_count 5
#define history_depth 7
#include "weak_memory.pml"

// The words: the hold, each worker's arena as the generation that last rewound it, the epoch, and the countdown.
#define held 0
#define arena(worker) (1 + (worker))
#define epoch 3
#define pending 4
#define workers 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef hold
#define hold true
#endif

byte callers_done;   // held or refused, either way finished
byte fanouts_active; // dispatched and not yet joined

proctype caller(byte t) {
    byte worker;
    int seen;
    // claim: the hold or a contended answer, never a wait
    if
    :: hold ->
        read_modify_write(t, held, order_acquire, seen, 1);
        if :: seen == 1 -> goto refused :: else fi
    :: else
    fi;
    // reserve: every arena rewound and seeded, which no live fan-out may be reading
    assert(fanouts_active == 0);
    for (worker : 0 .. workers - 1) { store(t, arena(worker), order_relaxed, t + 1) };
    // for_tasks: the fan-out, the pool's release on its epoch and the countdown
    store(t, pending, order_relaxed, workers);
    atomic { fanouts_active++; store(t, epoch, order_release, t + 1) };
    (newest_value(pending) == 0);
    load(t, pending, order_acquire, seen);
    fanouts_active--;
    if
    :: hold -> store(t, held, order_release, 0)
    :: else
    fi;
refused:
    callers_done++
}

proctype worker(byte t; byte own) {
    int last_epoch, new_epoch, seen, before;
    do
    :: (newest_value(epoch) != last_epoch || callers_done == 2) ->
        load(t, epoch, order_acquire, new_epoch);
        if
        :: new_epoch == last_epoch && callers_done == 2 -> break
        :: new_epoch == last_epoch
        :: else ->
            load(t, arena(own), order_relaxed, seen);
            assert(seen == new_epoch);
            read_modify_write(t, pending, order_acq_rel, before, before - 1);
            last_epoch = new_epoch
        fi
    od
}

init { atomic { run caller(0); run caller(1); run worker(2, 0); run worker(3, 1) } }
