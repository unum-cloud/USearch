/**
 *  `executor_t` from `include/usearch/utilities/ram_device.hpp`: one caller at a time, held
 *  through `claim`, which is a `lock_if_free` on a spin mutex that answers `contended_k`
 *  rather than waiting; the holder rewinds every thread's arena in `reserve` and fans out over
 *  the pool in `for_tasks`, and a second caller must never rewind an arena under a live fan-out.
 *
 *  Two callers and two workers. A caller takes the hold with an acquire exchange, or leaves
 *  contended having touched nothing; the holder writes each worker's arena relaxed, then
 *  dispatches with a release on the epoch; each worker acquires the epoch, reads its arena,
 *  and counts itself down with `acq_rel`; the holder waits for the countdown, then drops the
 *  hold with a release. The fan-out is the pool's protocol, which `flat_pool.pml` proves; here
 *  it is only the edge the arenas travel on.
 *
 *  Invariants:
 *  - a worker reads the arena its holder wrote, never one rewound by another caller, and no
 *    arena is rewound while a fan-out is live. `-Dwithout_hold` lets both callers past the
 *    claim, and the second rewinds under the first's fan-out;
 *  - a refused caller writes nothing.
 */
#include "weak_memory.pml"

// The threads, by role, apart from the processes that play them; callers and workers name themselves.
#define workers 2
#define worker_thread(worker) (2 + (worker))

// The words: the hold, each worker's arena as the generation that last rewound it, the epoch, and the countdown.
#define held 0
#define arena(worker) (1 + (worker))
#define epoch 3
#define pending 4

byte callers_started, workers_started;
byte callers_done;   // held or refused, either way finished
byte fanouts_active; // dispatched and not yet joined

active [2] proctype caller() {
    byte me, worker;
    int seen, generation;
    atomic { me = callers_started; callers_started++ };
    generation = me + 1;
#ifndef without_hold
    // claim: the hold or a contended answer, never a wait: ram_device.hpp:176-179
    read_modify_write(me, held, order_acquire, seen, 1);
    if :: seen == 1 -> goto refused :: else fi;
#endif
    // reserve: every arena rewound and seeded, which no live fan-out may be reading: ram_device.hpp:184-191
    assert(fanouts_active == 0);
    for (worker : 0 .. workers - 1) { store(me, arena(worker), order_relaxed, generation) };
    // for_tasks: the fan-out, the pool's release on its epoch and the countdown: ram_device.hpp:225-235
    store(me, pending, order_relaxed, workers);
    atomic { fanouts_active++; store(me, epoch, order_release, generation) };
    (newest_value(pending) == 0);
    load(me, pending, order_acquire, seen);
    fanouts_active--;
#ifndef without_hold
    store(me, held, order_release, 0);
#endif
refused:
    callers_done++
}

active [2] proctype worker() {
    byte me;
    int last_epoch, new_epoch, seen, before;
    atomic { me = workers_started; workers_started++ };
    do
    :: (newest_value(epoch) != last_epoch || callers_done == 2) ->
        load(worker_thread(me), epoch, order_acquire, new_epoch);
        if
        :: new_epoch == last_epoch && callers_done == 2 -> break
        :: new_epoch == last_epoch
        :: else ->
            load(worker_thread(me), arena(me), order_relaxed, seen);
            assert(seen == new_epoch);
            read_modify_write(worker_thread(me), pending, order_acq_rel, before, before - 1);
            last_epoch = new_epoch
        fi
    od
}
