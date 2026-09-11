/**
 *  `basic_shared_mutex` from `include/usearch/basic/ram_index.hpp`: the one-word gate every
 *  mutator of the basic index holds shared, the stop-the-world operations take unique, and
 *  the certified searches bracket with two wait-free probes.
 *
 *  Two mutators each take the shared side once and write their own pair of rows inside it,
 *  the way an insert edits neighbor rows below the watermark. One thread tries the unique
 *  side once and reads the rows. One prober snapshots the gate, reads every row, and
 *  snapshots again, certifying the reads when the word rested unlocked at the same
 *  generation. The word keeps two holder bits, the unique bit and the generation above,
 *  where silicon keeps twenty-two, one and forty-one.
 *
 *  Invariants:
 *  - exclusion: no mutator is inside its section while the unique side is held;
 *  - certification, under `-Dmemory=sequential`: a certified prober saw no acquisition of the gate.
 *    Breaks with `-Dgeneration_modulus=2`, the wrap the header prices at days of churn. Under
 *    the view model the exit probe may read an older word, so the clock-time claim is too
 *    strong there and consistency below is the one that carries;
 *  - consistency: a certified prober never saw one mutator's pair torn. `lock_shared` ends in
 *    a release fence for exactly this: it is an acquire read-modify-write, and without a
 *    release between it and the row stores RC11 lets the exit probe read the old word while a
 *    row read returned a store made under the lock. `-Dwithout_release_fence` drops it, and
 *    the invariant fails. The ISAs forbid the outcome on their own: an acquire read-modify-write
 *    orders the stores after it and the acquire fence orders the reads before the exit probe,
 *    which closes the cycle on Arm, x86 and RVWMO. The fence says so to the C++ model;
 *  - progress: the backed-off holder's wait is capped, so it re-checks on its own. With
 *    `-Dwithout_backoff_cap` the monitor arms after the unique side already left, and Spin
 *    reports the stuck mutator as an invalid end state - why `lock_shared` passes `wait_capped_t`.
 */
#include "weak_memory.pml"

// The threads, by role, apart from the processes that play them; the mutators name themselves as they start.
#define unique_taker_thread 2
#define prober_thread 3

// The words: the gate, and each mutator's pair of rows.
#define gate 0
#define row(mutator, half) (1 + (mutator) * 2 + (half))

// The gate's fields: two holder bits, the unique bit, the generation above, stepping modulo
// `generation_modulus`, which is 2^41 in silicon and small enough to wrap here on request.
#ifndef generation_modulus
#define generation_modulus 16
#endif
#define holders_of(word) ((word) & 3)
#define unique_of(word) (((word) >> 2) & 1)
#define generation_of(word) ((word) >> 3)
#define unique_bit 4
#define stepped(word) (((((word) >> 3) + 1) % generation_modulus) << 3) | ((word) & 7)

byte mutators_started;
byte reading;    // mutators inside their section
bool unique_held;
bool walking;    // the prober is between its snapshots
bool disturbed;  // the gate was acquired while it was

active [2] proctype mutator() {
    byte me;
    int observed, tag;
    atomic { me = mutators_started; mutators_started++; tag = me + 1 };
lock_shared:
    // one blind add of a holder and a generation step, backed off behind a unique holder
    atomic {
        read_modify_write(me, gate, order_acquire, observed, stepped(observed) + 1);
        if :: walking -> disturbed = true :: else fi
    };
    if
    :: unique_of(observed) ->
        read_modify_write(me, gate, order_release, observed, observed - 1);
#ifdef without_backoff_cap
        // a monitor armed on the word alone: deadlocks when the unique side left between the add and the sub
        (newest_value(gate) != observed - 1);
#else
        // the capped wait: re-checks on its own, so the retry comes once the unique side is gone
        (!unique_of(newest_value(gate)));
#endif
        goto lock_shared
    :: else
    fi;
#ifndef without_release_fence
    fence_release(me);
#endif
    reading++;
    assert(!unique_held);
    store(me, row(me, 0), order_relaxed, tag);
    store(me, row(me, 1), order_relaxed, tag);
    reading--;
    // unlock_shared: one release subtract
    read_modify_write(me, gate, order_release, observed, observed - 1)
}

active proctype unique_taker() {
    int observed, first, second;
    bool succeeded;
    load(unique_taker_thread, gate, order_relaxed, observed);
try_lock:
    // only from a genuinely free word; a refusal is the answer
    if
    :: holders_of(observed) == 0 && !unique_of(observed) ->
        atomic {
            compare_exchange(unique_taker_thread, gate, order_acquire, observed, stepped(observed) | unique_bit, succeeded);
            if :: succeeded && walking -> disturbed = true :: else fi
        };
        if
        :: !succeeded -> goto try_lock
        :: else
        fi;
        unique_held = true;
        assert(reading == 0);
        load(unique_taker_thread, row(0, 0), order_relaxed, first);
        load(unique_taker_thread, row(0, 1), order_relaxed, second);
        assert(first == second); // the unique side reads whole pairs: every mutator released before it
        unique_held = false;
        // unlock: clear the unique bit with a release
        read_modify_write(unique_taker_thread, gate, order_release, observed, observed - unique_bit)
    :: else
    fi
}

active proctype prober() {
    int entry, exit, first_of_0, second_of_0, first_of_1, second_of_1;
    atomic {
        load(prober_thread, gate, order_acquire, entry);
        walking = true;
        disturbed = false
    };
    if
    :: holders_of(entry) != 0 || unique_of(entry) -> walking = false; goto refused
    :: else
    fi;
    load(prober_thread, row(0, 0), order_relaxed, first_of_0);
    load(prober_thread, row(0, 1), order_relaxed, second_of_0);
    load(prober_thread, row(1, 0), order_relaxed, first_of_1);
    load(prober_thread, row(1, 1), order_relaxed, second_of_1);
    // the fence keeps the walk's reads from sinking below the reload, as the header says
    fence_acquire(prober_thread);
    atomic {
        load(prober_thread, gate, order_acquire, exit);
        walking = false;
        if
        :: holders_of(exit) == 0 && !unique_of(exit) && generation_of(exit) == generation_of(entry) ->
#if memory == sequential
            assert(!disturbed);
#endif
            assert(first_of_0 == second_of_0 && first_of_1 == second_of_1)
        :: else
        fi
    };
refused:
    skip
}
