/**
 *  @brief `basic::shared_mutex`: the one-word gate every mutator of the basic index holds shared,
 *      the stop-the-world operations take unique, and the certified searches bracket with two probes.
 *
 *  From `include/usearch/basic/ram_index.hpp`. Two mutators each take the shared side once and
 *  write their own pair of rows inside it, the way an insert edits neighbor rows below the
 *  watermark. One thread tries the unique side once and reads the rows. One prober snapshots the
 *  gate, reads every row, and snapshots again, certifying the reads when the word rested unlocked
 *  at the same generation. The word keeps two holder bits, the unique bit and the generation
 *  above, where silicon keeps twenty-two, one and forty-one.
 *
 *  No mutator is inside its section while the unique side is held, and a certified prober never
 *  sees one mutator's pair torn. Under `sc` a certified prober also saw no acquisition of the gate;
 *  under `rc11` the exit probe may read an older word, so that clock-time claim is too strong
 *  there, and the torn pair is the claim that carries.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc generation_modulus=2: the generation steps modulo 2^41, so wrapping it inside one
 *      probe bracket takes days of saturated acquisitions; modulo 2 it wraps in three steps, and a
 *      probe a mutator ran through is certified
 *  @verify fail rc11 lock_shared_fence=false: `lock_shared` ends in a release fence, since its
 *      acquire read-modify-write releases nothing before the row stores; without it, a certified
 *      prober reads a store made under the lock while its exit probe still reads the untouched word,
 *      a torn edit certified as quiet. `basic_shared_mutex.cpp` replays it under IMM too, so Arm
 *      and POWER need the fence as much as the C++ model does
 *  @verify fail rc11 probe_fence=false: the certified walk's acquire fence before its exit probe is
 *      the other half of that pair, so a row read that met a store made under the lock moves the
 *      exit probe past the lock's add; without it, the exit probe reads the untouched word and
 *      certifies the torn pair
 *  @verify fail rc11 unlock_shared_order=order_relaxed: `unlock_shared` subtracts its holder with a
 *      release, which carries the edits made under the shared side to the unique side; relaxed, the
 *      unique taker reads a mutator's pair torn
 *  @verify fail rc11 try_lock_order=order_relaxed: `try_lock` takes the unique side with an acquire
 *      compare-exchange, the other half of that pair; relaxed, the unique taker reads a mutator's
 *      pair torn
 *  @verify stuck sc backoff_cap=false: a mutator backing off behind a unique holder waits capped,
 *      so it re-checks on its own; armed on the word alone, its monitor arms after the holder already
 *      left, and the mutator waits forever. The same shape governs the deficit wait in
 *      `claim_capacity`
 */
#define thread_count 4
#define location_count 5
#define history_depth 11
#include "weak_memory.pml"

// The words: the gate, and each mutator's pair of rows.
#define gate 0
#define row(mutator, half) (1 + (mutator) * 2 + (half))

// The gate's fields: two holder bits, the unique bit, and the generation above, stepping modulo
// `generation_modulus`, which is 2^41 in silicon and small enough to wrap here on request.
#define holders_of(word) ((word) & 3)
#define unique_of(word) (((word) >> 2) & 1)
#define generation_of(word) ((word) >> 3)
#define unique_bit 4
#define stepped(word) (((((word) >> 3) + 1) % generation_modulus) << 3) | ((word) & 7)

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef generation_modulus
#define generation_modulus 16
#endif
#ifndef lock_shared_fence
#define lock_shared_fence true
#endif
#ifndef backoff_cap
#define backoff_cap true
#endif
#ifndef unlock_shared_order
#define unlock_shared_order order_release
#endif
#ifndef try_lock_order
#define try_lock_order order_acquire
#endif
#ifndef probe_fence
#define probe_fence true
#endif

byte reading;   // mutators inside their section
bool unique_held;
bool walking;   // the prober is between its snapshots
bool disturbed; // the gate was acquired while it was

proctype mutator(byte t) {
    int observed;
lock_shared:
    // one blind add of a holder and a generation step, backed off behind a unique holder
    atomic {
        read_modify_write(t, gate, order_acquire, observed, stepped(observed) + 1);
        if :: walking -> disturbed = true :: else fi
    };
    if
    :: unique_of(observed) ->
        read_modify_write(t, gate, order_release, observed, observed - 1);
        // the capped wait re-checks on its own; a monitor armed on the word alone misses a holder that already left
        if
        :: backoff_cap -> (!unique_of(newest_value(gate)))
        :: else -> (newest_value(gate) != observed - 1)
        fi;
        goto lock_shared
    :: else
    fi;
    if
    :: lock_shared_fence -> fence_release(t)
    :: else
    fi;
    reading++;
    assert(!unique_held);
    store(t, row(t, 0), order_relaxed, t + 1);
    store(t, row(t, 1), order_relaxed, t + 1);
    reading--;
    // unlock_shared: one release subtract
    read_modify_write(t, gate, unlock_shared_order, observed, observed - 1)
}

proctype unique_taker(byte t) {
    int observed, first, second;
    bool succeeded;
    load(t, gate, order_relaxed, observed);
try_lock:
    // only from a genuinely free word; a refusal is the answer
    if
    :: holders_of(observed) == 0 && !unique_of(observed) ->
        atomic {
            compare_exchange(t, gate, try_lock_order, observed, stepped(observed) | unique_bit, succeeded);
            if :: succeeded && walking -> disturbed = true :: else fi
        };
        if
        :: !succeeded -> goto try_lock
        :: else
        fi;
        unique_held = true;
        assert(reading == 0);
        load(t, row(0, 0), order_relaxed, first);
        load(t, row(0, 1), order_relaxed, second);
        assert(first == second); // the unique side reads whole pairs: every mutator released before it
        unique_held = false;
        // unlock: clear the unique bit with a release
        read_modify_write(t, gate, order_release, observed, observed - unique_bit)
    :: else
    fi
}

proctype prober(byte t) {
    int entry, exit, first_of_0, second_of_0, first_of_1, second_of_1;
    atomic {
        load(t, gate, order_acquire, entry);
        walking = true;
        disturbed = false
    };
    if
    :: holders_of(entry) != 0 || unique_of(entry) -> walking = false; goto refused
    :: else
    fi;
    load(t, row(0, 0), order_relaxed, first_of_0);
    load(t, row(0, 1), order_relaxed, second_of_0);
    load(t, row(1, 0), order_relaxed, first_of_1);
    load(t, row(1, 1), order_relaxed, second_of_1);
    // the fence keeps the walk's reads from sinking below the reload, as the header says
    if
    :: probe_fence -> fence_acquire(t)
    :: else
    fi;
    atomic {
        load(t, gate, order_acquire, exit);
        walking = false;
        if
        :: holders_of(exit) == 0 && !unique_of(exit) && generation_of(exit) == generation_of(entry) ->
            assert(memory != sc || !disturbed);
            assert(first_of_0 == second_of_0 && first_of_1 == second_of_1)
        :: else
        fi
    };
refused:
    skip
}

init { atomic { run mutator(0); run mutator(1); run unique_taker(2); run prober(3) } }
