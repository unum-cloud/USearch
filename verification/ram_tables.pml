/**
 *  @file verification/ram_tables.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief `chunked_multimap`'s probes: three inserters and a prober over one published chunk.
 *
 *  The insert-only linear-probe multimap from `include/usearch/utilities/ram_tables.hpp`, whose
 *  slot states advance `empty` → `building` → the key's probe tag once, whose entries each take a
 *  seat and cover its chunk before any slot is touched, and whose readers walk the control bytes
 *  with acquire loads, wait-free. `ram_tables_growth.pml` covers the chunks a claim publishes. One
 *  published chunk of two seats, two slots and the constant terminator at position 2.
 *
 *  - Bound: no inserter probes past the terminator, and no insert lands past the published seats.
 *  - Visibility: the value a prober reads beside a matching key is that key's value.
 *  - Completeness, under `sc`: a key published before the prober started is found.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail rc11 publish_order=order_relaxed: without it, the prober reads the new key beside the
 *      slot's initial value
 *  @verify fail rc11 probe_order=order_relaxed: without it, the prober reads the new key beside the
 *      slot's initial value
 *  @verify fail sc cover_refusal=false: without the refusal, the seat lands in a chunk never
 *      published
 */
#define thread_count 4
#define location_count 7
#define history_depth 4
#include <weak_memory.pml>

// The words: each slot's control byte, key and value, and the claimed seats; position 2 is the
// constant terminator, never claimed and never a word.
#define state(s) (s)
#define key(s) (2 + (s))
#define value(s) (4 + (s))
#define claimed_seats 6
#define terminator 2

// The values: the published seats, the states of `slot_state_t` and the probe tag both keys share,
// the keys the inserters carry, and the value each key maps to; the keys and values stay apart from
// the zero a word starts at.
#define published_seats 2
#define empty 0
#define building 1
#define tag 133
#define key_a 1
#define key_b 2
#define key_carried(t) ((t) == 2 -> key_b : key_a)
#define value_for(k) (10 + (k))

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef publish_order
#define publish_order order_release
#endif
#ifndef probe_order
#define probe_order order_acquire
#endif
#ifndef cover_refusal
#define cover_refusal true
#endif

bool probing;                // the prober has read its first state
bool published_before_probe; // key A was published before the prober read its first state

proctype inserter(byte t) {
    byte chain;
    int observed, expected, seen;
    bool succeeded;
    // the seat, then its cover, which the directory's last chunk refuses past: `chunked_multimap::claim`
    read_modify_write(t, claimed_seats, order_relaxed, observed, observed + 1);
    if
    :: cover_refusal && observed >= published_seats -> goto refused
    :: else
    fi;
    // the seat's chunk, which the insert indexes: `insert`'s `locate`
    assert(observed < published_seats);
look:
    // the chain: a relaxed peek, then the acquire claim of an empty slot: `insert`'s probe loop and chain step
    assert(chain < terminator);
    load(t, state(chain), order_relaxed, seen);
    if
    :: seen == empty ->
        expected = empty;
        compare_exchange(t, state(chain), order_acquire, expected, building, succeeded);
        if :: succeeded -> goto claimed_slot :: else fi
    :: else
    fi;
    chain++;
    goto look;
claimed_slot:
    // the payload relaxed, then the publication with the release: `insert`'s tail
    store(t, key(chain), order_relaxed, key_carried(t));
    store(t, value(chain), order_relaxed, value_for(key_carried(t)));
    atomic {
        store(t, state(chain), publish_order, tag);
        if :: key_carried(t) == key_a && !probing -> published_before_probe = true :: else fi
    };
refused:
    skip
}

proctype prober(byte t) {
    byte chain;
    int seen, seen_key, seen_value;
    bool found;
walk:
    if
    :: chain == terminator -> goto done
    :: else
    fi;
    // the walk: one acquire load per control byte, the terminator ends it: `find_each`
    atomic {
        load(t, state(chain), probe_order, seen);
        probing = true
    };
    if
    :: seen == empty -> goto done
    :: seen == tag ->
        // the payload, plain reads beside a matching tag: `find_each`'s callback
        load(t, key(chain), order_relaxed, seen_key);
        if
        :: seen_key == key_a ->
            load(t, value(chain), order_relaxed, seen_value);
            assert(seen_value == value_for(key_a));
            found = true
        :: else
        fi
    :: else // building or another tag: occupied, never a match, never the end of the chain
    fi;
    chain++;
    goto walk;
done:
    assert(memory != sc || !published_before_probe || found)
}

init { atomic { run inserter(0); run inserter(1); run inserter(2); run prober(3) } }
