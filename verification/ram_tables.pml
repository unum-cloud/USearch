/**
 *  `growing_multimap` from `include/usearch/utilities/ram_tables.hpp`: the insert-only
 *  linear-probe multimap whose slot states advance `empty` → `building` → `published` once,
 *  whose quota is claimed on `inserted_count_` before any slot is touched, and whose readers
 *  walk the control bytes with acquire loads, wait-free.
 *
 *  Three inserters and one finder over two slots and the constant terminator at position 2,
 *  under a quota of two. Every inserter hashes to chain 0, two carry key A and one key B, so
 *  the counter refuses one of them and the other two contend for the two slots. An inserter
 *  claims quota with a relaxed compare-exchange that never passes the quota, and else probes:
 *  a relaxed load of the state, an acquire compare-exchange from `empty` to `building`, the
 *  key and the value stored relaxed, and the state stored `published` with a release. The
 *  finder walks the chain with acquire loads until the terminator, and on a published slot
 *  carrying key A reads the value. A sizer reads the counter relaxed, as `size()` does.
 *
 *  Invariants:
 *  - bound: no inserter probes past the terminator, the ghost count of admitted entries never
 *    exceeds the quota, and neither does the counter itself, at any instant and under any
 *    memory model - the counter alone bounds every chain, and `size()` never passes
 *    `capacity()`. `-Dwithout_bounded_quota` restores the claim the header had, an
 *    unconditional add handed back with a subtract on refusal, and the sizer reads the
 *    counter one past the quota while a refusal is in flight;
 *  - visibility: the value a finder reads beside a matching key on a published slot is that
 *    key's value, never the slot half-written. The release on the publication and the acquire
 *    on the probe carry the key and value stores: `-Dwithout_publish_release` makes the
 *    publication relaxed, `-Dwithout_probe_acquire` makes the probe relaxed, and each fails on
 *    a finder that reads the new key beside the initial value;
 *  - completeness, under `-Dmemory=sequential`: a key published before the finder started is
 *    found, since a `building` slot is occupied and never the end of the chain.
 */
#include "weak_memory.pml"

// The threads, by role, apart from the processes that play them; the inserters name themselves as they start.
#define finder_thread 3
#define sizer_thread 4

// The words: each slot's control byte, key and value, and the counter; position 2 is the
// constant terminator, never claimed and never a word.
#define state(s) (s)
#define key(s) (2 + (s))
#define value(s) (4 + (s))
#define inserted_count 6
#define terminator 2

// The sizes and values: the quota, the states of `slot_state_t`, the keys the inserters carry,
// and the value each key maps to; the keys and values stay apart from the zero a word starts at.
#define quota 2
#define empty 0
#define building 1
#define published 2
#define key_a 1
#define key_b 2
#define key_carried(me) ((me) == 2 -> key_b : key_a)
#define value_for(k) (10 + (k))

#ifdef without_publish_release
#define publish_order order_relaxed
#else
#define publish_order order_release
#endif

#ifdef without_probe_acquire
#define probe_order order_relaxed
#else
#define probe_order order_acquire
#endif

byte inserters_started;
byte claimed;                // entries the counter admitted
bool probing;                // the finder has read its first state
bool published_before_probe; // key A was published before the finder read its first state

active [3] proctype inserter() {
    byte me, chain;
    int observed, expected, seen;
    bool succeeded;
    atomic { me = inserters_started; inserters_started++ };
#ifdef without_bounded_quota
    // the claim the header had: an unconditional add, handed back with a subtract on refusal
    atomic {
        read_modify_write(me, inserted_count, order_relaxed, observed, observed + 1);
        if :: observed < quota -> claimed++ :: else fi;
        assert(claimed <= quota)
    };
    if
    :: observed >= quota ->
        read_modify_write(me, inserted_count, order_relaxed, observed, observed - 1);
        goto refused
    :: else
    fi;
#else
    // the quota, claimed before any slot is touched and never passed: ram_tables.hpp:417-420
    atomic {
        read_modify_write_if(me, inserted_count, order_relaxed, observed < quota, observed, observed + 1);
        if :: observed < quota -> claimed++ :: else fi;
        assert(claimed <= quota)
    };
    if :: observed >= quota -> goto refused :: else fi;
#endif
probe:
    // the chain: a relaxed peek, then the acquire claim of an empty slot: ram_tables.hpp:421-427, 433
    assert(chain < terminator);
    load(me, state(chain), order_relaxed, seen);
    if
    :: seen == empty ->
        expected = empty;
        compare_exchange(me, state(chain), order_acquire, expected, building, succeeded);
        if :: succeeded -> goto claimed_slot :: else fi
    :: else
    fi;
    chain++;
    goto probe;
claimed_slot:
    // the payload relaxed, then the publication with the release: ram_tables.hpp:428-430
    store(me, key(chain), order_relaxed, key_carried(me));
    store(me, value(chain), order_relaxed, value_for(key_carried(me)));
    atomic {
        store(me, state(chain), publish_order, published);
        if :: key_carried(me) == key_a && !probing -> published_before_probe = true :: else fi
    };
refused:
    skip
}

active proctype finder() {
    byte chain;
    int seen, seen_key, seen_value;
    bool found;
probe:
    if
    :: chain == terminator -> goto done
    :: else
    fi;
    // the walk: one acquire load per control byte, the terminator ends it: ram_tables.hpp:450-452
    atomic {
        load(finder_thread, state(chain), probe_order, seen);
        probing = true
    };
    if
    :: seen == empty -> goto done
    :: seen == published ->
        // the payload, plain reads beside the published state: ram_tables.hpp:454
        load(finder_thread, key(chain), order_relaxed, seen_key);
        if
        :: seen_key == key_a ->
            load(finder_thread, value(chain), order_relaxed, seen_value);
            assert(seen_value == value_for(key_a));
            found = true
        :: else
        fi
    :: else // building: occupied, never a match, never the end of the chain
    fi;
    chain++;
    goto probe;
done:
#if memory == sequential
    assert(!published_before_probe || found)
#else
    skip
#endif
}

// size(): one relaxed load of the counter, which never reads past the quota: ram_tables.hpp:470
active proctype sizer() {
    int seen;
    load(sizer_thread, inserted_count, order_relaxed, seen);
    assert(seen <= quota)
}
