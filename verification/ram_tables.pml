/**
 *  `chunked_multimap` from `include/usearch/utilities/ram_tables.hpp`: the insert-only
 *  linear-probe multimap whose slot states advance `empty` → `building` → `published` once,
 *  whose entries each take a seat and cover its chunk before any slot is touched, whose chunks
 *  any claimer publishes with a compare-exchange, and whose readers walk the control bytes with
 *  acquire loads, wait-free.
 *
 *  `-Dscenario=probe`, the default: three inserters and one prober over one published chunk of
 *  two seats, two slots and the constant terminator at position 2. Every inserter hashes to
 *  chain 0, two carry key A and one key B, so the cover refuses one of them - its seat lies past
 *  the last chunk the directory holds - and the other two contend for the two slots. An inserter
 *  takes a seat with a relaxed add, is refused when its cover fails, and else probes: a relaxed load of the state, an acquire compare-exchange from `empty` to
 *  `building`, the key and the value stored relaxed, and the state stored `published` with a
 *  release. The prober walks the chain with acquire loads until the terminator, and on a
 *  published slot carrying key A reads the value.
 *
 *  `-Dscenario=growth`: two inserters claim one seat each from an empty map whose chunks hold one
 *  seat and one slot apiece, and a prober walks every chunk it sees published. A claim covers
 *  every chunk up to its seat's: a missing one it allocates, zeroes from the allocation's
 *  garbage and publishes into the directory with a compare-exchange, and a peer that published
 *  first keeps its own; the insert then reaches its chunk through the directory.
 *
 *  Invariants:
 *  - bound: no inserter probes past the terminator, and no insert lands past the published
 *    seats, so every insert finds its table. `-Dwithout_cover_refusal` inserts although the
 *    cover failed, and the seat lands in a chunk never published;
 *  - visibility: the value a prober reads beside a matching key on a published slot is that
 *    key's value, never the slot half-written. The release on the publication and the acquire
 *    on the probe carry the key and value stores: `-Dwithout_publish_release` makes the
 *    publication relaxed, `-Dwithout_probe_acquire` makes the probe relaxed, and each fails on
 *    a prober that reads the new key beside the initial value;
 *  - completeness, under `-Dmemory=sequential`: a key published before the prober started is
 *    found, since a `building` slot is occupied and never the end of the chain;
 *  - growth: an inserter's chunk and a prober's are never read before their publisher zeroed
 *    them. The publishing compare-exchange releases the zeroing, and the directory loads acquire
 *    it: `-Dwithout_cover_release`, `-Dwithout_directory_acquire` and `-Dwithout_walk_acquire`
 *    each fail on a control byte still holding the garbage.
 */
#include "weak_memory.pml"

// The scenarios: probes over one published chunk, or claims that grow the chunks.
#define probe 1
#define growth 2
#ifndef scenario
#define scenario probe
#endif

#if scenario == probe

// The threads, by role, apart from the processes that play them; the inserters name themselves as they start.
#define prober_thread 3

// The words: each slot's control byte, key and value, and the claimed seats; position 2 is the
// constant terminator, never claimed and never a word.
#define state(s) (s)
#define key(s) (2 + (s))
#define value(s) (4 + (s))
#define claimed_seats 6
#define terminator 2

// The sizes and values: the published seats, the states of `slot_state_t`, the keys the
// inserters carry, and the value each key maps to; the keys and values stay apart from the zero
// a word starts at.
#define published_seats 2
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
bool probing;                // the prober has read its first state
bool published_before_probe; // key A was published before the prober read its first state

active [3] proctype inserter() {
    byte me, chain;
    int observed, expected, seen;
    bool succeeded;
    atomic { me = inserters_started; inserters_started++ };
    // the seat, then its cover, which the directory's last chunk refuses past: `chunked_multimap::claim`
    read_modify_write(me, claimed_seats, order_relaxed, observed, observed + 1);
#ifndef without_cover_refusal
    if :: observed >= published_seats -> goto refused :: else fi;
#endif
    // the seat's chunk, which the insert indexes: `insert`'s `locate`
    assert(observed < published_seats);
look:
    // the chain: a relaxed peek, then the acquire claim of an empty slot: `insert`'s probe loop and chain step
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
    goto look;
claimed_slot:
    // the payload relaxed, then the publication with the release: `insert`'s tail
    store(me, key(chain), order_relaxed, key_carried(me));
    store(me, value(chain), order_relaxed, value_for(key_carried(me)));
    atomic {
        store(me, state(chain), publish_order, published);
        if :: key_carried(me) == key_a && !probing -> published_before_probe = true :: else fi
    };
refused:
    skip
}

active proctype prober() {
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
        load(prober_thread, state(chain), probe_order, seen);
        probing = true
    };
    if
    :: seen == empty -> goto done
    :: seen == published ->
        // the payload, plain reads beside the published state: `find_each`'s callback
        load(prober_thread, key(chain), order_relaxed, seen_key);
        if
        :: seen_key == key_a ->
            load(prober_thread, value(chain), order_relaxed, seen_value);
            assert(seen_value == value_for(key_a));
            found = true
        :: else
        fi
    :: else // building: occupied, never a match, never the end of the chain
    fi;
    chain++;
    goto walk;
done:
#if memory == sequential
    assert(!published_before_probe || found)
#else
    skip
#endif
}

#elif scenario == growth

// The threads, by role, apart from the processes that play them; the inserters name themselves as they start.
#define prober_thread 2

// The words: the claimed seats, each chunk's directory entry - zero, or one past the thread whose allocation it
// holds - and each thread's own allocation of each chunk, one control byte apiece. A chunk holds one seat.
#define claimed_seats 0
#define directory(c) (1 + (c))
#define allocation(t, c) (3 + (t) * 2 + (c))

// The sizes and values: the chunks the directory holds, and the states, whose `empty` stays
// apart from the garbage a fresh allocation - a word's starting zero - holds.
#define max_chunks 2
#define garbage 0
#define building 1
#define published 2
#define empty 3

#ifdef without_cover_release
#define publish_chunk_order order_acquire
#else
#define publish_chunk_order order_acq_rel
#endif

#ifdef without_directory_acquire
#define directory_order order_relaxed
#undef publish_chunk_order
#define publish_chunk_order order_release
#else
#define directory_order order_acquire
#endif

#ifdef without_walk_acquire
#define walk_order order_relaxed
#else
#define walk_order order_acquire
#endif

byte inserters_started;

active [2] proctype inserter() {
    byte me, chunk;
    int seat, owner, expected, seen;
    bool succeeded;
    atomic { me = inserters_started; inserters_started++ };
    // the seat, then every chunk up to its own covered: `chunked_multimap::claim` and `chunked_column::cover`
    read_modify_write(me, claimed_seats, order_relaxed, seat, seat + 1);
    do
    :: chunk > seat -> break
    :: else ->
        load(me, directory(chunk), directory_order, owner);
        if
        :: owner == 0 ->
            // a fresh allocation, zeroed, then offered to the directory; a peer's earlier one wins
            store(me, allocation(me, chunk), order_relaxed, empty);
            expected = 0;
            compare_exchange(me, directory(chunk), publish_chunk_order, expected, me + 1, succeeded)
        :: else
        fi;
        chunk++
    od;
    // the seat's chunk through the directory, zeroed by whoever published it: `insert`
    load(me, directory(seat), directory_order, owner);
    assert(owner != 0);
    load(me, allocation(owner - 1, seat), order_relaxed, seen);
    assert(seen != garbage);
    seen = empty;
    compare_exchange(me, allocation(owner - 1, seat), order_acquire, seen, building, succeeded);
    assert(succeeded);
    store(me, allocation(owner - 1, seat), order_release, published)
}

active proctype prober() {
    byte chunk;
    int owner, seen;
    // every published chunk, stopping at the first missing one: `find_each`
    do
    :: chunk == max_chunks -> break
    :: else ->
        load(prober_thread, directory(chunk), walk_order, owner);
        if
        :: owner == 0 -> break
        :: else
        fi;
        load(prober_thread, allocation(owner - 1, chunk), order_acquire, seen);
        assert(seen != garbage);
        chunk++
    od
}

#endif
