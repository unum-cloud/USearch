/**
 *  `dense_vram_indexes` from `include/usearch/dense/vram_indexes.cuh`: the sharded CUDA index,
 *  whose batches admit rows against the key planes' ceiling, spread them over the shards, and
 *  write every key before the shard's claim publishes the row; whose one mutation gate answers
 *  `contended_k` to a second batch; and whose load lands under the same gate and in the same
 *  key-then-publication order.
 *
 *  Two mutators, one searcher, and under `-Dscenario=loading` one loader, over two shards of
 *  two rows each. A mutator takes the gate with an acquire exchange, admits two rows with a
 *  relaxed compare-exchange on `admitted_rows` against the ceiling, then per shard reads the
 *  shard's claim with acquire as the first node, stores the key relaxed, and runs the shard's
 *  insert: a relaxed read of the claim and a release store past it. A first node that the
 *  shard's own claim does not confirm is a claimer outside the gate and the batch stops. The
 *  searcher reads each shard's claim with acquire and every key below it. The loader takes the
 *  gate, writes each shard's key at the append base, publishes the shard's claim with release,
 *  and stores the total on `admitted_rows` relaxed, after checking under the gate that no row
 *  was claimed before it.
 *
 *  Invariants:
 *  - ownership: every key a searcher reads below a claim it acquired is the key of the thread
 *    that claimed the row, never unset and never another's. `-Dwithout_gate` lets the mutators
 *    interleave their claims: the shard refuses the second, which stops its batch with rows
 *    admitted that no shard claimed, the ceiling below; `-Dwithout_key_before_claim`
 *    publishes the claim first and the searcher reads an unset key under every memory model;
 *    `-Dwithout_claim_release` publishes relaxed and the key is unset under views;
 *  - the ceiling: no first node reaches the plane, and `admitted_rows` ends equal to the rows
 *    the shards claimed. `-Dscenario=loading -Dwithout_load_gate` runs the load outside the
 *    gate, the header before the fix, and a batch's claim lands between the load's claim and
 *    its total; `-Dwithout_empty_under_gate` checks emptiness only before the gate, and a
 *    batch admitted in between is overwritten by the load's total, so the next batch's first
 *    node reaches the plane; `-Dwithout_load_key_before_finish` publishes the loaded shard
 *    before its key, the last third of that fix.
 */
#include "weak_memory.pml"

// The knob's values are integers, so a typo fails the range check below.
#define batches 1
#define loading 2
#ifndef scenario
#define scenario batches
#endif
#if scenario < batches || scenario > loading
#error "scenario is batches or loading"
#endif

// The threads, by role, apart from the processes that play them; the mutators name themselves.
#define searcher_thread 2
#define loader_thread 3

// The words: the gate, the admission counter, each shard's claim, and each shard's two keys.
#define mutating 0
#define admitted_rows 1
#define claimed_rows(shard) (2 + (shard))
#define key(shard, id) (4 + (shard) * 2 + (id))

// The sizes and values: two shards of two rows, a batch of one row per shard, and the keys.
#define shards 2
#define plane 2
#define capacity (shards * plane)
#define batch_rows shards
#define unset 0
#define key_of(t) (10 + (t))

#ifdef without_claim_release
#define claim_order order_relaxed
#else
#define claim_order order_release
#endif

byte mutators_started;
byte claimers_done; // mutators and the loader that returned, whatever they answered
byte owner[shards * plane]; // one past the thread that claimed the row; zero while unclaimed
#define owner_of(shard, id) owner[(shard) * plane + (id)]

// the gate: an acquire exchange, and a contended answer when it was held: vram_indexes.cuh:287-288, 1181-1182
inline take_gate(t, seen) { read_modify_write(t, mutating, order_acquire, seen, 1) }
inline drop_gate(t) { store(t, mutating, order_release, 0) }

// the shard's insert: the claim read relaxed, the row's key by the plane's owner, the claim stored past it: vram_index.cuh:2840, 2879
inline claim_row(t, shard, first_id) {
    load(t, claimed_rows(shard), order_relaxed, first_id);
    atomic {
        store(t, claimed_rows(shard), claim_order, first_id + 1);
        owner_of(shard, first_id) = (t) + 1
    }
}

active [2] proctype mutator() {
    byte me, shard;
    int seen, base, first_node, first_id;
    bool admitted;
    atomic { me = mutators_started; mutators_started++ };
#ifndef without_gate
    take_gate(me, seen);
    if :: seen == 1 -> goto contended :: else fi;
#endif
    // admission against the ceiling, a relaxed compare-exchange: vram_indexes.cuh:694-699
    load(me, admitted_rows, order_relaxed, base);
    do
    :: capacity - base < batch_rows -> goto refused
    :: else ->
        compare_exchange(me, admitted_rows, order_relaxed, base, base + batch_rows, admitted);
        if :: admitted -> break :: else fi
    od;
    for (shard : 0 .. shards - 1) {
        // the first node from the shard's claim, then the key, then the shard's insert: vram_indexes.cuh:712-726
        load(me, claimed_rows(shard), order_acquire, first_node);
        assert(first_node < plane);
#ifdef without_key_before_claim
        claim_row(me, shard, first_id);
        store(me, key(shard, first_node), order_relaxed, key_of(me));
#else
        store(me, key(shard, first_node), order_relaxed, key_of(me));
        claim_row(me, shard, first_id);
#endif
        // a first node the shard did not confirm is a claimer outside the gate: the batch stops
        if :: first_id != first_node -> break :: else fi
    };
refused:
#ifndef without_gate
    drop_gate(me);
#endif
contended:
    claimers_done++
}

active proctype searcher() {
    byte shard, id;
    int bound, seen;
    for (shard : 0 .. shards - 1) {
        // the shard's claim is the search's bound: vram_index.cuh:1946, 2971
        load(searcher_thread, claimed_rows(shard), order_acquire, bound);
        id = 0;
        do
        :: id < bound ->
            load(searcher_thread, key(shard, id), order_relaxed, seen);
            assert(owner_of(shard, id) != 0 && seen == key_of(owner_of(shard, id) - 1));
            id++
        :: else -> break
        od
    }
}

#if scenario == loading
active proctype loader() {
    byte shard;
    int seen, first_id, total;
#ifndef without_load_gate
    take_gate(loader_thread, seen);
    if :: seen == 1 -> goto contended :: else fi;
#endif
#ifndef without_empty_under_gate
    // empty under the hold, since the total stored at the end would overwrite a batch admitted before it: vram_indexes.cuh:1185
    load(loader_thread, claimed_rows(0), order_acquire, seen);
    load(loader_thread, claimed_rows(1), order_acquire, total);
    if :: seen + total != 0 -> goto refused :: else fi;
#endif
    // each shard: the key at the append base, then the plan's finish publishing the claim: vram_indexes.cuh:1153-1166
    for (shard : 0 .. shards - 1) {
#ifdef without_load_key_before_finish
        claim_row(loader_thread, shard, first_id);
        store(loader_thread, key(shard, first_id), order_relaxed, key_of(loader_thread));
#else
        load(loader_thread, claimed_rows(shard), order_acquire, first_id); // plan_load's base is size()
        store(loader_thread, key(shard, first_id), order_relaxed, key_of(loader_thread));
        claim_row(loader_thread, shard, first_id);
#endif
    };
    // the total, stored relaxed once every shard landed: vram_indexes.cuh:1242
    total = shards;
    store(loader_thread, admitted_rows, order_relaxed, total);
refused:
#ifndef without_load_gate
    drop_gate(loader_thread);
#endif
contended:
    claimers_done++
}
#endif

// the ceiling, once every claimer returned: the admitted rows are exactly the rows the shards claimed
active proctype auditor() {
    int seen;
    (claimers_done == 2 + (scenario == loading));
    seen = newest_value(claimed_rows(0)) + newest_value(claimed_rows(1));
    assert(newest_value(admitted_rows) == seen)
}
