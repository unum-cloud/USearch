/**
 *  `dense_vram_indexes` from `include/usearch/dense/vram_indexes.cuh`: the sharded CUDA index,
 *  whose batches spread rows over the shards, each shard growing as it claims, and write every
 *  key before the shard's claim publishes the row; whose one mutation gate answers `contended_k`
 *  to a second batch, and so serializes the key columns' covers too; and whose load lands under
 *  the same gate and in the same key-then-publication order.
 *
 *  Two mutators, one searcher, and under `-Dscenario=loading` one loader, over two shards of
 *  three rows each, one per claimer. A mutator takes the gate with an acquire exchange, then per shard reads the
 *  shard's claim with acquire as the first node, stores the key relaxed, and runs the shard's
 *  insert: a relaxed read of the claim and a release store past it. A first node that the
 *  shard's own claim does not confirm is a claimer outside the gate and the batch stops. The
 *  searcher reads each shard's claim with acquire and every key below it. The loader takes the
 *  gate, checks under it that no row was claimed before it, writes each shard's key at the
 *  append base, and publishes the shard's claim with release.
 *
 *  Invariants:
 *  - ownership: every key a searcher reads below a claim it acquired is the key of the thread
 *    that claimed the row, never unset and never another's. `-Dwithout_gate` lets the mutators
 *    interleave their claims, and one writes its key over a row the other's claim confirmed;
 *    `-Dwithout_key_before_claim` publishes the claim first and the searcher reads an unset key
 *    under every memory model; `-Dwithout_claim_release` publishes relaxed and the key is unset
 *    under views; `-Dscenario=loading -Dwithout_load_gate` runs the load outside the gate, the
 *    header before the fix, and a batch's key lands on a row the load claims;
 *    `-Dwithout_load_key_before_finish` publishes the loaded shard before its key.
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

// The words: the gate, each shard's claim, and each shard's two keys.
#define mutating 0
#define claimed_rows(shard) (1 + (shard))
#define key(shard, id) (3 + (shard) * plane + (id))

// The sizes and values: two shards of three rows - one per claimer - a batch of one row per shard, and the keys.
#define shards 2
#define plane 3
#define unset 0
#define key_of(t) (10 + (t))

#ifdef without_claim_release
#define claim_order order_relaxed
#else
#define claim_order order_release
#endif

byte mutators_started;
byte owner[shards * plane]; // one past the thread that claimed the row; zero while unclaimed
#define owner_of(shard, id) owner[(shard) * plane + (id)]

// the gate: an acquire exchange, and a contended answer when it was held: `insert` and `gate_load_`
inline take_gate(t, seen) { read_modify_write(t, mutating, order_acquire, seen, 1) }
inline drop_gate(t) { store(t, mutating, order_release, 0) }

// the shard's insert: the claim read relaxed, the row's key by the plane's owner, the claim stored
// past it: `dense_vram_index::insert`
inline claim_row(t, shard, first_id) {
    load(t, claimed_rows(shard), order_relaxed, first_id);
    atomic {
        store(t, claimed_rows(shard), claim_order, first_id + 1);
        owner_of(shard, first_id) = (t) + 1
    }
}

active [2] proctype mutator() {
    byte me, shard;
    int seen, first_node, first_id;
    atomic { me = mutators_started; mutators_started++ };
#ifndef without_gate
    take_gate(me, seen);
    if :: seen == 1 -> goto contended :: else fi;
#endif
    for (shard : 0 .. shards - 1) {
        // the first node from the shard's claim, then the key, then the shard's insert: `insert_gated_`
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
#ifndef without_gate
    drop_gate(me);
#endif
contended:
    skip
}

active proctype searcher() {
    byte shard, id;
    int bound, seen;
    for (shard : 0 .. shards - 1) {
        // the shard's claim is the search's bound: `size()`, the node count `stage_walk_` hands the kernel
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
    // empty under the hold, as a load requires: `gate_load_`
    load(loader_thread, claimed_rows(0), order_acquire, seen);
    load(loader_thread, claimed_rows(1), order_acquire, total);
    if :: seen + total != 0 -> goto refused :: else fi;
    // each shard: the key at the append base, then the plan's finish publishing the claim: `load_plan_t::finish`
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
refused:
#ifndef without_load_gate
    drop_gate(loader_thread);
#endif
contended:
    skip
}
#endif
