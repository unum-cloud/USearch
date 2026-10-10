/**
 *  @brief A load beside one batch and a searcher: the load under the same gate, its emptiness
 *      checked under the hold, and each shard's key written before the shard plan's `finish`.
 *
 *  The loader takes the gate, checks under it that no row is visible, writes each shard's key at
 *  the append base, and runs the shard plan's `finish`: the claim, then the turn. `insert` takes
 *  the executor's hold and then `mutating_`, and the load's opening, `gate_load_`, takes the gate
 *  too, so a batch through a second executor cannot pass it.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc load_gate=false: `gate_load_` takes the mutation gate the batches take; without
 *      it, a batch through a second executor passes the gate, and its key lands on a row the load
 *      claims
 *  @verify fail sc load_key_before_finish=false: the load writes each shard's key before its plan
 *      finishes, the order the RAM twin had already corrected; after it, the searcher reads the
 *      loaded shard's row before its key
 */
#define thread_count 3
#define location_count 8
#define history_depth 5
#define mutators 1
#include "protocol.pml"

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

// `gate_load_`, then per shard the key at the append base and `load_plan_t::finish`, the claim and the turn
proctype loader(byte t) {
    byte shard;
    int seen, first_id, total, watermark;
    if
    :: load_gate ->
        take_gate(t, seen);
        if :: seen == 1 -> goto contended :: else fi
    :: else
    fi;
    // empty under the hold, as a load requires: `gate_load_` reads `size()`
    load(t, last_visible_id(0), order_acquire, seen);
    load(t, last_visible_id(1), order_acquire, total);
    if :: seen + total != 0 -> goto refused :: else fi;
    for (shard : 0 .. shards - 1) {
        if
        :: load_key_before_finish ->
            first_id = claims[shard]; // plan_load's base, read under the shard's mutex
            store(t, key(shard, first_id), order_relaxed, key_of(t));
            claim_row(t, shard, first_id);
            take_turn(t, shard, first_id, watermark)
        :: else ->
            claim_row(t, shard, first_id);
            take_turn(t, shard, first_id, watermark);
            store(t, key(shard, first_id), order_relaxed, key_of(t))
        fi
    };
refused:
    if
    :: load_gate -> drop_gate(t)
    :: else
    fi;
contended:
    skip
}

init { atomic { run mutator(0); run searcher(1); run loader(2) } }
