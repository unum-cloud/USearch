/**
 *  @brief The sharded index's claims: two batch adders and a single-row adder on two shards.
 *
 *  `include/usearch/dense/ram_indexes.hpp` claims each row from its shard's own capacity counter,
 *  one shard at a time, and releases a refused claim with a release subtract. Two shards of two
 *  rows each; two batch adders bring one row per shard, and one single-row adder probes the shards
 *  from its routing base; the dispenser is a ghost, and a row's publication is its watermark turn.
 *  The batch adder claims shard by shard, releases every shard claimed so far on a refusal, and may
 *  have its fan-out refused after the claims, which releases them all. A checker outside the memory
 *  model compares each shard's claim word with its dispenser once every adder is done.
 *
 *  The shard's claim is `claim_capacity` seen from outside: it grows the shard on demand and
 *  refuses only past the rows one shard's IDs reach, the model's row limit.
 *  `dense_ram_index_capacity/` covers the counter itself.
 *
 *  Invariants:
 *  - the row limit: a dispensed ID is below its shard's row limit, and once every adder is done
 *    each shard's claim word equals its dispenser;
 *  - publication: every ID below a watermark a reader acquired has its key.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 key_before_watermark=false: `on_assigned` stores the key before the
 *      watermark turn, so every ID below a watermark a searcher acquired resolves through the key
 *      column; after the turn, the searcher reads an unset key under every model
 *  @verify fail rc11 watermark_order=order_relaxed: the watermark's release carries the key stored
 *      before it; relaxed, the searcher reads an unset key below the watermark it acquired
 *  @verify fail sc release_on_refusal=false: ForkUnion runs every task or none, so a refused
 *      fan-out releases every claim whole; without the release, the claim word stays above the
 *      dispenser
 *  @verify fail sc claim_consume=false: the single-row `insert` consumes its claim once the row is
 *      dispensed against; without it, the landed row is released, the claim word falls below the
 *      dispenser, and with two batches behind it the shard claims one row more than it holds
 */
#define thread_count 4
#define location_count 9
#define history_depth 7
#define shards 2
#define row_limit 2
#include "weak_memory.pml"
#include "watermark_turn.pml"

// The words, by the index member each stands for, per shard.
#define routed_rows 0
#define claimed(shard) (1 + (shard))
#define last_visible_id(shard) (3 + (shard))
#define key(shard, id) (5 + (shard) * row_limit + (id))

// The values a key slot carries: unset, or the key of the thread that landed the row.
#define unset 0
#define key_of(thread) (10 + (thread))

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef key_before_watermark
#define key_before_watermark true
#endif
#ifndef watermark_order
#define watermark_order order_release
#endif
#ifndef release_on_refusal
#define release_on_refusal true
#endif
#ifndef claim_consume
#define claim_consume true
#endif

byte adders_done;       // the checker waits on it under `sc`
byte dispensed[shards]; // each shard's `next_dispensed_id_`, a ghost, since no plan reads it

inline dispense(shard, id) {
    atomic {
        id = dispensed[shard];
        dispensed[shard]++;
        assert(id < row_limit) // the claim is what keeps an ID inside the row limit
    }
}

// `insert`'s tail, for one row of one shard: the key before the turn, then the turn, whose acquire
// load is what merges the predecessor's publication into this one
inline land_row(t, shard, id) {
    int watermark;
    // on_assigned fires right after the dispense: one step for the two
    atomic {
        dispense(shard, id);
        if
        :: key_before_watermark -> store(t, key(shard, id), order_relaxed, key_of(t))
        :: else
        fi
    };
    take_turn(t, last_visible_id(shard), id, id + 1, watermark_order, watermark);
    if
    :: !key_before_watermark -> store(t, key(shard, id), order_relaxed, key_of(t))
    :: else
    fi
}

// the shard's own row claim up to its row limit, `claim_capacity` seen from outside: one
// unconditional add whose value says what happened, a fit; the crossing, which elects this
// claimer and the row limit then refuses; or a peer's refusal still in flight, handed back and
// waited out. `before + 1 <= row_limit` afterwards names a fit
inline claim(t, shard, before) {
    do
    :: read_modify_write(t, claimed(shard), order_acquire, before, before + 1);
        if
        :: before + 1 <= row_limit -> break
        :: before == row_limit -> add_no_return(t, claimed(shard), order_release, 0 - 1); break
        :: else -> add_no_return(t, claimed(shard), order_release, 0 - 1); (newest_value(claimed(shard)) <= row_limit)
        fi
    od
}

// the claims' destructors: every shard claimed so far, from shard zero, releases its row
inline release_up_to(t, count) {
    byte back;
    do
    :: back == count -> break
    :: else -> add_no_return(t, claimed(back), order_release, 0 - 1); back++
    od
}

proctype batch_adder(byte t) {
    byte shard;
    int routed, rotation, before, id;
    read_modify_write(t, routed_rows, order_relaxed, routed, routed + shards);
    rotation = routed % shards;
    // claim: shard by shard, or into none
    do
    :: shard == shards -> break
    :: else ->
        claim(t, shard, before);
        if
        :: before + 1 <= row_limit -> shard++
        :: else -> release_up_to(t, shard); goto refused
        fi
    od;
    // the fan-out refused: ForkUnion runs every task or none, so the claims go back whole
    if
    :: if
        :: release_on_refusal -> release_up_to(t, shards)
        :: else
        fi;
        goto refused
    :: skip
    fi;
    land_row(t, rotation, id);
    land_row(t, (rotation + 1) % shards, id);
refused:
    if
    :: memory == sc -> adders_done++
    :: else
    fi
}

proctype single_adder(byte t) {
    byte probe, shard;
    int routed, before, id;
    read_modify_write(t, routed_rows, order_relaxed, routed, routed + 1);
    // the routing tally names the first shard to try; a full one passes the row on to the next
    do
    :: probe == shards -> goto refused
    :: else ->
        shard = (routed + probe) % shards;
        claim(t, shard, before);
        if
        :: before + 1 <= row_limit -> break
        :: else -> probe++
        fi
    od;
    // reserve_traversal refused: only this shard's row goes back
    if
    :: add_no_return(t, claimed(shard), order_release, 0 - 1); goto refused
    :: skip
    fi;
    land_row(t, shard, id);
    if
    :: !claim_consume -> add_no_return(t, claimed(shard), order_release, 0 - 1)
    :: else
    fi;
refused:
    if
    :: memory == sc -> adders_done++
    :: else
    fi
}

// the search merge: every ID below the watermark it acquired resolves through the key column
proctype searcher(byte t) {
    byte shard, id;
    int watermark, seen;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermark);
        id = 0;
        do
        :: id == watermark -> break
        :: else -> load(t, key(shard, id), order_relaxed, seen); assert(seen != unset); id++
        od
    }
}

// once every adder is done: each shard's claim word equals its dispenser, and every row that took
// its turn, which is when the locator map records it, has its key
proctype checker() {
    byte shard;
    if
    :: memory == sc ->
        (adders_done == 3);
        for (shard : 0 .. shards - 1) {
            assert(newest_value(claimed(shard)) == dispensed[shard]);
            assert(newest_value(last_visible_id(shard)) <= 0 ||
                   newest_value(key(shard, 0)) != unset);
            assert(newest_value(last_visible_id(shard)) <= 1 ||
                   newest_value(key(shard, 1)) != unset)
        }
    :: else
    fi
}

init { atomic { run batch_adder(0); run batch_adder(1); run single_adder(2); run searcher(3); run checker() } }
