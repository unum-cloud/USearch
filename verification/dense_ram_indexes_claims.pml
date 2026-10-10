/**
 *  @brief The sharded index's claims: two planned batches, a dropped plan and a single-row insert on
 *      two shards, beside a snapshot.
 *
 *  `include/usearch/dense/ram_indexes.hpp` claims a batch's rows from every shard's own capacity
 *  counter in `plan_insert`, one shard at a time, and a refused claim releases every share granted
 *  as the claims die. Two shards of two rows each. Two planners bring one row per shard: `insert`
 *  dispenses both shards' ranges in one step under the dispensing mutex, consuming the claims, and
 *  a refused fan-out lands the rows on the calling thread; `finish` awaits each shard's turn, then
 *  stores the watermarks inside the bracket of the commit sequence word. A third planner drops its
 *  plan before `insert`, and a single-row `insert` probes the shards from its routing base and may
 *  be refused after its claim. A searcher takes `snapshot_`. The dispenser is a ghost, and a row's
 *  wiring writes no word watched here. Every batch routes an even count of rows, so the single-row
 *  insert's routing base is shard zero in every order, and `routed_rows_` is left out.
 *
 *  The shard's claim is `claim_capacity` seen from outside: it grows the shard on demand and
 *  refuses only past the rows one shard's IDs reach, the model's row limit.
 *  `dense_ram_index_capacity/` covers the counter itself, and `dense_ram_indexes_commits.pml` the
 *  bracket's orders.
 *
 *  Invariants:
 *  - quiescence, checked under `sc` alone: each shard's claim word and watermark equal its dispenser;
 *  - the snapshot: each planned batch is on both shards or on neither.
 *
 *  @verify pass sc
 *  @verify stuck rc11 release_after_dispense=true: `insert` dispenses before its fan-out, so a
 *      refused fan-out lands the rows on the caller; releasing the claims there instead abandons
 *      dispensed IDs, and the next range on their shard waits for a turn no store gives; under `sc`
 *      the quiescence check finds the claim word below the dispenser first
 *  @verify stuck sc dispensing_mutex=false: the dispensing mutex keeps two batches' ranges in one
 *      order on every shard; dispensing each shard on its own, the ranges cross between the shards,
 *      and each `finish` waits for a turn only the other's stores give
 *  @verify fail sc claim_consume=false: the dispense consumes the claim it draws on; unconsumed,
 *      the claim releases its rows when the plan dies, and the claim word falls below the dispenser
 */
#define thread_count 5
#define location_count 5
#define history_depth 9
#define shards 2
#define row_limit 2
#define planners 2
#include "weak_memory.pml"
#include "watermark_turn.pml"

// The words, by the index member each stands for, per shard; the last is `commit_sequence_`.
#define claimed(shard) (shard)
#define last_visible_id(shard) (2 + (shard))
#define sequence_word 4

// The counted sequence's carry: opening a bracket adds one, closing it adds the unit less one.
#define sequence_unit 4

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef release_after_dispense
#define release_after_dispense false
#endif
#ifndef dispensing_mutex
#define dispensing_mutex true
#endif
#ifndef claim_consume
#define claim_consume true
#endif

byte adders_done;                   // the checker waits on it under `sc`
byte dispensed[shards];             // each shard's `next_dispensed_id_`, a ghost, since no plan reads it
bool batch_dispensed[planners];     // the planner's ranges are dispensed on every shard
byte batch_part[planners * shards]; // the ID each planner's range starts at on each shard
#define part(planner, shard) batch_part[(planner) * shards + (shard)]

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

// `plan_insert`'s claims: every shard's share in shard order, or none, a refusal releasing the
// shares granted; `claimed_shards` reaches `shards` only when the plan holds them all
inline plan_claims(t, claimed_shards) {
    int before;
    do
    :: claimed_shards == shards -> break
    :: else ->
        claim(t, claimed_shards, before);
        if
        :: before + 1 <= row_limit -> claimed_shards++
        :: else -> release_up_to(t, claimed_shards); break
        fi
    od
}

// an adder's last step, which the checker counts under `sc`
inline adder_done() {
    if
    :: memory == sc -> adders_done++
    :: else
    fi
}

proctype planner(byte t) {
    byte shard, claimed_shards;
    int watermark, sequence;
    plan_claims(t, claimed_shards);
    if
    :: claimed_shards != shards -> goto refused
    :: else
    fi;
    // insert: every shard's range dispensed in one step, consuming the claims
    if
    :: dispensing_mutex ->
        atomic {
            for (shard : 0 .. shards - 1) {
                part(t, shard) = dispensed[shard];
                dispensed[shard]++
            }
        }
    :: else ->
        for (shard : 0 .. shards - 1) {
            atomic { part(t, shard) = dispensed[shard]; dispensed[shard]++ }
        }
    fi;
    batch_dispensed[t] = true;
    // a refused fan-out lands the rows on the calling thread, since a dispensed row must still land
    if
    :: release_after_dispense ->
        if
        :: release_up_to(t, shards); goto refused
        :: skip
        fi
    :: else
    fi;
    // finish: every turn awaited, then the watermarks stored inside the bracket
    for (shard : 0 .. shards - 1) {
        await_turn(t, last_visible_id(shard), part(t, shard), watermark)
    };
    read_modify_write(t, sequence_word, order_release, sequence, sequence + 1);
    for (shard : 0 .. shards - 1) {
        end_turn(t, last_visible_id(shard), order_release, part(t, shard) + 1)
    };
    read_modify_write(t, sequence_word, order_release, sequence, sequence + sequence_unit - 1);
    // the plan dies, and an unconsumed claim releases its rows
    if
    :: !claim_consume -> release_up_to(t, shards)
    :: else
    fi;
refused:
    adder_done()
}

// a plan dropped before its first step: every share it was granted goes back as its claims die
proctype dropper(byte t) {
    byte claimed_shards;
    plan_claims(t, claimed_shards);
    if
    :: claimed_shards == shards -> release_up_to(t, shards)
    :: else
    fi;
    adder_done()
}

proctype single_adder(byte t) {
    byte shard;
    int before, id, watermark;
    // the routing base is shard zero; a full shard passes the row on to the next
    do
    :: shard == shards -> goto refused
    :: else ->
        claim(t, shard, before);
        if
        :: before + 1 <= row_limit -> break
        :: else -> shard++
        fi
    od;
    // the key column's cover, the locator seats or the traversal refused: the claim dies unconsumed
    if
    :: add_no_return(t, claimed(shard), order_release, 0 - 1); goto refused
    :: skip
    fi;
    // the shard's `insert`: the dispense consumes the claim, then the turn
    atomic { id = dispensed[shard]; dispensed[shard]++ };
    take_turn(t, last_visible_id(shard), id, id + 1, order_release, watermark);
    if
    :: !claim_consume -> add_no_return(t, claimed(shard), order_release, 0 - 1)
    :: else
    fi;
refused:
    adder_done()
}

// `snapshot_`: the word acquired and waited out while a bracket is open, every shard's `size()`
// acquired, and a retry when the word moved
proctype searcher(byte t) {
    byte shard, batch;
    int before, after, watermarks[shards];
retry:
    load(t, sequence_word, order_acquire, before);
    if
    :: before % sequence_unit != 0 -> (newest_value(sequence_word) % sequence_unit == 0); goto retry
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermarks[shard])
    };
    load(t, sequence_word, order_relaxed, after);
    if
    :: after != before -> goto retry
    :: else
    fi;
    for (batch : 0 .. planners - 1) {
        assert(!batch_dispensed[batch] ||
               (watermarks[0] > part(batch, 0)) == (watermarks[1] > part(batch, 1)))
    }
}

// once every adder is done: each shard's claim word and watermark equal its dispenser
proctype checker() {
    byte shard;
    if
    :: memory == sc ->
        (adders_done == planners + 2);
        for (shard : 0 .. shards - 1) {
            assert(newest_value(claimed(shard)) == dispensed[shard]);
            assert(newest_value(last_visible_id(shard)) == dispensed[shard])
        }
    :: else
    fi
}

init { atomic { run planner(0); run planner(1); run dropper(2); run single_adder(3); run searcher(4); run checker() } }
