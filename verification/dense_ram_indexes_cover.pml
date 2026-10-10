/**
 *  @brief Three adders on one shard, each covering its key column before it dispenses.
 *
 *  A shard of `include/usearch/dense/ram_indexes.hpp` keeps its keys in a `chunked_column` on the
 *  shard's own chunk schedule, and every claimer covers it to what the shard hosts after its claim,
 *  before dispensing. One shard whose rows and key chunks hold one row apiece; three adders claim a
 *  row each, cover the key column to the shard's capacity, dispense, store their key before the
 *  watermark turn and publish; a searcher reads every key below the watermark it acquired.
 *
 *  Every key slot an insert writes or a searcher reads lies in a chunk the key column published.
 *  The ID an adder dispenses may lie past its own claim, in a chunk only a later claimer covered,
 *  but that claimer dispensed first, and the dispenser's acquire-release carries its cover along.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc cover=false: every claimer covers the key column to what the shard hosts after
 *      its claim; without the cover, an adder writes a key into a chunk never published
 *  @verify fail rc11 dispense_order=order_relaxed: the dispenser's acquire-release carries the
 *      cover of the claimer that dispensed first; relaxed, an adder whose ID lies in a chunk only
 *      that claimer covered writes a key into a chunk it never saw published
 */
#define thread_count 4
#define location_count 9
#define history_depth 4
#include "weak_memory.pml"
#include "watermark_turn.pml"

// The words, by the index member each stands for, of the one shard.
#define last_visible_id 0
#define next_dispensed_id 1
#define key(id) (2 + (id))
#define claimed_rows 5
#define key_chunk(id) (6 + (id))

// The values a key slot carries: unset, or the key of the thread that landed the row.
#define unset 0
#define key_of(thread) (10 + (thread))

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef cover
#define cover true
#endif
#ifndef dispense_order
#define dispense_order order_acq_rel
#endif

// `chunked_column::cover`: every chunk below @p rows, published with a release unless a peer did
inline cover_column(t, rows) {
    byte covered;
    int chunk;
    bool published;
    do
    :: covered == rows -> break
    :: else ->
        load(t, key_chunk(covered), order_acquire, chunk);
        if
        :: chunk == 0 -> compare_exchange(t, key_chunk(covered), order_release, chunk, 1, published)
        :: else
        fi;
        covered++
    od
}

// `insert`'s tail: the dispense with `on_assigned`'s key in one step, the slot's chunk loaded
// relaxed as `chunked_column::operator[]` does, then the turn
inline land_row(t, id) {
    int chunk, watermark;
    atomic {
        read_modify_write(t, next_dispensed_id, dispense_order, id, id + 1);
        load(t, key_chunk(id), order_relaxed, chunk);
        assert(chunk != 0);
        store(t, key(id), order_relaxed, key_of(t))
    };
    take_turn(t, last_visible_id, id, id + 1, order_release, watermark)
}

// the single-row insert: the claim, the cover to what the shard hosts after it, then the row
proctype adder(byte t) {
    int first, hosted, id;
    read_modify_write(t, claimed_rows, order_acquire, first, first + 1);
    load(t, claimed_rows, order_acquire, hosted);
    if
    :: cover -> cover_column(t, hosted)
    :: else
    fi;
    land_row(t, id)
}

// the search merge: every ID below the watermark it acquired resolves through a published chunk
proctype searcher(byte t) {
    byte id;
    int watermark, seen;
    load(t, last_visible_id, order_acquire, watermark);
    do
    :: id == watermark -> break
    :: else ->
        load(t, key_chunk(id), order_relaxed, seen);
        assert(seen != 0);
        load(t, key(id), order_relaxed, seen);
        assert(seen != unset);
        id++
    od
}

init { atomic { run adder(0); run adder(1); run adder(2); run searcher(3) } }
