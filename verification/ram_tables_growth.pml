/**
 *  @file verification/ram_tables_growth.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief `chunked_multimap`'s growth: two claims publish chunks a prober walks.
 *
 *  The multimap from `include/usearch/utilities/ram_tables.hpp`, whose chunks hold one seat and one
 *  slot apiece here. Two inserters claim one seat each from an empty map, and a prober walks every
 *  chunk it sees published. A claim covers every chunk up to its seat's: a missing one it
 *  allocates, zeroes from the allocation's garbage and publishes into the directory with a
 *  compare-exchange, and a peer that published first keeps its own.
 *
 *  - An inserter's chunk and a prober's are never read before their publisher zeroed them.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail rc11 publish_chunk_order=order_acquire: without it, an inserter reads a control byte
 *      still holding the allocation's garbage
 *  @verify fail rc11 directory_order=order_relaxed publish_chunk_order=order_release: relaxed, and
 *      with the exchange releasing only, an inserter reads a control byte still holding the garbage
 *  @verify fail rc11 walk_order=order_relaxed: without it, the prober reads a control byte still
 *      holding the garbage
 */
#define thread_count 3
#define location_count 7
#define history_depth 4
#include <weak_memory.pml>

// The words: the claimed seats, each chunk's directory entry, zero or one past the thread whose
// allocation it holds, and each thread's own allocation of each chunk, one control byte apiece.
#define claimed_seats 0
#define directory(c) (1 + (c))
#define allocation(t, c) (3 + (t) * 2 + (c))

// The values: the chunks the directory holds, and the states, whose `empty` stays apart from the
// garbage a fresh allocation, a word's starting zero, holds.
#define max_chunks 2
#define garbage 0
#define building 1
#define published 2
#define empty 3

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef publish_chunk_order
#define publish_chunk_order order_acq_rel
#endif
#ifndef directory_order
#define directory_order order_acquire
#endif
#ifndef walk_order
#define walk_order order_acquire
#endif

proctype inserter(byte t) {
    byte chunk;
    int seat, owner, expected, seen;
    bool succeeded;
    // the seat, then every chunk up to its own covered: `chunked_multimap::claim` and `chunked_column::cover`
    read_modify_write(t, claimed_seats, order_relaxed, seat, seat + 1);
    do
    :: chunk > seat -> break
    :: else ->
        load(t, directory(chunk), directory_order, owner);
        if
        :: owner == 0 ->
            // a fresh allocation, zeroed, then offered to the directory; a peer's earlier one wins
            store(t, allocation(t, chunk), order_relaxed, empty);
            expected = 0;
            compare_exchange(t, directory(chunk), publish_chunk_order, expected, t + 1, succeeded)
        :: else
        fi;
        chunk++
    od;
    // the seat's chunk through the directory, zeroed by whoever published it: `insert`
    load(t, directory(seat), directory_order, owner);
    assert(owner != 0);
    load(t, allocation(owner - 1, seat), order_relaxed, seen);
    assert(seen != garbage);
    seen = empty;
    compare_exchange(t, allocation(owner - 1, seat), order_acquire, seen, building, succeeded);
    assert(succeeded);
    store(t, allocation(owner - 1, seat), order_release, published)
}

proctype prober(byte t) {
    byte chunk;
    int owner, seen;
    // every published chunk, stopping at the first missing one: `find_each`
    do
    :: chunk == max_chunks -> break
    :: else ->
        load(t, directory(chunk), walk_order, owner);
        if
        :: owner == 0 -> break
        :: else
        fi;
        load(t, allocation(owner - 1, chunk), order_acquire, seen);
        assert(seen != garbage);
        chunk++
    od
}

init { atomic { run inserter(0); run inserter(1); run prober(2) } }
