/**
 *  @file verification/ram_tables_growth.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief `chunked_multimap`'s growth: two growers publish chunks a prober walks.
 *
 *  The multimap from `include/usearch/utilities/ram_tables.hpp`, whose chunks hold one node and one
 *  slot apiece here, in two columns: the control bytes and the slots. Two inserters each take the
 *  next node from an empty map, as their index's claim dispenses it, and a prober walks every chunk
 *  it sees published. `reserve` covers the control bytes of every chunk up to the node's, then the
 *  slots: a missing chunk is allocated, zeroed and published into its column's directory with a
 *  compare-exchange, and a peer that published first keeps its own.
 *
 *  - An inserter's chunk and a prober's are never read before their publisher zeroed them.
 *  - A prober that found a chunk's slots never indexes an empty chunk of control bytes.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc chunk_reads=states_first: the prober reads the control bytes' directory empty, a
 *      cover publishes both columns, and the slots it reads next send it into the empty chunk
 *  @verify fail rc11 publish_chunk_order=order_acquire: without it, an inserter reads a control byte
 *      still holding the allocation's garbage
 *  @verify fail rc11 directory_order=order_relaxed publish_chunk_order=order_release: relaxed, and
 *      with the exchange releasing only, an inserter reads a control byte still holding the garbage
 *  @verify fail rc11 walk_order=order_relaxed: without it, the prober reads a control byte still
 *      holding the garbage
 */
#define thread_count 3
#define location_count 9
#define history_depth 4
#include <weak_memory.pml>

// The words: the dispensed nodes, each chunk's directory entry in the control bytes' column and in the
// slots', zero or one past the thread whose allocation it holds, and each thread's own allocation of
// each chunk of control bytes, one byte apiece; the slots' contents are read only past a matching tag.
#define claimed_seats 0
#define states_directory(c) (1 + (c))
#define slots_directory(c) (3 + (c))
#define allocation(t, c) (5 + (t) * 2 + (c))

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

// The readers' column order: `slots_first` returns on an empty slots chunk before it loads the
// control bytes, which `cover_` published first.
#define slots_first 0
#define states_first 1
#ifndef chunk_reads
#define chunk_reads slots_first
#endif
#if chunk_reads != slots_first && chunk_reads != states_first
#error "chunk_reads is slots_first or states_first"
#endif

proctype inserter(byte t) {
    byte chunk;
    int seat, owner, expected, seen;
    bool succeeded;
    // the node the index's claim dispensed, then the control bytes of every chunk up to its own
    // covered: `chunked_multimap::reserve`, through `chunked_column::cover`
    read_modify_write(t, claimed_seats, order_relaxed, seat, seat + 1);
    do
    :: chunk > seat -> break
    :: else ->
        load(t, states_directory(chunk), directory_order, owner);
        if
        :: owner == 0 ->
            // a fresh allocation, zeroed, then offered to the directory; a peer's earlier one wins
            store(t, allocation(t, chunk), order_relaxed, empty);
            expected = 0;
            compare_exchange(t, states_directory(chunk), publish_chunk_order, expected, t + 1, succeeded)
        :: else
        fi;
        chunk++
    od;
    // then the slots of the same chunks
    chunk = 0;
    do
    :: chunk > seat -> break
    :: else ->
        load(t, slots_directory(chunk), directory_order, owner);
        if
        :: owner == 0 ->
            expected = 0;
            compare_exchange(t, slots_directory(chunk), publish_chunk_order, expected, t + 1, succeeded)
        :: else
        fi;
        chunk++
    od;
    // the node's chunk through the directory, zeroed by whoever published it: `insert`
    load(t, states_directory(seat), directory_order, owner);
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
    int owner, slots_owner, seen;
    // every published chunk, stopping at the first whose slots are missing: `find_each`
    do
    :: chunk == max_chunks -> break
    :: else ->
        if
        :: chunk_reads == slots_first ->
            load(t, slots_directory(chunk), walk_order, slots_owner);
            if
            :: slots_owner == 0 -> break
            :: else
            fi;
            load(t, states_directory(chunk), walk_order, owner)
        :: else ->
            load(t, states_directory(chunk), walk_order, owner);
            load(t, slots_directory(chunk), walk_order, slots_owner);
            if
            :: slots_owner == 0 -> break
            :: else
            fi
        fi;
        assert(owner != 0);
        load(t, allocation(owner - 1, chunk), order_acquire, seen);
        assert(seen != garbage);
        chunk++
    od
}

init { atomic { run inserter(0); run inserter(1); run prober(2) } }
