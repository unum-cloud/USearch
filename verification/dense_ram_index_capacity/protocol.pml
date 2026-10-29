/**
 *  @file verification/dense_ram_index_capacity/protocol.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Storage capacity in the dense index: every index grows on demand, and `create_next_chunk`
 *      adds chunks until the deficit is covered.
 *
 *  From `include/usearch/dense/ram_index.hpp`. `claim_capacity` is one unconditional subtract on
 *  `remaining_capacity_` whose sign says what happened: the thread that took it from non-negative
 *  to negative is elected to grow alone. `release_capacity` returns rows no ID was dispensed
 *  against.
 *
 *  - The election, one thread growing at a time.
 *  - Existence, the rows a claim returns lying inside the chunks the claimer can see, in every column:
 *    the rows, the key column, and the map's control bytes and slots at seat = node.
 */
#include <weak_memory.pml>

// The words, by the index member each stands for; `keyed_` covers three columns of its own, its key
// column and its map's control bytes and slots, each its own directory; the frozen geometry comes last.
#define remaining_capacity 0
#define last_allocated_id 1
#define chunks_count 2
#define keyed_chunks(column) (3 + (column))
#define keyed_columns 3
#define geometry 6

// The values: the rows one chunk adds, and the geometries a reservation may ask for; zero is the
// unfrozen word.
#define rows_per_chunk 2
#define first_geometry 1
#define second_geometry 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef take_order
#define take_order order_acq_rel
#endif
#ifndef deflation_order
#define deflation_order order_release
#endif
#ifndef waits_for_whole
#define waits_for_whole true
#endif
#ifndef same_geometry
#define same_geometry false
#endif
#ifndef release_floor
#define release_floor count
#endif
#ifndef keyed_cover
#define keyed_cover true
#endif

byte growing; // threads inside `create_next_chunk`

// The inlines below name their callers' locals: `whole`, `old_remaining`, `reverted`,
// `seen_chunks`, `added`, `column` and `unclaimed`, since a local an inline declares costs a step
// every time the inline runs.

// the rows from `first` to `first + count` inside the chunks of every column the claimer sees
inline check_covered(t, first, count) {
    load(t, chunks_count, order_relaxed, seen_chunks);
    assert(first + count <= seen_chunks * rows_per_chunk);
    column = 0;
    do
    :: column == keyed_columns -> break
    :: else ->
        load(t, keyed_chunks(column), order_relaxed, seen_chunks);
        assert(first + count <= seen_chunks * rows_per_chunk);
        column++
    od
}

// `release_capacity`: the deflation, then `fetch_add_if_at_least(count, release_floor)`, which adds
// only where `whole + count >= release_floor`; the guard stands in for the header's acquire loads
// and its wait while the counter reads negative
inline release_capacity(t, count) {
    add_no_return(t, last_allocated_id, deflation_order, 0 - count);
    if
    :: waits_for_whole ->
        do
        :: (newest_value(remaining_capacity) >= 0);
            read_modify_write_if(t, remaining_capacity, order_release, whole + count >= release_floor,
                                 whole, whole + count);
            if :: whole + count >= release_floor -> break :: else fi
        od
    :: else -> add_no_return(t, remaining_capacity, order_release, count)
    fi
}

// `claim_capacity`: the subtract, then the election, the fit, or the deficit handed back and waited out
inline claim(t, count, first) {
    do
    :: read_modify_write(t, remaining_capacity, order_acquire, old_remaining, old_remaining - count);
        if
        :: old_remaining >= 0 && old_remaining - count < 0 ->
            // elected: nothing but this thread's own release lifts the counter while it is inside.
            // `create_next_chunk` adds chunks until the whole deficit is covered, so one election
            // always answers the rows it was elected for
            growing++;
            assert(growing == 1);
            load(t, chunks_count, order_relaxed, seen_chunks);
            added = 0;
            do
            :: added >= count -> break
            :: else -> added = added + rows_per_chunk; seen_chunks = seen_chunks + 1
            od;
            store(t, chunks_count, order_relaxed, seen_chunks);
            // `keyed_.cover`: the key column, then the map's control bytes and slots
            if
            :: keyed_cover ->
                column = 0;
                do
                :: column == keyed_columns -> break
                :: else -> store(t, keyed_chunks(column), order_relaxed, seen_chunks); column++
                od
            :: else
            fi;
            growing--;
            // the grower's take releases the new chunks to a lucky taker its take pushed past the
            // released rows; the counter's own release sequence carries it
            read_modify_write(t, last_allocated_id, take_order, first, first + count);
            check_covered(t, first, count);
            add_no_return(t, remaining_capacity, order_release, added);
            break
        :: old_remaining - count >= 0 ->
            read_modify_write(t, last_allocated_id, order_acquire, first, first + count);
            check_covered(t, first, count);
            break
        :: else ->
            // a deficit someone else is filling: release the subtract and wait. The wait is capped,
            // so the retry comes at some moment the counter is whole again
            read_modify_write(t, remaining_capacity, order_release, reverted, reverted + count);
            (newest_value(remaining_capacity) >= 0)
        fi
    od
}

// `capacity()`: the release first, so one seen carries its deflation, then the claimed rows, an
// output too, since a reservation claims against them
inline capacity(t, claimed_rows, total) {
    load(t, remaining_capacity, order_acquire, unclaimed);
    load(t, last_allocated_id, order_acquire, claimed_rows);
    total = claimed_rows;
    if
    :: unclaimed > 0 -> total = total + unclaimed
    :: else
    fi
}
