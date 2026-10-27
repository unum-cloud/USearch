/**
 *  Storage capacity in the dense index from `include/usearch/dense/ram_index.hpp`: the ceiling is
 *  gone, every index grows on demand, and `create_next_chunk` adds chunks until the deficit is
 *  covered.
 *
 *  `claim_capacity` is one unconditional subtract on `remaining_capacity_` whose sign says what
 *  happened: the thread that took it from non-negative to negative is elected to grow alone, takes
 *  its rows from `last_allocated_id_` with an `acq_rel`, and only then releases the fresh rows; a
 *  thread that stayed non-negative takes its rows at once; a thread that found a deficit releases
 *  its subtract and waits for the counter to move. `release_capacity` returns rows no ID was
 *  dispensed against: a release no-return subtract on `last_allocated_id_`, a wait while
 *  `remaining_capacity_` reads negative, then a release `fetch_add_if_at_least(count, count)`, which
 *  adds only where the sum stays at least the count, so only onto a non-negative counter. Every
 *  scenario beside this file checks the election, one thread growing at a time, and existence,
 *  the rows a claim returns lying inside the chunks the claimer can see.
 */
#include "../weak_memory.pml"

// The words, by the index member each stands for; the frozen geometry comes last.
#define remaining_capacity 0
#define last_allocated_id 1
#define chunks_count 2
#define geometry 3

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

byte growing; // threads inside `create_next_chunk`

// The inlines below name their callers' locals: `whole`, `old_remaining`, `reverted`,
// `seen_chunks`, `added` and `unclaimed`, since a local an inline declares costs a step every time
// the inline runs.

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
            growing--;
            // the grower's take releases the new chunks to a lucky taker its take pushed past the
            // released rows; the counter's own release sequence carries it
            read_modify_write(t, last_allocated_id, take_order, first, first + count);
            assert(first + count <= seen_chunks * rows_per_chunk);
            add_no_return(t, remaining_capacity, order_release, added);
            break
        :: old_remaining - count >= 0 ->
            read_modify_write(t, last_allocated_id, order_acquire, first, first + count);
            load(t, chunks_count, order_relaxed, seen_chunks);
            assert(first + count <= seen_chunks * rows_per_chunk);
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
