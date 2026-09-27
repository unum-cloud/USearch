/**
 *  Storage capacity in the dense index from `include/usearch/dense/ram_index.hpp`: the ceiling
 *  is gone and every index grows on demand, `create_next_chunk` adds chunks until the deficit
 *  is covered, and `-Dscenario=regrow` models `reserve` called twice, the second time above the
 *  physical rows.
 *
 *  `claim_capacity` is one unconditional subtract on `remaining_capacity_` whose sign says what
 *  happened: the thread that took it from non-negative to negative is elected to grow alone,
 *  takes its rows from `last_allocated_id_` with an `acq_rel`, and only then releases the fresh
 *  rows; a thread that stayed non-negative takes its rows at once; a thread that found a deficit
 *  releases its subtract and waits for the counter to move. `release_capacity` returns rows no id
 *  was dispensed against: a release no-return subtract on `last_allocated_id_`, then a release
 *  compare-exchange that adds to `remaining_capacity_` only once it reads non-negative.
 *
 *  Invariants:
 *  - election: one thread grows at a time;
 *  - existence: the rows a claim returns lie inside the chunks_count the claimer can see.
 *    `-Dwithout_deflation_release` makes the deflation a relaxed no-return subtract, which
 *    `-Dmemory=far` posts; it is still in flight when the release lands, and the claimer takes
 *    rows past the end of the chunks it can see;
 *  - `-Dwithout_take_release` makes the grower's take an acquire, and a lucky taker the grower
 *    pushed into the fresh chunk has no edge to it;
 *  - `-Dwithout_whole_release` lets a release lift a negative counter, and a second grower is
 *    elected into the plain chunk directory;
 *  - progress: `-Dscenario=regrow` reserves two rows, then three. The second reservation's
 *    deficit is one row and two free rows are sitting on the counter, so the deficit-sized
 *    claim the header had was granted without growing anything, released, and the same
 *    capacity re-read forever - the verified hang. `reserve` now claims
 *    `capacity - last_allocated_id_`: strictly more than the counter holds, so the subtract
 *    crosses zero, elects a grower, and the loop ends in one round. `reserve_rounds_k` bounds
 *    the rounds and the assertion names an over-run; `-Dwithout_deficit_growth` restores the
 *    deficit-sized claim and that assertion fires.
 */
#include "weak_memory.pml"

// The scenarios: the chunk growth with a release, two reservations, an abandoned load plan and
// the plan after it, or one reservation growing over a second.
#define growth 1
#define reservers 2
#define loaders 3
#define regrow 4
#ifndef scenario
#define scenario growth
#endif

// The threads, by role, apart from the processes that play them; the adders and the reservers name themselves as they start.
#define grower_thread 0
#define claimer_thread 1
#define abandoner_thread 0
#define retrier_thread 1
#define regrower_thread 0

// The words, by the index member each stands for.
#define remaining_capacity 0
#define last_allocated_id 1
#define chunks_count 2
#define geometry 3

// The scenario's sizes.
#define rows_per_chunk 2 // one chunk adds this many rows
#define requested(me) (2 - (me)) // rows each reserver asks for
// Rounds a terminating `reserve` may take: one to grow, one to read the grown capacity back.
#define reserve_rounds_k 2

// The geometries a reservation may ask for; zero is the unfrozen word.
#define first_geometry 1
#define second_geometry 2

byte growing;    // threads inside create_next_chunk
byte frozen;     // the geometry the one successful exchange wrote
byte reserved;   // reservers done
byte claims;     // claims the reservers made between them

#ifdef without_take_release
#define take_order order_acquire
#else
#define take_order order_acq_rel
#endif

#ifdef without_deflation_release
#define deflation_order order_relaxed
#else
#define deflation_order order_release
#endif

// the release: the deflation, then an add that lands only on a whole counter; the guard stands in
// for the header's acquire loads before the wait: `release_capacity`
inline release_capacity(t, count) {
    add_no_return(t, last_allocated_id, deflation_order, 0 - count);
#ifdef without_whole_release
    add_no_return(t, remaining_capacity, order_release, count)
#else
    do
    :: (newest_value(remaining_capacity) >= 0);
       read_modify_write_if(t, remaining_capacity, order_release, whole >= 0, whole, whole + count);
       if :: whole >= 0 -> break :: else fi
    od
#endif
}

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
           // a deficit someone else is filling: release the subtract and wait. The wait is
           // capped, so the retry comes at some moment the counter is whole again
           read_modify_write(t, remaining_capacity, order_release, reverted, reverted + count);
           (newest_value(remaining_capacity) >= 0)
       fi
    od
}

// capacity(): the release first, so one seen carries its deflation, then the claimed rows. The
// claimed rows are an output too - a reservation claims against them
inline capacity(t, claimed_rows, total) {
    load(t, remaining_capacity, order_acquire, unclaimed);
    load(t, last_allocated_id, order_acquire, claimed_rows);
    total = claimed_rows;
    if
    :: unclaimed > 0 -> total = total + unclaimed
    :: else
    fi
}

// `reserve`: grow until the rows are there, claiming every free row along with the deficit so the
// subtract elects a grower, and releasing the whole claim
inline reserve(t, target, rounds) {
    rounds = 0;
    do
    :: capacity(t, claimed_rows, total);
       if
       :: total >= target -> break
       :: else ->
           rounds++;
           assert(rounds <= reserve_rounds_k);
#ifdef without_deficit_growth
           deficit = target - total;
#else
           deficit = target - claimed_rows;
#endif
           claim(t, deficit, first);
           release_capacity(t, deficit)
       fi
    od
}

#if scenario == growth
byte claimers_started;

active proctype grower() {
    int old_remaining, reverted, seen_chunks, first, whole, added;
    claim(grower_thread, 2, first);
    release_capacity(grower_thread, 2);
    landed(grower_thread)
}

active proctype claimer() {
    byte me;
    int old_remaining, reverted, seen_chunks, first, whole, added;
    atomic { me = claimer_thread + claimers_started; claimers_started++ };
    claim(me, 1, first)
}
#elif scenario == loaders
byte abandoned; // the first plan died

// `plan_load`'s claim of two rows, then the plan's death: the claim it owns releases them
active proctype abandoner() {
    int old_remaining, reverted, seen_chunks, first, whole, added;
    claim(abandoner_thread, 2, first);
    assert(first == 0);
#ifndef without_abandon_release
    release_capacity(abandoner_thread, 2);
#endif
    landed(abandoner_thread);
    abandoned = 1
}

// the next `plan_load`: its rows start at the watermark again, and no second chunk grew
active proctype retrier() {
    int old_remaining, reverted, seen_chunks, first, whole, unclaimed, total, claimed_rows, added;
    (abandoned == 1);
    claim(retrier_thread, 2, first);
    assert(first == 0);
    capacity(retrier_thread, claimed_rows, total);
    assert(total == rows_per_chunk)
}
#elif scenario == regrow
// `reserve` twice over one index: two rows, then three. The second call's deficit is one row and
// two free rows are sitting on the counter - the shape that hung while `reserve` claimed the deficit
active proctype regrower() {
    int old_remaining, reverted, seen_chunks, first, whole, added;
    int unclaimed, total, claimed_rows, deficit, rounds;
    reserve(regrower_thread, 2, rounds);
    capacity(regrower_thread, claimed_rows, total);
    assert(total >= 2);
    reserve(regrower_thread, 3, rounds);
    capacity(regrower_thread, claimed_rows, total);
    assert(total >= 3);
    landed(regrower_thread)
}
#else
byte reservers_started;

active [2] proctype reserver() {
    byte me;
    int old_remaining, reverted, seen_chunks, first, whole, added;
    int unclaimed, total, claimed_rows, deficit, desired, observed, rounds;
    bool succeeded;
    atomic { me = reservers_started; reservers_started++ };
    // the geometry, frozen by whichever exchange lands first; a loser reads the frozen one back
    if
    :: desired = first_geometry
#ifndef same_geometry
    :: desired = second_geometry
#endif
    fi;
    atomic {
        compare_exchange(me, geometry, order_acq_rel, observed, desired, succeeded);
        if
        :: succeeded -> assert(frozen == 0); frozen = desired
        :: else -> assert(observed == frozen)
        fi
    };
    if
    :: succeeded || observed == desired ->
        // grow through the claimer's election and release the whole claim until the rows are
        // there. Two reservers can still send each other round, each reading the other's
        // transient dip; three claims between them fit the histories
        capacity(me, claimed_rows, total);
        do
        :: total >= requested(me) || claims == 3 -> break
        :: else ->
            claims++;
            deficit = requested(me) - claimed_rows;
            claim(me, deficit, first);
            release_capacity(me, deficit);
            capacity(me, claimed_rows, total)
        od;
        landed(me)
    :: else // config_mismatch_k: nothing claimed
    fi;
#if memory == sequential
    // once both are done, a reservation that froze or matched outlives the other's transient dips
    reserved++;
    (reserved == 2);
    if
    :: succeeded || observed == desired -> capacity(me, claimed_rows, total); assert(total >= requested(me))
    :: else
    fi
#endif
}
#endif
