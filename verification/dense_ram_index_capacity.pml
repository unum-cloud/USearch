/**
 *  Storage capacity in the dense index from `include/usearch/dense/ram_index.hpp`, and the
 *  admission ceilings of the sharded and basic indexes.
 *
 *  `claim_capacity` is one unconditional subtract on `remaining_capacity_` whose sign says what
 *  happened: the thread that took it from non-negative to negative is elected to grow a chunk
 *  alone, takes its rows from `last_allocated_id_` with an `acq_rel`, and only then credits
 *  the fresh rows with a release; a thread that stayed non-negative takes its rows at once; a
 *  thread that found a deficit hands its subtract back and waits for the counter to move.
 *  `release_capacity` returns rows no id was dispensed against: a release no-return subtract
 *  on `last_allocated_id_`, then a release compare-exchange that credits `remaining_capacity_`
 *  only once it reads non-negative, waiting out a grower with a capped wait; the header calls
 *  the order load-bearing.
 *
 *  Claimer 0 takes two rows, which grows the first chunk, and hands them back. Claimer 1 takes
 *  one row. Chunks hold two rows.
 *
 *  Invariants:
 *  - election: one thread grows at a time;
 *  - existence: the rows a claim returns lie inside the chunks_count the claimer can see. The
 *    subtract releases for this: with `-Dwithout_deflation_release` it is a relaxed no-return subtract,
 *    which the C++ model still keeps in the modification order, but which `-Dmemory=far` posts.
 *    Then it is still in flight when the credit lands, claimer 1 sees the credit, takes rows
 *    past the end of the only chunk, and the assertion names it.
 *
 *  `-Dscenario=admission` swaps the scenario for three adders against one shard's plane of two through
 *  `fetch_add_if_at_most`, one of them handing a refused batch back with a release no-return
 *  subtract - the sharded index's per-shard admission; the basic index's claim has the same
 *  shape at `acq_rel`, with no hand-back.
 *
 *  `-Dscenario=reservers` is `reserve` twice over: each reserver freezes the geometry with one
 *  compare-exchange, and a loser whose geometry differs returns a mismatch without claiming;
 *  then each reads `capacity()` - the credit first, then the claimed rows - and while short
 *  claims the deficit and hands it back, the other reserver being its concurrent claimer.
 *  Reserver 0 asks for two rows, reserver 1 for one; the pair gets three claims. The geometry
 *  freezes exactly once, a failed exchange reads the frozen one, and a reserver that froze or
 *  matched sees its request in `capacity()` once both are done. `-Dsame_geometry` makes both
 *  ask for the first geometry, so both reserve. Two things this scenario found:
 *  - a hand-back lifts the counter while a grower is inside, and the next subtract crossing
 *    zero is elected too. A hand-back now credits only a whole counter, waiting out a growth,
 *    so nothing but the grower's own credit lifts a negative one; `-Dwithout_whole_credit`
 *    puts two threads in the chunk directory, sequentially already;
 *  - a lucky taker whose subtract left the counter whole can still be pushed past the credited
 *    rows by a grower that takes first, so its rows lie in the chunk that grower just made,
 *    and its only edge to that chunk is the take it read on `last_allocated_id_`. The grower's
 *    take releases for this; `-Dwithout_take_release` makes it an acquire, and under the view
 *    model the taker's rows lie past every chunk it can see.
 *
 *  `-Dscenario=loaders` is a load plan that dies before `finish`: `plan_load` claims two rows,
 *  the plan owns the claim, and its death releases them; the next plan then claims the same
 *  rows at the watermark and `capacity()` names one chunk. `-Dwithout_abandon_release` leaves
 *  the rows claimed, the next plan lands past the watermark, and every later one is refused.
 */
#include "weak_memory.pml"

// The scenarios: the chunk growth with a hand-back, the admission ceiling, two reservations, or
// an abandoned load plan and the plan after it.
#define growth 1
#define admission 2
#define reservers 3
#define loaders 4
#ifndef scenario
#define scenario growth
#endif

// The threads, by role, apart from the processes that play them; the adders and the reservers name themselves as they start.
#define grower_thread 0
#define claimer_thread 1
#define abandoner_thread 0
#define retrier_thread 1

// The words, by the index member each stands for.
#define remaining_capacity 0
#define last_allocated_id 1
#define chunks_count 2
#define admitted 3
#define geometry 4

// The scenario's sizes.
#define rows_per_chunk 2 // one growth adds this many rows
#define plane 2          // one shard's key plane, the admission plane
#define requested(me) (2 - (me)) // rows each reserver asks for

// The geometries a reservation may ask for; zero is the unfrozen word.
#define first_geometry 1
#define second_geometry 2

byte growing;    // threads inside create_next_chunk
byte admissions; // rows the plane let through
byte frozen;     // the geometry the one successful exchange wrote
byte reserved;   // reservers done
byte claims;     // claims the reservers made between them

#ifdef without_take_release
#define take_order order_acquire
#else
#define take_order order_acq_rel
#endif

inline claim(t, count, first) {
    do
    :: read_modify_write(t, remaining_capacity, order_acquire, old_remaining, old_remaining - count);
       if
       :: old_remaining >= 0 && old_remaining - count < 0 ->
           // elected: nothing but this thread's own credit lifts the counter while it is inside
           growing++;
           assert(growing == 1);
           load(t, chunks_count, order_relaxed, seen_chunks);
           store(t, chunks_count, order_relaxed, seen_chunks + 1);
           growing--;
           // the grower's take releases the new chunk to a lucky taker its take pushed past the
           // credited rows; the counter's own release sequence carries it
           read_modify_write(t, last_allocated_id, take_order, first, first + count);
           assert(first + count <= (seen_chunks + 1) * rows_per_chunk);
           add_no_return(t, remaining_capacity, order_release, rows_per_chunk);
           break
       :: old_remaining - count >= 0 ->
           read_modify_write(t, last_allocated_id, order_acquire, first, first + count);
           load(t, chunks_count, order_relaxed, seen_chunks);
           assert(first + count <= seen_chunks * rows_per_chunk);
           break
       :: else ->
           // a deficit someone else is filling: hand the subtract back and wait. The wait is
           // capped, so the retry comes at some moment the counter is whole again
           read_modify_write(t, remaining_capacity, order_release, reverted, reverted + count);
           (newest_value(remaining_capacity) >= 0)
       fi
    od
}

#ifdef without_deflation_release
#define deflation_order order_relaxed
#else
#define deflation_order order_release
#endif

// the hand-back: the deflation, then a credit that lands only on a whole counter; the guard stands in
// for the header's acquire loads before the wait: ram_index.hpp:834-846
inline release_capacity(t, count) {
    add_no_return(t, last_allocated_id, deflation_order, 0 - count);
#ifdef without_whole_credit
    add_no_return(t, remaining_capacity, order_release, count)
#else
    do
    :: (newest_value(remaining_capacity) >= 0);
       read_modify_write_if(t, remaining_capacity, order_release, credited >= 0, credited, credited + count);
       if :: credited >= 0 -> break :: else fi
    od
#endif
}

// capacity(): the credit first, so one seen carries its deflation, then the claimed rows
inline capacity(t, total) {
    load(t, remaining_capacity, order_acquire, unclaimed);
    load(t, last_allocated_id, order_acquire, total);
    if
    :: unclaimed > 0 -> total = total + unclaimed
    :: else
    fi
}

#if scenario == growth
active proctype grower() {
    int old_remaining, reverted, seen_chunks, first, credited;
    claim(grower_thread, 2, first);
    release_capacity(grower_thread, 2);
    landed(grower_thread)
}

active proctype claimer() {
    int old_remaining, reverted, seen_chunks, first, credited;
    claim(claimer_thread, 1, first)
}
#elif scenario == admission
byte adders_started;

active [3] proctype adder() {
    byte me;
    int observed;
    atomic { me = adders_started; adders_started++ };
    atomic {
        read_modify_write_if(me, admitted, order_relaxed, observed + 1 <= plane, observed, observed + 1);
        if
        :: observed + 1 <= plane -> admissions++
        :: else
        fi;
        assert(admissions <= plane)
    };
    if
    :: me == 2 && observed + 1 <= plane ->
        // the fan-out refused: the admission goes back whole
        atomic { add_no_return(me, admitted, order_release, 0 - 1); admissions-- }
    :: else
    fi;
    landed(me)
}
#elif scenario == loaders
byte abandoned; // the first plan died

// `plan_load`'s claim of two rows, then the plan's death: the claim it owns releases them
active proctype abandoner() {
    int old_remaining, reverted, seen_chunks, first, credited;
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
    int old_remaining, reverted, seen_chunks, first, credited, unclaimed, total;
    (abandoned == 1);
    claim(retrier_thread, 2, first);
    assert(first == 0);
    capacity(retrier_thread, total);
    assert(total == rows_per_chunk)
}
#else
byte reservers_started;

active [2] proctype reserver() {
    byte me;
    int old_remaining, reverted, seen_chunks, first, credited, unclaimed, total, deficit, desired, observed;
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
        // grow through the claimer's election and hand the whole claim back until the rows are
        // there. The header loops without a bound, and two reservers can send each other round
        // forever, each reading the other's dip; three claims between them fit the histories
        capacity(me, total);
        do
        :: total >= requested(me) || claims == 3 -> break
        :: else ->
            claims++;
            deficit = requested(me) - total;
            claim(me, deficit, first);
            release_capacity(me, deficit);
            capacity(me, total)
        od;
        landed(me)
    :: else // config_mismatch_k: nothing claimed
    fi;
#if memory == sequential
    // once both are done, a reservation that froze or matched outlives the other's transient dips
    reserved++;
    (reserved == 2);
    if
    :: succeeded || observed == desired -> capacity(me, total); assert(total >= requested(me))
    :: else
    fi
#endif
}
#endif
