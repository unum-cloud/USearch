/**
 *  @brief Two concurrent reservations, each growing through the claimer's election.
 *
 *  Reserver 0 asks for two rows, reserver 1 for one. Each picks a geometry, freezes it with the
 *  exchange that lands first or reads the frozen one back, and on a match claims its deficit from
 *  the claimed rows and releases the whole claim until the rows are there. Two reservers can still
 *  send each other round, each reading the other's transient dip, so three claims between them
 *  bound the loop. Under `sc`, once both are done, a reservation that froze or matched outlives the
 *  other's transient dips.
 *
 *  @verify pass sc,rc11,far
 *  @verify pass sc same_geometry=true
 *  @verify fail far deflation_order=order_relaxed: `release_capacity`'s subtract on
 *      `last_allocated_id_` releases; relaxed, it is posted and still in flight when the release
 *      lands, and a reserver takes rows past the end of the chunks it can see
 *  @verify fail sc waits_for_whole=false: `release_capacity` lands its add only on a whole counter,
 *      waiting out a growth the way a deficit does, so nothing but the grower's own release lifts a
 *      negative counter; without the wait, one reserver's release lifts the counter back to zero
 *      while the other still grows, its next subtract crosses zero again, and two threads are
 *      inside `create_next_chunk`, whose `chunks_count_` and directory writes are plain
 *  @verify fail sc release_floor=0: `release_capacity`'s floor is the count it adds, so the sum
 *      stays at least the count only where the counter read non-negative; a floor of zero admits a
 *      counter as low as minus the count, lifting a grower's deficit back across zero, and a
 *      second reserver is elected into `create_next_chunk`
 *  @verify fail rc11 take_order=order_acquire: the grower's take on `last_allocated_id_` releases
 *      the new chunks; acquiring only, a reserver that fit on the counter before the growth, but
 *      whose rows the take pushed into the new chunk, has no edge to the directory and reads it
 *      stale
 */
#define thread_count 2
#define location_count 4
#define history_depth 11
#include "protocol.pml"

// The rows each reserver asks for.
#define requested(t) (2 - (t))

byte frozen;   // the geometry the one successful exchange wrote
byte reserved; // reservers done
byte claims;   // claims the reservers made between them

proctype reserver(byte t) {
    int claimed_rows, total, deficit, desired, observed, first;
    int whole, old_remaining, reverted, seen_chunks, added, unclaimed;
    bool succeeded;
    // the geometry, frozen by whichever exchange lands first; a loser reads the frozen one back
    if
    :: desired = first_geometry
    :: !same_geometry -> desired = second_geometry
    fi;
    atomic {
        compare_exchange(t, geometry, order_acq_rel, observed, desired, succeeded);
        if
        :: succeeded -> assert(frozen == 0); frozen = desired
        :: else -> assert(observed == frozen)
        fi
    };
    if
    :: succeeded || observed == desired ->
        // grow through the claimer's election and release the whole claim until the rows are there
        capacity(t, claimed_rows, total);
        do
        :: total >= requested(t) || claims == 3 -> break
        :: else ->
            claims++;
            deficit = requested(t) - claimed_rows;
            claim(t, deficit, first);
            release_capacity(t, deficit);
            capacity(t, claimed_rows, total)
        od;
        landed(t)
    :: else // config_mismatch_k: nothing claimed
    fi;
    if
    :: memory == sc ->
        reserved++;
        (reserved == 2);
        if
        :: succeeded || observed == desired -> capacity(t, claimed_rows, total); assert(total >= requested(t))
        :: else
        fi
    :: else
    fi
}

init { atomic { run reserver(0); run reserver(1) } }
