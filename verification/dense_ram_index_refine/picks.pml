/**
 *  @file verification/dense_ram_index_refine/picks.pml
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief One planned round's back-links picking the next round's nodes, then the await and the read.
 *
 *  The run starts holding what the previous round read. The caller runs `advance` and forks the
 *  round; two back-linkers each edit any of the two residents' rows and pick them; the caller awaits
 *  them and reads the run as the next round's `picked_` does.
 *
 *  - After the await, the run holds exactly the residents a back-linker of another part edited.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 bitset_insert=plain: two back-linkers picking the two residents of one word
 *      each read it clear, and the second write drops the first's bit
 *  @verify fail rc11 join_order=order_relaxed: the caller reads both arrivals without their picks
 *      and the next round skips an edited resident
 *  @verify fail sc,rc11 clear_before_round=false: the run keeps the previous round's picks, and the
 *      next round re-selects residents nobody edited
 */
#define thread_count 3
#define location_count 2
#define history_depth 5
#include <dense_ram_index_refine/protocol.pml>

proctype back_linker(byte t) {
    int seen;
    await_fork(t, seen);
    if
    :: edit_row(t, 0, seen)
    :: skip
    fi;
    if
    :: edit_row(t, 1, seen)
    :: skip
    fi;
    arrive(t, seen)
}

proctype caller(byte t) {
    int seen;
    byte resident;
    fork_round(t);
    await_join(t, seen);
    load(t, picked_next_round, order_relaxed, seen);
    for (resident : 0 .. residents - 1) {
        assert(((seen & bit_of(resident)) != 0) == crossed[resident])
    }
}

// the run as the previous round left it, both residents picked, and the counter at rest
init {
    atomic {
        seed(picked_next_round, bit_of(0) | bit_of(1));
        run back_linker(0); run back_linker(1); run caller(2)
    }
}
