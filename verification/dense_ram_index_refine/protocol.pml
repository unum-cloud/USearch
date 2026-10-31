/**
 *  @file verification/dense_ram_index_refine/protocol.pml
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief The picks between a refinement's rounds, from `include/usearch/dense/ram_index.hpp`.
 *
 *  A planned round of `refine_plan_t` re-selects nodes across parts, and `back_link_neighbors` picks
 *  every resident it edits for the re-selected appended node into `picked_next_round_` when the two
 *  sit in different parts: `bitset_insert`, a relaxed load, and a relaxed `set_bits` only when the
 *  bit is clear. `advance_round_` empties that run before the round, the executor forks the round's
 *  steps and awaits them, and the next round's `picked_` reads the run plainly. Two residents share
 *  one word of the run, resident `n` in part `n`.
 */
#include <weak_memory.pml>

// The words: the run the round picks into, one word for both residents, and the executor's fork and
// await counter, which the caller sets to one and every back-linker's arrival bumps.
#define picked_next_round 0
#define pool_counter 1
#define residents 2
#define back_linkers 2

// The parts: back-linker `t` re-selects an appended node of part `t + 1`, so the first shares a part with
// resident 1 and the second with neither.
#define bit_of(resident) (1 << (resident))
#define across(t, resident) ((t) + 1 != (resident))

// The insert's forms: the shipped `set_bits`, or a plain read-or-write of the word.
#define set_bits 0
#define plain 1

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef bitset_insert
#define bitset_insert set_bits
#endif
#if bitset_insert != set_bits && bitset_insert != plain
#error "bitset_insert is set_bits or plain"
#endif
#ifndef join_order
#define join_order order_acq_rel
#endif
#ifndef clear_before_round
#define clear_before_round true
#endif

bool crossed[residents]; // a back-linker of another part edited the resident's row this round

// `bitset_insert` behind `back_link_neighbors` for back-linker @p t 's appended node and the edited @p resident
inline pick(t, resident, seen) {
    if
    :: across(t, resident) ->
        load(t, picked_next_round, order_relaxed, seen);
        if
        :: (seen & bit_of(resident)) == 0 && bitset_insert == set_bits ->
            read_modify_write(t, picked_next_round, order_relaxed, seen, seen | bit_of(resident))
        :: (seen & bit_of(resident)) == 0 && bitset_insert == plain ->
            store(t, picked_next_round, order_relaxed, seen | bit_of(resident))
        :: else
        fi
    :: else
    fi
}

// the row edit `back_link_neighbors` makes under the resident's lock, then its pick
inline edit_row(t, resident, seen) {
    if
    :: across(t, resident) -> crossed[resident] = true
    :: else
    fi;
    pick(t, resident, seen)
}

// `advance_round_` empties the run before the round, then the executor forks the round's steps
inline fork_round(t) {
    if
    :: clear_before_round -> store(t, picked_next_round, order_relaxed, 0)
    :: else
    fi;
    store(t, pool_counter, order_release, 1)
}

// a step starts once it acquires the fork
inline await_fork(t, seen) {
    (newest_value(pool_counter) >= 1);
    do
    :: load(t, pool_counter, order_acquire, seen);
        if
        :: seen >= 1 -> break
        :: else
        fi
    od
}

// a step's arrival at the await
inline arrive(t, seen) { read_modify_write(t, pool_counter, join_order & order_release, seen, seen + 1) }

// the caller waits out every arrival before its `advance`
inline await_join(t, seen) {
    (newest_value(pool_counter) == 1 + back_linkers);
    do
    :: load(t, pool_counter, join_order & order_acquire, seen);
        if
        :: seen == 1 + back_linkers -> break
        :: else
        fi
    od
}
