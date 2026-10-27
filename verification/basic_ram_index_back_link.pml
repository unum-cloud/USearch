/**
 *  @file verification/basic_ram_index_back_link.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The basic index's back-link re-selection on one full row, against `link` and `unlink`.
 *
 *  `back_link_` from `include/usearch/basic/ram_index.hpp`, on a row of two edges, to nodes `a` and
 *  `b`, at a budget of two. Two inserters back-link their newcomers `N1` and `N2` into it, each
 *  before its batch's turn: a row with room takes an append, and a full one is copied, unlocked,
 *  re-selected off-lock from the copy and the newcomer, relocked, revalidated slot by slot against
 *  the copy, and rewritten only when unchanged. A linker runs `link(row, N1)` once the watermark
 *  covers `N1`; an unlinker runs `unlink(row, a)`, moving the last slot into `a`'s.
 *
 *  - At every unlock: the row holds no edge twice.
 *  - No edge `unlink` removed comes back.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc revalidate=false: unchecked, a re-selection copied before `unlink(row, a)` writes
 *      `a` back
 *  @verify fail sc compare_slots=false: size alone misses an `unlink` followed by an append, which
 *      restore the size with other edges, and the stale re-selection writes `a` back
 *  @verify fail sc back_link_before_turn=false: run after the turn, it races a `link` naming the
 *      newcomer, and the row holds `N1` twice
 */
#define thread_count 4
#define location_count 6
#define history_depth 25
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for: the row's mutex flag, its size and three slots,
// one past the budget for `link`'s growth, and the watermark.
#define row_flag 0
#define row_size 1
#define row_slot(position) (2 + (position))
#define last_visible_id 5

// The values: a slot holds an edge, and the IDs are `a`, `b`, then each back-linker's newcomer.
#define hole 0
#define linked(id) (1 + (id))
#define node_a 0
#define node_b 1
#define newcomer(t) (2 + (t))
#define connectivity 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef revalidate
#define revalidate true
#endif
#ifndef compare_slots
#define compare_slots true
#endif
#ifndef back_link_before_turn
#define back_link_before_turn true
#endif

bool a_gone; // `unlink` removed the edge to `a`

// whether the row, as its holder sees it, holds the edge to @p id at @p position
#define holds(position, id) (newest_value(row_size) > (position) && newest_value(row_slot(position)) == linked(id))
#define holds_anywhere(id) (holds(0, id) || holds(1, id) || holds(2, id))

// the row mutex: the peek, then the claiming exchange, one step so the exchange never reads the flag back set
inline lock_row(t, displaced) {
    atomic {
        newest_value(row_flag) == hole;
        read_modify_write(t, row_flag, order_acquire, displaced, 1)
    }
}

// the release store, after what every holder leaves behind: no edge twice, and `a` gone for good
inline unlock_row(t) {
    atomic {
        assert(newest_value(row_size) < 2 || newest_value(row_slot(0)) != newest_value(row_slot(1)));
        assert(newest_value(row_size) < 3 || (newest_value(row_slot(2)) != newest_value(row_slot(0)) &&
                                              newest_value(row_slot(2)) != newest_value(row_slot(1))));
        assert(!a_gone || !holds_anywhere(node_a))
    };
    store(t, row_flag, order_release, hole)
}

// `link`'s scan of the row for @p id, under the row mutex
inline scan(t, size, id, found, seen) {
    found = false;
    if :: size > 0 -> load(t, row_slot(0), order_relaxed, seen); found = seen == linked(id) :: else fi;
    if :: size > 1 && !found -> load(t, row_slot(1), order_relaxed, seen); found = seen == linked(id) :: else fi;
    if :: size > 2 && !found -> load(t, row_slot(2), order_relaxed, seen); found = seen == linked(id) :: else fi
}

// the append both `link` and the back-link's room path take
inline append_edge(t, size, id) {
    store(t, row_slot(size), order_relaxed, linked(id));
    store(t, row_size, order_relaxed, size + 1)
}

// the newcomer's batch takes its turn, in dispensed order
inline publish(t, watermark) {
    take_turn(t, last_visible_id, newcomer(t), newcomer(t) + 1, order_release, watermark)
}

// the pool's member at @p index: the copied slots, then the newcomer
#define pooled(index) (index == degree -> linked(newcomer(t)) : (index == 0 -> first : (index == 1 -> second : third)))

// `back_link_` of this inserter's newcomer into the row, before or after its batch's turn
proctype back_linker(byte t) {
    byte nearest, other;
    int size, degree, first, second, third, kept_first, kept_second, seen, displaced;
    bool unchanged;
    if
    :: !back_link_before_turn -> publish(t, seen)
    :: else
    fi;
retry:
    lock_row(t, displaced);
    load(t, row_size, order_relaxed, size);
    if
    :: size < connectivity -> append_edge(t, size, newcomer(t)); unlock_row(t); goto turn
    :: else
    fi;
    degree = size;
    load(t, row_slot(0), order_relaxed, first);
    load(t, row_slot(1), order_relaxed, second);
    if :: degree > 2 -> load(t, row_slot(2), order_relaxed, third) :: else fi;
    unlock_row(t);
    // off-lock: the selector keeps the nearest of the copy and the newcomer, and at most one more,
    // where picking the same member twice keeps one
    nearest = 0;
    other = 0;
    do
    :: nearest < degree -> nearest++
    :: break
    od;
    do
    :: other < degree -> other++
    :: break
    od;
    kept_first = pooled(nearest);
    kept_second = pooled(other);
    lock_row(t, displaced);
    unchanged = true;
    if
    :: revalidate ->
        load(t, row_size, order_relaxed, size);
        unchanged = size == degree;
        if
        :: compare_slots && unchanged ->
            load(t, row_slot(0), order_relaxed, seen);
            unchanged = seen == first;
            if :: unchanged -> load(t, row_slot(1), order_relaxed, seen); unchanged = seen == second :: else fi;
            if :: unchanged && degree > 2 -> load(t, row_slot(2), order_relaxed, seen); unchanged = seen == third :: else fi
        :: else
        fi
    :: else
    fi;
    if
    :: unchanged ->
        store(t, row_slot(0), order_relaxed, kept_first);
        if
        :: nearest != other -> store(t, row_slot(1), order_relaxed, kept_second); store(t, row_size, order_relaxed, 2)
        :: else -> store(t, row_size, order_relaxed, 1)
        fi
    :: else
    fi;
    unlock_row(t);
    if
    :: !unchanged -> goto retry
    :: else
    fi;
turn:
    if
    :: back_link_before_turn -> publish(t, seen)
    :: else
    fi
}

// `link(row, N1)`: refused until the watermark covers the first newcomer, idempotent after
proctype linker(byte t) {
    int visible, size, seen, displaced;
    bool found;
    load(t, last_visible_id, order_acquire, visible);
    if
    :: newcomer(0) < visible ->
        lock_row(t, displaced);
        load(t, row_size, order_relaxed, size);
        scan(t, size, newcomer(0), found, seen);
        if
        :: !found -> append_edge(t, size, newcomer(0))
        :: else
        fi;
        unlock_row(t)
    :: else
    fi
}

// `unlink(row, a)`: the last slot moves into `a`'s
proctype unlinker(byte t) {
    byte position;
    int size, seen, last, displaced;
    lock_row(t, displaced);
    load(t, row_size, order_relaxed, size);
    do
    :: position == size -> break
    :: else ->
        load(t, row_slot(position), order_relaxed, seen);
        if
        :: seen == linked(node_a) ->
            load(t, row_slot(size - 1), order_relaxed, last);
            store(t, row_slot(position), order_relaxed, last);
            atomic { store(t, row_size, order_relaxed, size - 1); a_gone = true };
            break
        :: else -> position++
        fi
    od;
    unlock_row(t)
}

// the row starts full with the edges to `a` and `b`, both published, and neither newcomer is
init {
    atomic {
        seed(row_size, 2);
        seed(row_slot(0), linked(node_a));
        seed(row_slot(1), linked(node_b));
        seed(last_visible_id, newcomer(0));
        run back_linker(0); run back_linker(1); run linker(2); run unlinker(3)
    }
}
