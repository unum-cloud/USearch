/**
 *  @file verification/basic_ram_index_back_link.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The basic index's back-link re-selection on one full row, against `link` and `unlink`.
 *
 *  `back_link_` from `include/usearch/basic/ram_index.hpp`, on a row of two edges, to nodes `a` and
 *  `b`, at a budget of two. Two inserters back-link their appended nodes `N1` and `N2` into it, each
 *  before its batch's turn: a row with room takes an append, and a full one is copied, unlocked,
 *  re-selected off-lock from the copy and the appended node, relocked, revalidated slot by slot against
 *  the copy, and rewritten only when unchanged. A linker runs `link(row, N1)` once the watermark
 *  covers `N1`; an unlinker runs `unlink(row, a)`, moving the last slot into `a`'s. Every holder
 *  claims the row's `held_k` bit in its state word, beside `wired_k`, as `node_t::lock` does. The
 *  back-linkers hand the row to a refinement round's marker: `N1` is of another part than the row,
 *  `N2` of its own.
 *
 *  - At every unlock: the row holds no edge twice.
 *  - No edge `unlink` removed comes back.
 *  - Every lock finds the row's `wired_k` bit still set.
 *  - A back-linker marks the row exactly when it put an appended node of another part there, on any try.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc revalidate=false: unchecked, a re-selection copied before `unlink(row, a)` writes
 *      `a` back
 *  @verify fail sc compare_slots=false: size alone misses an `unlink` followed by an append, which
 *      restore the size with other edges, and the stale re-selection writes `a` back
 *  @verify fail sc back_link_before_turn=false: run after the turn, it races a `link` naming the
 *      appended node, and the row holds `N1` twice
 *  @verify fail sc mark_reselected=false: marking only the append, a re-selection that keeps `N1`
 *      leaves the row unmarked
 */
#define thread_count 4
#define location_count 6
#define history_depth 25
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for: the row's state, its size and three slots, one
// past the budget for `link`'s growth, and the watermark.
#define row_state 0
#define row_size 1
#define row_slot(position) (2 + (position))
#define last_visible_id 5

// The values: a slot holds an edge, and the IDs are `a`, `b`, then each back-linker's appended node.
#define linked(id) (1 + (id))
#define node_a 0
#define node_b 1
#define appended_node(t) (2 + (t))
#define connectivity 2
#define across(t) ((t) == 0)

// The bits of the row's state: `wired_k`, set before the row is reachable, and `held_k`.
#define wired_bit 1
#define held_bit 2

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
#ifndef mark_reselected
#define mark_reselected true
#endif

bool a_gone; // `unlink` removed the edge to `a`
bool crossed[2]; // the back-linker stored its appended node of another part into the row
bool marked[2]; // the back-linker's marker took the row

// the marker after the unlock: only an appended node of another part marks
inline mark_row(t) {
    if
    :: across(t) -> marked[t] = true
    :: else
    fi
}

// whether the row, as its holder sees it, holds the edge to @p id at @p position
#define holds(position, id) (newest_value(row_size) > (position) && newest_value(row_slot(position)) == linked(id))
#define holds_anywhere(id) (holds(0, id) || holds(1, id) || holds(2, id))

// `node_t::lock`: the relaxed peek until `held_k` is clear, then `fetch_test_and_set_bit` with an
// acquire, one step so the claim never finds the bit set and loops
inline lock_row(t, displaced) {
    atomic {
        (newest_value(row_state) & held_bit) == 0;
        read_modify_write(t, row_state, order_acquire, displaced, displaced | held_bit);
        assert((displaced & held_bit) == 0 && (displaced & wired_bit) != 0)
    }
}

// `node_t::unlock`'s `clear_bits` with a release, after what every holder leaves behind: no edge
// twice, and `a` gone for good
inline unlock_row(t, displaced) {
    atomic {
        assert(newest_value(row_size) < 2 || newest_value(row_slot(0)) != newest_value(row_slot(1)));
        assert(newest_value(row_size) < 3 || (newest_value(row_slot(2)) != newest_value(row_slot(0)) &&
                                              newest_value(row_slot(2)) != newest_value(row_slot(1))));
        assert(!a_gone || !holds_anywhere(node_a))
    };
    read_modify_write(t, row_state, order_release, displaced, displaced & ~held_bit)
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

// the appended node's batch takes its turn, in dispensed order
inline publish(t, watermark) {
    take_turn(t, last_visible_id, appended_node(t), appended_node(t) + 1, order_release, watermark)
}

// the pool's member at @p index: the copied slots, then the appended node
#define pooled(index) (index == degree -> linked(appended_node(t)) : (index == 0 -> first : (index == 1 -> second : third)))

// `back_link_` of this inserter's appended node into the row, before or after its batch's turn
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
    :: size < connectivity ->
        append_edge(t, size, appended_node(t));
        crossed[t] = across(t);
        unlock_row(t, displaced);
        mark_row(t);
        goto turn
    :: else
    fi;
    degree = size;
    load(t, row_slot(0), order_relaxed, first);
    load(t, row_slot(1), order_relaxed, second);
    if :: degree > 2 -> load(t, row_slot(2), order_relaxed, third) :: else fi;
    unlock_row(t, displaced);
    // off-lock: the selector keeps the nearest of the copy and the appended node, and at most one more,
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
        fi;
        crossed[t] = across(t) && (kept_first == linked(appended_node(t)) ||
                                   nearest != other && kept_second == linked(appended_node(t)))
    :: else
    fi;
    unlock_row(t, displaced);
    if
    :: unchanged && mark_reselected && (nearest == degree || other == degree) -> mark_row(t)
    :: else
    fi;
    if
    :: !unchanged -> goto retry
    :: else
    fi;
turn:
    assert(marked[t] == crossed[t]);
    if
    :: back_link_before_turn -> publish(t, seen)
    :: else
    fi
}

// `link(row, N1)`: refused until the watermark covers the first appended node, idempotent after
proctype linker(byte t) {
    int visible, size, seen, displaced;
    bool found;
    load(t, last_visible_id, order_acquire, visible);
    if
    :: appended_node(0) < visible ->
        lock_row(t, displaced);
        load(t, row_size, order_relaxed, size);
        scan(t, size, appended_node(0), found, seen);
        if
        :: !found -> append_edge(t, size, appended_node(0))
        :: else
        fi;
        unlock_row(t, displaced)
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
    unlock_row(t, displaced)
}

// the row starts full with the edges to `a` and `b`, both published, and neither appended node is
init {
    atomic {
        seed(row_state, wired_bit);
        seed(row_size, 2);
        seed(row_slot(0), linked(node_a));
        seed(row_slot(1), linked(node_b));
        seed(last_visible_id, appended_node(0));
        run back_linker(0); run back_linker(1); run linker(2); run unlinker(3)
    }
}
