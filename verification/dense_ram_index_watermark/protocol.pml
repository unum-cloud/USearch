/**
 *  The dense index's publication protocol from `include/usearch/dense/ram_index.hpp`: identity
 *  dispensed from `next_dispensed_id_`, rows written and unlocked below it, the hole-free
 *  `last_visible_id_` watermark advanced in dispensed order, and the readers that rest on the
 *  watermark: a traversal, a cursor, and a concurrent save with its settlement probe, beside a
 *  refinement that relocks published rows. The dispenser closes for a load or a refinement: one
 *  compare-exchange from the watermark to `node_id_unknown_k`, refused while a batch sits between
 *  its dispense and its turn, after which every dispense refuses until the closer reopens it.
 *
 *  A row is slot zero, which locks and publishes it, and one tail slot. Every scenario beside this
 *  file defines its shape, `named_rows` for the key column `on_assigned` writes, `appended_rows`
 *  for rows that arrive only through the load path's `finish`, and `refined_rows` for row 0's
 *  second tail slot, which a refinement rewrites beside the first.
 */
#include "../weak_memory.pml"
#include "../watermark_turn.pml"

// The words, by the index member each stands for; the key column, the load plan's tally and row 0's
// second tail slot come last.
#define next_dispensed_id 0
#define last_visible_id 1
#define last_allocated_id 2
#define row_lock(id) (3 + (id) * 2)
#define row_tail(id) (4 + (id) * 2)
#define key(id) (7 + (id))
#define covered 9
#define second_tail 10
#define rows 2

// The values rows and keys carry: the sentinel is the all-ones ID in silicon, and a slot names an ID.
#define sentinel 0
#define published(id) (10 + (id))
#define edited(id) (20 + (id))
#define named(id) (30 + (id))
#define refined(id) (40 + (id))
#define id_of(word) ((word) % 10)

// The closed dispenser: the all-ones ID in silicon, past every count here.
#define closed_id 15

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef watermark_order
#define watermark_order order_release
#endif
#ifndef appender_order
#define appender_order order_release
#endif
#ifndef covered_order
#define covered_order order_release
#endif
#ifndef back_link_fence
#define back_link_fence true
#endif
#ifndef key_before_turn
#define key_before_turn true
#endif
#ifndef watermark_filter
#define watermark_filter true
#endif
#ifndef spends_plan
#define spends_plan true
#endif
#ifndef lock_recheck
#define lock_recheck false
#endif
#ifndef save_locks_rows
#define save_locks_rows true
#endif
#ifndef snapshot_check
#define snapshot_check true
#endif
#ifndef refresh_order
#define refresh_order order_acquire
#endif
#ifndef dispenser_closes
#define dispenser_closes true
#endif
#ifndef refine_closes
#define refine_closes true
#endif

bool editing[rows];      // the rows under their locks
byte published_at[rows]; // the history index of each row's publication
byte dispensed_at[rows]; // the history index of each ID's dispense, which its hosting chunk precedes
bool taken[rows];        // the IDs a dispense or a load took
bool linked;             // the inserter appended its back-link into row 0

// a traversal's read of slot zero below the watermark it acquired, and of the key column behind it
inline read_published(t, id, seen) {
    atomic {
        load(t, row_lock(id), order_acquire, seen);
        assert(memory != sc || seen != sentinel || editing[id]);
        assert(seen_write(t, row_lock(id)) >= published_at[id])
    };
    if
    :: named_rows -> load(t, key(id), order_relaxed, seen); assert(seen == named(id))
    :: else
    fi
}

// `neighbors_ref::lock`: the peek waits out a holder, then the exchange takes the row unless it
// displaces the sentinel a peer locked with in between
inline lock(t, id, displaced) {
    do
    :: (newest_value(row_lock(id)) != sentinel);
        atomic {
            read_modify_write(t, row_lock(id), order_acq_rel, displaced, sentinel);
            editing[id] = editing[id] || displaced != sentinel
        };
        if
        :: displaced != sentinel -> break
        :: else
        fi
    od
}

// `neighbors_ref::unlock`: the release store of slot zero, which republishes the tail written under the lock
inline unlock(t, id, first) {
    atomic {
        store(t, row_lock(id), order_release, first);
        editing[id] = false
    }
}

// the rest of `insert` once its ID is dispensed
inline insert_dispensed(t, id, first_invisible, seen, displaced, watermark) {
    if
    :: named_rows && key_before_turn -> store(t, key(id), order_relaxed, named(id))
    :: else
    fi;
    load(t, last_visible_id, order_acquire, first_invisible);
    if
    :: first_invisible > 0 -> read_published(t, 0, seen)
    :: else
    fi;
    // the own row: relaxed tail, then slot zero with the release that publishes it
    store(t, row_tail(id), order_relaxed, published(id));
    atomic {
        store(t, row_lock(id), order_release, published(id));
        published_at[id] = newest_write(row_lock(id))
    };
    // back_link_neighbors: the fence, then the row lock, the tail, the unlock
    if
    :: first_invisible > 0 ->
        if
        :: back_link_fence -> fence_release(t)
        :: else
        fi;
        lock(t, 0, displaced);
        atomic {
            store(t, row_tail(0), order_relaxed, edited(id));
            linked = true
        };
        unlock(t, 0, edited(id))
    :: else
    fi;
    take_turn(t, last_visible_id, id, id + 1, watermark_order, watermark);
    if
    :: named_rows && !key_before_turn -> store(t, key(id), order_relaxed, named(id))
    :: else
    fi
}

// `insert` of one row: identity, the key in `on_assigned`, the walk, the publication, the back-link,
// the turn; a closed dispenser refuses it before anything is written
inline insert(t) {
    int id, first_invisible, seen, displaced, watermark;
    // identity, one ID per node, chaining the hosting chunk's growth through the dispenser: the
    // bounded add of `dispense_`, refused only by a closed dispenser
    atomic {
        read_modify_write_if(t, next_dispensed_id, order_acq_rel, id != closed_id, id, id + 1);
        if
        :: id != closed_id ->
            assert(!taken[id]);
            taken[id] = true;
            dispensed_at[id] = newest_write(next_dispensed_id)
        :: else
        fi
    };
    if
    :: id != closed_id -> insert_dispensed(t, id, first_invisible, seen, displaced, watermark)
    :: else
    fi
}

// a search: the watermark acquired, and every row below it read as published
inline search(t, watermark) {
    int seen;
    load(t, last_visible_id, order_acquire, watermark);
    if
    :: watermark > 0 -> read_published(t, 0, seen)
    :: else
    fi;
    if
    :: watermark > 1 -> read_published(t, 1, seen)
    :: else
    fi
}

// `dense::ram_cursor::follow` of a staged ID: `seek` refuses it at or past the cursor's snapshot,
// else locates its row through the chunk directory, which a reader reaches only behind the ID's
// dispense, and reads it as a traversal does
inline follow(t, staged, visible_count, seen) {
    if
    :: id_of(staged) < visible_count || !snapshot_check ->
        assert(seen_write(t, next_dispensed_id) >= dispensed_at[id_of(staged)]);
        read_published(t, id_of(staged), seen)
    :: else
    fi
}

// a row the load path lands: the tail and slot zero relaxed, published by the release that follows
inline land_row(t, id) {
    store(t, row_tail(id), order_relaxed, published(id));
    atomic {
        store(t, row_lock(id), order_relaxed, published(id));
        published_at[id] = newest_write(row_lock(id))
    }
}

// the load path's `finish` of the rows from `base` to `bound`: an open dispenser moved past them by
// one compare-exchange from `base`, a closed one left closed, then the watermark released, which
// wakes the turns armed on it as every store to it does
inline finish_load(t, base, bound, expected, moved) {
    expected = base;
    compare_exchange(t, next_dispensed_id, appender_order, expected, bound, moved);
    end_turn(t, last_visible_id, watermark_order, bound)
}

// `close_`: the watermark acquired, then one compare-exchange from it to the closed state, refused
// while a batch sits between its dispense and its turn
inline close_dispenser(t, visible, closed) {
    load(t, last_visible_id, order_acquire, visible);
    compare_exchange(t, next_dispensed_id, order_acq_rel, visible, closed_id, closed)
}

// `reopen_`: from the closed state to the watermark acquired, past every row a load published meanwhile
inline reopen_dispenser(t, visible, expected, reopened) {
    load(t, last_visible_id, order_acquire, visible);
    expected = closed_id;
    compare_exchange(t, next_dispensed_id, order_acq_rel, expected, visible, reopened)
}

// `refine_one_` on row 0, without its walk, which writes nothing: the lock, the tail rewritten
// relaxed, and the unlock with the first neighbor the re-selection kept, the value the lock
// displaced; then `back_link_neighbors` into row 1, which names row 0 already: the release fence,
// the lock, and the unlock with the value it held
inline refine(t) {
    int displaced;
    lock(t, 0, displaced);
    store(t, row_tail(0), order_relaxed, refined(0));
    store(t, second_tail, order_relaxed, refined(0));
    unlock(t, 0, displaced);
    fence_release(t);
    lock(t, 1, displaced);
    unlock(t, 1, displaced)
}

// a save's copy of one row: slot zero and the tail under the row's lock, or both relaxed
inline copy_row(t, id, lock_copy, tail_copy, second_copy) {
    if
    :: save_locks_rows -> lock(t, id, lock_copy)
    :: else -> load(t, row_lock(id), order_relaxed, lock_copy)
    fi;
    load(t, row_tail(id), order_relaxed, tail_copy);
    if
    :: refined_rows && id == 0 -> load(t, second_tail, order_relaxed, second_copy)
    :: else
    fi;
    if
    :: save_locks_rows -> unlock(t, id, lock_copy)
    :: else
    fi
}

// `plan_save` and `finish` under `concurrent_k`: three acquire loads, every row copied under its
// lock, the acquire fence that pairs with the one heading `back_link_neighbors`, and the relaxed
// re-check
inline save(t) {
    byte covered_rows;
    int visible, dispensed, allocated, again;
    int lock_copy_0, tail_copy_0, second_copy, lock_copy_1, tail_copy_1;
    // plan_save: the watermark, refused while an insert is mid-flight; a closed dispenser has none
    load(t, last_visible_id, order_acquire, visible);
    load(t, next_dispensed_id, order_acquire, dispensed);
    load(t, last_allocated_id, order_acquire, allocated);
    assert(!appended_rows || visible == 0 || dispensed == visible || dispensed == closed_id);
    if
    :: visible != dispensed && dispensed != closed_id -> goto contended
    :: else
    fi;
    // the copies, each tallied on this plan alone
    if
    :: visible > 0 -> copy_row(t, 0, lock_copy_0, tail_copy_0, second_copy); covered_rows++
    :: else
    fi;
    if
    :: visible > 1 -> copy_row(t, 1, lock_copy_1, tail_copy_1, second_copy); covered_rows++
    :: else
    fi;
    // finish: the tally whole, then the fence, then the re-check
    assert(covered_rows == visible);
    fence_acquire(t);
    load(t, last_visible_id, order_relaxed, again);
    if
    :: again != visible -> goto contended
    :: else
    fi;
    load(t, next_dispensed_id, order_relaxed, again);
    if
    :: again != visible && again != closed_id -> goto contended
    :: else
    fi;
    load(t, last_allocated_id, order_relaxed, again);
    if
    :: again != allocated -> goto contended
    :: else
    fi;
    // the slot-zero words re-read, refused where a copy caught one held or it moved since
    if
    :: lock_recheck ->
        if
        :: visible > 0 ->
            load(t, row_lock(0), order_relaxed, again);
            if
            :: lock_copy_0 == sentinel || again != lock_copy_0 -> goto contended
            :: else
            fi
        :: else
        fi;
        if
        :: visible > 1 ->
            load(t, row_lock(1), order_relaxed, again);
            if
            :: lock_copy_1 == sentinel || again != lock_copy_1 -> goto contended
            :: else
            fi
        :: else
        fi
    :: else
    fi;
    // settled: whole rows, none locked, and a refined row's tail slots from one rewrite
    if
    :: visible > 0 ->
        assert(lock_copy_0 != sentinel && (refined_rows -> tail_copy_0 == second_copy : lock_copy_0 == tail_copy_0))
    :: else
    fi;
    if
    :: visible > 1 -> assert(lock_copy_1 != sentinel && lock_copy_1 == tail_copy_1)
    :: else
    fi;
contended:
    skip
}
