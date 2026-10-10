/**
 *  The dense index's publication protocol from `include/usearch/dense/ram_index.hpp`: identity
 *  dispensed from `next_dispensed_id_`, rows written and unlocked below it, the hole-free
 *  `last_visible_id_` watermark advanced in dispensed order, and the readers that rest on the
 *  watermark: a traversal, a cursor, and a concurrent save with its settlement probe.
 *
 *  A row is slot zero, which locks and publishes it, and one tail slot. Every scenario beside this
 *  file defines its shape, `named_rows` for the key column `on_assigned` writes, and `appended_rows`
 *  for rows that arrive only through the load path's `finish`.
 */
#include "../weak_memory.pml"
#include "../watermark_turn.pml"

// The words, by the index member each stands for; the key column and the load plan's tally come last.
#define next_dispensed_id 0
#define last_visible_id 1
#define last_allocated_id 2
#define row_lock(id) (3 + (id) * 2)
#define row_tail(id) (4 + (id) * 2)
#define key(id) (7 + (id))
#define covered 9
#define rows 2

// The values rows and keys carry: the sentinel is the all-ones ID in silicon, and a slot names an ID.
#define sentinel 0
#define published(id) (10 + (id))
#define edited(id) (20 + (id))
#define named(id) (30 + (id))
#define id_of(word) ((word) % 10)

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

bool editing;            // row 0 is under its lock
byte published_at[rows]; // the history index of each row's publication

// a traversal's read of slot zero below the watermark it acquired, and of the key column behind it
inline read_published(t, id, seen) {
    atomic {
        load(t, row_lock(id), order_acquire, seen);
        assert(memory != sc || seen != sentinel || editing);
        assert(seen_write(t, row_lock(id)) >= published_at[id])
    };
    if
    :: named_rows -> load(t, key(id), order_relaxed, seen); assert(seen == named(id))
    :: else
    fi
}

// `insert` of one row: identity, the key in `on_assigned`, the walk, the publication, the back-link, the turn
inline insert(t) {
    int id, first_invisible, seen, displaced, watermark;
    // identity, one ID per node, chaining the hosting chunk's growth through the dispenser
    read_modify_write(t, next_dispensed_id, order_acq_rel, id, id + 1);
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
lock_row:
        atomic {
            read_modify_write(t, row_lock(0), order_acq_rel, displaced, sentinel);
            editing = editing || displaced != sentinel
        };
        if
        :: displaced == sentinel -> (newest_value(row_lock(0)) != sentinel); goto lock_row
        :: else
        fi;
        store(t, row_tail(0), order_relaxed, edited(id));
        atomic {
            store(t, row_lock(0), order_release, edited(id));
            editing = false
        }
    :: else
    fi;
    take_turn(t, last_visible_id, id, id + 1, watermark_order, watermark);
    if
    :: named_rows && !key_before_turn -> store(t, key(id), order_relaxed, named(id))
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

// a row the load path lands: the tail and slot zero relaxed, published by the release that follows
inline land_row(t, id) {
    store(t, row_tail(id), order_relaxed, published(id));
    atomic {
        store(t, row_lock(id), order_relaxed, published(id));
        published_at[id] = newest_write(row_lock(id))
    }
}

// the load path's `finish`: the dispenser bumped without a return, then the watermark released
inline finish_load(t) {
    add_no_return(t, next_dispensed_id, appender_order, 2);
    store(t, last_visible_id, watermark_order, 2)
}

// `plan_save` and `finish` under `concurrent_k`: three acquire loads, the row copies relaxed, the
// acquire fence that pairs with the one heading `back_link_neighbors`, and the relaxed re-check
inline save(t) {
    byte covered_rows;
    int visible, dispensed, allocated, again;
    int lock_copy_0, tail_copy_0, lock_copy_1, tail_copy_1;
    // plan_save: the watermark, refused while an insert is mid-flight
    load(t, last_visible_id, order_acquire, visible);
    load(t, next_dispensed_id, order_acquire, dispensed);
    load(t, last_allocated_id, order_acquire, allocated);
    assert(!appended_rows || visible == 0 || dispensed == visible);
    if
    :: visible != dispensed -> goto contended
    :: else
    fi;
    // the copies, each tallied on this plan alone
    if
    :: visible > 0 -> load(t, row_lock(0), order_relaxed, lock_copy_0); load(t, row_tail(0), order_relaxed, tail_copy_0); covered_rows++
    :: else
    fi;
    if
    :: visible > 1 -> load(t, row_lock(1), order_relaxed, lock_copy_1); load(t, row_tail(1), order_relaxed, tail_copy_1); covered_rows++
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
    :: again != dispensed -> goto contended
    :: else
    fi;
    load(t, last_allocated_id, order_relaxed, again);
    if
    :: again != allocated -> goto contended
    :: else
    fi;
    // settled: whole rows, none locked
    if
    :: visible > 0 -> assert(lock_copy_0 != sentinel && lock_copy_0 == tail_copy_0)
    :: else
    fi;
    if
    :: visible > 1 -> assert(lock_copy_1 != sentinel && lock_copy_1 == tail_copy_1)
    :: else
    fi;
contended:
    skip
}
