/**
 *  The dense index's publication protocol from `include/usearch/dense/ram_index.hpp`: identity
 *  dispensed one range per batch from `next_dispensed_id_`, rows written and unlocked below it,
 *  the hole-free `last_visible_id_` watermark advanced in dispensed order, and the readers that
 *  rest on the watermark - a traversal, a cursor, and a concurrent save with its settlement probe.
 *
 *  Two inserters of one row each dispense IDs 0 and 1. Each writes its own row's tail relaxed
 *  and slot zero with a release, the publication; the second, when it saw the first below the
 *  watermark, back-links into row 0: a release fence, the slot-zero lock exchange, a relaxed
 *  tail store, the release unlock. A searcher acquires the watermark and reads every row below
 *  it. A saver runs `plan_save` and `finish` under `concurrent_k`: three acquire loads, the row
 *  copies as relaxed loads, an acquire fence, and the relaxed re-check of the three counters.
 *
 *  Invariants:
 *  - publication: a row below the watermark a reader acquired is visible to it as published.
 *    `-Dwithout_watermark_release` drops the release from the watermark store, and it fails;
 *  - settlement: a settled save copied no locked row and no torn edit. The release fence at the
 *    head of `back_link_neighbors` is what carries the relaxed tail stores into the re-check;
 *    `-Dwithout_release_fence` drops it, and a settled copy holds a new tail beside an old slot;
 *  - `-Dscenario=appender`: the load path's `finish`, one release no-return add on `next_dispensed_id_`
 *    before the watermark's release store. A saver that acquired the watermark must read the
 *    dispenser level with it. `-Dwithout_appender_release` makes the add relaxed: still carried under
 *    the C++ model, but posted under `-Dmemory=far`, where it may be in flight when the
 *    watermark lands and the plan reports contention on a settled index. A successful `finish`
 *    spends the plan; `-Dtwice_finished` publishes twice, and the saver finds the dispenser
 *    past the watermark for good;
 *  - `-Dscenario=keyed`: the sharded index's key column, written in `on_assigned` right after the
 *    dispense, before the walk and the publication, so every row a reader locates carries its
 *    key. The searcher then walks: one relaxed read of row 0's tail, whose dispensed id is
 *    located only below the snapshot - the traversal's `neighbor_id >= first_invisible_id`
 *    filter. `-Dwithout_key_before_turn` moves the key store after the watermark turn, and a
 *    reader finds a row without one; `-Dwithout_watermark_filter` follows the fresher id, whose
 *    watermark turn has not come and whose row the snapshot did not carry;
 *  - `-Dscenario=cursor`: `dense_ram_cursor` on the saver's thread: the constructor's acquire
 *    of the watermark, `seek` acquiring slot zero exactly once per try and waiting out a
 *    sentinel on the index's monitor, the relaxed tail, `follow` refusing a staged id at or
 *    past the snapshot, `refresh` widening the snapshot, and the same follow admitted.
 *    `-Dwithout_watermark_release` fails through the searcher; the follower's own seek waits
 *    out the stale sentinel a traversal would trip on, and the row it follows rides the
 *    back-link fence into even a relaxed watermark store, so the follower itself reads a
 *    followed row stale only when `-Dwithout_release_fence` drops that fence as well;
 *  - `-Dscenario=steps`: the load path's fan-out: two steps land one row each and tally it
 *    on `covered` with a release add, the one word standing for the `covered_vectors` and
 *    `covered_neighbors` each `load_plan_t::load` step adds to; the finisher acquires the
 *    tally and, when whole, bumps the dispenser and releases the watermark. The tally's
 *    release is what carries the rows into the watermark: `-Dwithout_covered_release` makes
 *    it relaxed, and the searcher finds a row below the watermark unpublished;
 *  - `-Dscenario=savers`: two concurrent saves on the searcher's and the saver's threads, each
 *    with its own plan and its own covered tallies; `-Dwithout_release_fence` fails for either;
 *  - `-Dscenario=batched`: a batch's commit, which dispenses its whole range with one add
 *    and moves the watermark once. Two commits of two rows each; a commit's leader dispenses,
 *    hands the range to a helper as the executor's fan-out would, each wires one row, the helper
 *    joins with a release the leader acquires, and the leader takes one turn from the range's
 *    first ID to its end. A row's walk admits the rows below its snapshot, and past it a row of
 *    its own range, or of the other commit once an acquire of the dispenser shows it dispensed,
 *    only once an acquire of that row's slot zero shows it published; it back-links into row 0
 *    behind the fence whenever it reached it, so commits in flight together link to each other.
 *    The searcher acquires the watermark, which never lies strictly inside a range, reads every
 *    row below it as published, and follows row 0's tail only below its snapshot, so a back-link
 *    into a later commit's rows never leads into them. `-Dwithout_range_turn` takes a turn per
 *    row, and the searcher sees half a commit; `-Dwithout_own_range_check` admits the sibling row
 *    unchecked, and a walk reads it before its wirer unlocked it.
 */
#include "weak_memory.pml"

// The scenarios: two inserters, the load path's appender, the keyed rows and their walk, the
// cursor, the load fan-out, two savers, or two batched commits.
#define inserters 1
#define appender 2
#define keyed 3
#define cursor 4
#define steps 5
#define savers 6
#define batched 7
#ifndef scenario
#define scenario inserters
#endif

// The threads, by role, apart from the processes that play them; the inserters, the steps, the
// savers and the commits name themselves as they start.
#define loader_thread 0
#if scenario == batched
#define leader_thread(commit) ((commit) * 2)
#define helper_thread(commit) ((commit) * 2 + 1)
#define searcher_thread 4
#else
#define searcher_thread 2
#endif
#define saver_thread 3
#define cursor_thread 3
#define finisher_thread 3

// The words, by the index member each stands for; a row is its lock word and one tail slot.
// The key column and the load plan's tally take the last words in turn: no scenario has both.
// The commits' four rows keep slot zero alone, but for row 0's tail, which the back-links edit.
#if scenario == batched
#define range_rows 2
#define rows 4
#define next_dispensed_id 0
#define last_visible_id 1
#define row_lock(id) (2 + (id))
#define row_tail(id) 6
#define joined(commit) (7 + (commit))
#else
#define rows 2
#define next_dispensed_id 0
#define last_visible_id 1
#define last_allocated_id 2
#define row_lock(id) (3 + (id) * 2)
#define row_tail(id) (4 + (id) * 2)
#define key(id) (7 + (id))
#define covered 7
#endif

// The values rows and keys carry: the sentinel is the all-ones id in silicon, and a slot names an id.
#define sentinel 0
#define published(id) (10 + (id))
#define edited(id) (20 + (id))
#define named(id) (30 + (id))
#define id_of(word) ((word) % 10)

#ifdef without_watermark_release
#define watermark_order order_relaxed
#else
#define watermark_order order_release
#endif
#ifdef without_appender_release
#define appender_order order_relaxed
#else
#define appender_order order_release
#endif
#ifdef without_covered_release
#define covered_order order_relaxed
#else
#define covered_order order_release
#endif

// the traversal's frontier: only an id below the snapshot may be located
#ifdef without_watermark_filter
#define admitted(id, snapshot) true
#else
#define admitted(id, snapshot) ((id) < (snapshot))
#endif

#if scenario == savers
#define saver_count 2
#else
#define saver_count 1
#endif

byte inserters_started;
byte steps_started;
byte savers_started;
bool editing; // row 0 is under its lock
#if memory != sequential
byte published_at[rows]; // the history index of each row's publication
#endif

// a traversal's read of slot zero below the watermark it acquired, and of the key column behind it
inline read_published(t, id, seen) {
    atomic {
        load(t, row_lock(id), order_acquire, seen);
#if memory == sequential
        assert(seen != sentinel || editing);
#else
        assert(current[thread_index(t, row_lock(id))] >= published_at[id]);
#endif
    };
#if scenario == keyed || scenario == cursor
    load(t, key(id), order_relaxed, seen);
    assert(seen == named(id));
#endif
}

#if scenario == batched
byte commits_started;
byte helpers_started;
byte range_first[2];  // the range each commit dispensed, handed to its helper
bool dispatched[2];   // the fan-out's dispatch, which carries no row
bool wired[rows];     // each row unlocked, set with the unlock

// a read of a row a walk admitted: unlocked before it, and in the reader's view
inline read_wired(t, id, seen) {
    atomic {
        load(t, row_lock(id), order_acquire, seen);
        assert(wired[id]);
#if memory != sequential
        assert(current[thread_index(t, row_lock(id))] >= published_at[id]);
#endif
    }
}

// the watermark turn from @p from to @p to, strictly in dispensed order
inline take_turn(t, from, to) {
    (newest_value(last_visible_id) == from);
    do
    :: load(t, last_visible_id, order_acquire, watermark);
       if
       :: watermark == from -> break
       :: else
       fi
    od;
    store(t, last_visible_id, watermark_order, to)
}

// one row of a commit: the walk below the snapshot and over the published rows dispensed before it
// began, slot zero's release, then the back-link into row 0 when the walk reached it
inline wire(t, id, sibling) {
    load(t, last_visible_id, order_acquire, first_invisible);
    load(t, next_dispensed_id, order_acquire, first_undispensed);
    linked = first_invisible > 0;
    if
    :: linked -> read_wired(t, 0, seen)
    :: else
    fi;
    // past the snapshot, the other commit's first row, once dispensed and its slot zero shows it published
    other = (id < range_rows -> range_rows : 0);
    if
    :: !linked && other < first_undispensed ->
        load(t, row_lock(other), order_acquire, seen);
        if
        :: seen != sentinel -> read_wired(t, other, seen); linked = other == 0
        :: else
        fi
    :: else
    fi;
#ifdef without_own_range_check
    read_wired(t, sibling, seen);
#else
    load(t, row_lock(sibling), order_acquire, seen);
    if
    :: seen != sentinel -> read_wired(t, sibling, seen)
    :: else
    fi;
#endif
    if
    :: id == 0 -> store(t, row_tail(0), order_relaxed, published(0))
    :: else
    fi;
    atomic {
        store(t, row_lock(id), order_release, published(id));
        wired[id] = true;
#if memory != sequential
        published_at[id] = newest[row_lock(id)]
#endif
    };
    if
    :: linked ->
#ifndef without_release_fence
        fence_release(t);
#endif
        do
        :: read_modify_write(t, row_lock(0), order_acq_rel, displaced, sentinel);
           if
           :: displaced != sentinel -> break
           :: else -> (newest_value(row_lock(0)) != sentinel)
           fi
        od;
        store(t, row_tail(0), order_relaxed, edited(id));
        store(t, row_lock(0), order_release, edited(id))
    :: else
    fi
}

// a commit's leader: the whole range from one add, its own row, the helper's join, then one turn
active [2] proctype committer() {
    byte me, other;
    int first, first_invisible, first_undispensed, seen, displaced, watermark, joins;
    bool linked;
    atomic { me = commits_started; commits_started++ };
    read_modify_write(leader_thread(me), next_dispensed_id, order_acq_rel, first, first + range_rows);
    atomic { range_first[me] = first; dispatched[me] = true };
    wire(leader_thread(me), first, first + 1);
#ifdef without_range_turn
    take_turn(leader_thread(me), first, first + 1);
#endif
    (newest_value(joined(me)) != 0);
    do
    :: load(leader_thread(me), joined(me), order_acquire, joins);
       if
       :: joins != 0 -> break
       :: else
       fi
    od;
#ifndef without_range_turn
    take_turn(leader_thread(me), first, first + range_rows);
#endif
}

// the executor's second worker on a commit's range: the range's second row, then the join
active [2] proctype helper() {
    byte me, other;
    int first, first_invisible, first_undispensed, seen, displaced, watermark;
    bool linked;
    atomic { me = helpers_started; helpers_started++ };
    dispatched[me];
    first = range_first[me];
    wire(helper_thread(me), first + 1, first);
#ifdef without_range_turn
    take_turn(helper_thread(me), first + 1, first + 2);
#endif
    store(helper_thread(me), joined(me), order_release, 1)
}

// a search: the watermark acquired, never inside a range, every row below it published, and the
// walk through row 0's tail, which a later commit's back-link may name, filtered by the snapshot
active proctype searcher() {
    byte id;
    int watermark, seen, found;
    load(searcher_thread, last_visible_id, order_acquire, watermark);
    assert(watermark % range_rows == 0);
    id = 0;
    do
    :: id == watermark -> break
    :: else -> read_wired(searcher_thread, id, seen); id++
    od;
    if
    :: watermark > 0 ->
        load(searcher_thread, row_tail(0), order_relaxed, found);
        if
        :: admitted(id_of(found), watermark) -> read_wired(searcher_thread, id_of(found), seen)
        :: else
        fi
    :: else
    fi
}
#elif scenario == appender
// the load path's `finish`: rows landed, the dispenser bumped without a return, the watermark released
active proctype loader() {
    store(loader_thread, row_tail(0), order_relaxed, published(0));
    store(loader_thread, row_lock(0), order_relaxed, published(0));
    store(loader_thread, row_tail(1), order_relaxed, published(1));
    store(loader_thread, row_lock(1), order_relaxed, published(1));
#if memory != sequential
    published_at[0] = newest[row_lock(0)];
    published_at[1] = newest[row_lock(1)];
#endif
    add_no_return(loader_thread, next_dispensed_id, appender_order, 2);
    store(loader_thread, last_visible_id, watermark_order, 2);
#ifdef twice_finished
    // a second `finish` on a spent plan: the dispenser moves again, the watermark does not
    add_no_return(loader_thread, next_dispensed_id, appender_order, 2);
    store(loader_thread, last_visible_id, watermark_order, 2);
#endif
    landed(loader_thread)
}
#elif scenario == steps
// the load fan-out: each step lands one row as the loader does, then tallies it on the plan
active [2] proctype step() {
    byte me;
    atomic { me = steps_started; steps_started++ };
    store(me, row_tail(me), order_relaxed, published(me));
    atomic {
        store(me, row_lock(me), order_relaxed, published(me));
#if memory != sequential
        published_at[me] = newest[row_lock(me)]
#endif
    };
    add_no_return(me, covered, covered_order, 1);
    landed(me)
}

// the load path's `finish` behind the fan-out: the tally acquired, and when whole, the publication
active proctype finisher() {
    int tally;
    load(finisher_thread, covered, order_acquire, tally);
    if
    :: tally == 2 ->
        add_no_return(finisher_thread, next_dispensed_id, appender_order, 2);
        store(finisher_thread, last_visible_id, watermark_order, 2)
    :: else
    fi;
    landed(finisher_thread)
}
#else
active [2] proctype inserter() {
    byte me;
    int id, first_invisible, seen, displaced, watermark;
    atomic { me = inserters_started; inserters_started++ };
    // identity, one id per node, chaining the hosting chunk's growth through the dispenser
    read_modify_write(me, next_dispensed_id, order_acq_rel, id, id + 1);
#if (scenario == keyed || scenario == cursor) && !defined(without_key_before_turn)
    // on_assigned: the sharded index writes the key column before the walk and the publication
    store(me, key(id), order_relaxed, named(id));
#endif
    load(me, last_visible_id, order_acquire, first_invisible);
    if
    :: first_invisible > 0 -> read_published(me, 0, seen)
    :: else
    fi;
    // the own row: relaxed tail, then slot zero with the release that publishes it
    store(me, row_tail(id), order_relaxed, published(id));
    atomic {
        store(me, row_lock(id), order_release, published(id));
#if memory != sequential
        published_at[id] = newest[row_lock(id)]
#endif
    };
    // back_link_neighbors: the fence, then the row lock, the tail, the unlock
    if
    :: first_invisible > 0 ->
#ifndef without_release_fence
        fence_release(me);
#endif
lock_row:
        atomic {
            read_modify_write(me, row_lock(0), order_acq_rel, displaced, sentinel);
            editing = editing || displaced != sentinel
        };
        if
        :: displaced == sentinel -> (newest_value(row_lock(0)) != sentinel); goto lock_row
        :: else
        fi;
        store(me, row_tail(0), order_relaxed, edited(id));
        atomic {
            store(me, row_lock(0), order_release, edited(id));
            editing = false
        }
    :: else
    fi;
    // the watermark turn, strictly in dispensed order
    (newest_value(last_visible_id) == id);
turn:
    load(me, last_visible_id, order_acquire, watermark);
    if
    :: watermark != id -> goto turn
    :: else
    fi;
    store(me, last_visible_id, watermark_order, id + 1);
#ifdef without_key_before_turn
    store(me, key(id), order_relaxed, named(id));
#endif
}
#endif

#if scenario != savers && scenario != batched
active proctype searcher() {
    int watermark, seen, found;
    load(searcher_thread, last_visible_id, order_acquire, watermark);
    if
    :: watermark > 0 -> read_published(searcher_thread, 0, seen)
    :: else
    fi;
    if
    :: watermark > 1 -> read_published(searcher_thread, 1, seen)
    :: else
    fi;
#if scenario == keyed
    // the walk: one relaxed read of row 0's tail, and the id it holds located only below the
    // snapshot - a fresher one, read from a re-linked slot, has no happens-before edge to its row yet
    if
    :: watermark > 0 ->
        load(searcher_thread, row_tail(0), order_relaxed, found);
        assert(id_of(found) < newest_value(next_dispensed_id)); // a row names dispensed ids alone
        if
        :: admitted(id_of(found), watermark) ->
#if memory == sequential
            assert(newest_value(last_visible_id) > id_of(found)); // a located id has taken its watermark turn
#endif
            read_published(searcher_thread, id_of(found), seen)
        :: else
        fi
    :: else
    fi;
#endif
}
#endif

#if scenario == cursor
// a dense_ram_cursor holder: the constructor's snapshot, seek(0), one follow refused, refresh, the follow admitted
active proctype follower() {
    int visible_count, first_value, staged, seen;
    load(cursor_thread, last_visible_id, order_acquire, visible_count);
    if
    :: visible_count == 0 -> goto past_end // seek(0) is malformed at or past the snapshot
    :: else
    fi;
    // seek: slot zero acquired exactly once per try, a sentinel waited out on the index's monitor, the tail relaxed
seek:
    (newest_value(row_lock(0)) != sentinel);
    load(cursor_thread, row_lock(0), order_acquire, first_value);
    if
    :: first_value == sentinel -> goto seek
    :: else
    fi;
    load(cursor_thread, row_tail(0), order_relaxed, staged);
    // follow: malformed at or past the snapshot, since the row it names may not have reached this cursor
    if
    :: id_of(staged) < visible_count -> read_published(cursor_thread, id_of(staged), seen)
    :: else
    fi;
    // refresh: the snapshot widens and never shrinks, and the staged id may be admitted now
    load(cursor_thread, last_visible_id, order_acquire, seen);
    assert(seen >= visible_count);
    visible_count = seen;
    if
    :: id_of(staged) < visible_count -> read_published(cursor_thread, id_of(staged), seen)
    :: else
    fi;
past_end:
    skip
}
#elif scenario != steps && scenario != batched
active [saver_count] proctype saver() {
    byte me, covered_rows;
    int visible, dispensed, allocated, again;
    int lock_copy_0, tail_copy_0, lock_copy_1, tail_copy_1;
    // a second saver takes the searcher's thread
    atomic { me = saver_thread - savers_started; savers_started++ };
    // plan_save: the watermark, refused while an insert is mid-flight
    load(me, last_visible_id, order_acquire, visible);
    load(me, next_dispensed_id, order_acquire, dispensed);
    load(me, last_allocated_id, order_acquire, allocated);
#if scenario == appender
    assert(visible == 0 || dispensed == visible);
#endif
    if
    :: visible != dispensed -> goto contended
    :: else
    fi;
    // the copies, each tallied on this plan alone
    if
    :: visible > 0 -> load(me, row_lock(0), order_relaxed, lock_copy_0); load(me, row_tail(0), order_relaxed, tail_copy_0); covered_rows++
    :: else
    fi;
    if
    :: visible > 1 -> load(me, row_lock(1), order_relaxed, lock_copy_1); load(me, row_tail(1), order_relaxed, tail_copy_1); covered_rows++
    :: else
    fi;
    // finish: the tally whole, then the fence pairs with the one heading back_link_neighbors, then the re-check
    assert(covered_rows == visible);
    fence_acquire(me);
    load(me, last_visible_id, order_relaxed, again);
    if
    :: again != visible -> goto contended
    :: else
    fi;
    load(me, next_dispensed_id, order_relaxed, again);
    if
    :: again != dispensed -> goto contended
    :: else
    fi;
    load(me, last_allocated_id, order_relaxed, again);
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
#endif
