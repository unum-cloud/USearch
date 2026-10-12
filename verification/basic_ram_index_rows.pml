/**
 *  @brief The basic index's rows under their mutex, behind the gate's shared and unique sides.
 *
 *  One neighbor row from `include/usearch/basic/ram_index.hpp` under its `basic::mutex`, edited by
 *  the manual `link` and `unlink` and by an insert's `back_link_` behind the shared side of
 *  `basic::shared_mutex`, copied by the traversal's `copy_row_`, the cursor's `seek` and `degree_of`
 *  under the row mutex alone, and read plain by `health` and `plan_save` on the unique side. The
 *  gate inlines are the header's lines: the acquire add and the release fence of `lock_shared`, the
 *  release subtract of `unlock_shared`, the acquire compare-exchange from a free word of
 *  `try_lock`, and the release clear of `unlock`. The row mutex is the header's
 *  test-and-test-and-set: a peek at the flag, the acquire exchange, the release store.
 *  `basic_ram_index_steps.pml` covers the load path.
 *
 *  Row 0 holds two edges at most, the model's bound rather than `link`'s, which grows a row past
 *  any extent, and starts with one, to node 0; node 1 is the newcomer. A linker links node 1 into
 *  the row, which `link` allows once the watermark covers it, or unlinks the edge to node 0, moving
 *  the last slot into its place. The inserter stores node 1's key and back-links before its batch's
 *  turn: an append when the row has room, else the re-selection that clears the row and writes it
 *  anew; then it moves the watermark to 2 with a release. A reader acquires the watermark, reads
 *  `key_of` for node 1 when it is published, copies the row under its mutex, and follows every
 *  copied edge below its snapshot to that node's key. A unique taker reads the row plain. The
 *  idempotence re-checks of `link` and `back_link_`, and the re-selection's off-lock scoring with
 *  its relock and revalidation, are `basic_ram_index_back_link.pml`'s: no ordering claim rests on
 *  them.
 *
 *  Invariants:
 *  - a copied row holds no unwritten slot, under the row mutex or the unique side;
 *  - a reader that acquired the watermark reads the published key, for `key_of` and for every edge
 *    it follows; under `sc` the ghost adds that the record was published before the reader's
 *    snapshot, which is why the traversal skips IDs at or above it.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: the watermark's release carries the newcomer's
 *      key; without it, `key_of` below the watermark reads no key
 *  @verify fail rc11 row_lock_order=order_relaxed: the row lock's exchange acquires the previous
 *      holder's edit; without it, a copy taken under the lock holds the new size beside an
 *      unwritten slot
 *  @verify fail rc11 row_unlock_order=order_relaxed: the row unlock releases the edit to the next
 *      holder; without it, the next holder copies the same torn row
 *  @verify fail rc11 unlock_shared_order=order_relaxed: `unlock_shared`'s release carries the
 *      shared side's edits through the gate; without it, the unique side reads the row torn
 *  @verify fail sc watermark_filter=false: the frontier is a protocol claim, not an ordering one:
 *      the back-link lands before the newcomer's turn and the lock carries its record, so under
 *      `rc11` following its edge reads the key just fine, and the filter keeps that every record a
 *      walk reads was published before the walk's snapshot; without it, the reader follows an edge
 *      to an ID at or above its snapshot
 */
#define thread_count 4
#define location_count 7
#define history_depth 11
#include "weak_memory.pml"
#include "basic_gate.pml"

// The words, by the index member each stands for; only the newcomer's key is ever read.
#define shared_mutex 0
#define row_flag 1
#define row_size 2
#define row_slot(position) (3 + (position))
#define last_visible_id 5
#define newcomer_key 6

// The values: a slot holds an edge or nothing yet, a key word holds the record's key or nothing yet.
#define hole 0
#define linked(id) (1 + (id))
#define id_of(slot) ((slot) - 1)
#define keyed(id) (10 + (id))
#define row_capacity 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef watermark_order
#define watermark_order order_release
#endif
#ifndef row_lock_order
#define row_lock_order order_acquire
#endif
#ifndef row_unlock_order
#define row_unlock_order order_release
#endif
#ifndef unlock_shared_order
#define unlock_shared_order order_release
#endif
#ifndef watermark_filter
#define watermark_filter true
#endif

bool editing;            // a mutator holds the row mutex
bool published;          // the inserter moved the watermark past node 1
bool snapshot_published; // it had, when the reader took its snapshot

// lock_shared: the blind acquire add, backed off behind a unique holder, then the release fence
inline lock_shared(t, observed) {
    do
    :: read_modify_write(t, shared_mutex, order_acquire, observed, stepped(observed) + 1);
        if
        :: unique_of(observed) ->
            read_modify_write(t, shared_mutex, order_release, observed, observed - 1);
            (!unique_of(newest_value(shared_mutex)))
        :: else -> break
        fi
    od;
    fence_release(t)
}
inline unlock_shared(t, observed) { read_modify_write(t, shared_mutex, unlock_shared_order, observed, observed - 1) }

// the row mutex: the peek, then the claiming exchange, one step so the exchange never reads the
// flag back set; the caller declares `displaced`, since a local here costs a step
inline lock_row(t) {
    atomic {
        newest_value(row_flag) == hole;
        read_modify_write(t, row_flag, row_lock_order, displaced, 1)
    }
}
inline unlock_row(t) { store(t, row_flag, row_unlock_order, hole) }

// the row as one holder sees it: the size, then the slots below it
inline copy_row(t, size, first, second) {
    load(t, row_size, order_relaxed, size);
    if :: size > 0 -> load(t, row_slot(0), order_relaxed, first) :: else fi;
    if :: size > 1 -> load(t, row_slot(1), order_relaxed, second) :: else fi;
    assert(size == 0 || first != hole);
    assert(size < 2 || second != hole)
}

// the traversal's translation frontier: an edge below the snapshot leads to a record the snapshot covers
inline follow(t, slot, watermark, seen) {
    if
    :: slot == linked(1) && (!watermark_filter || id_of(slot) < watermark) ->
        if
        :: memory == sc -> assert(snapshot_published)
        :: else
        fi;
        load(t, newcomer_key, order_relaxed, seen);
        assert(seen == keyed(1))
    :: else
    fi
}

proctype linker(byte t) {
    int observed, visible, size, first, last, displaced;
    bool linking;
    if :: linking = true :: linking = false fi;
    // link(0, 1) needs node 1 below the watermark; unlink(0, 0) always may
    load(t, last_visible_id, order_acquire, visible);
    if
    :: linking && id_of(linked(1)) >= visible -> goto refused
    :: else
    fi;
    lock_shared(t, observed);
    lock_row(t);
    editing = true;
    load(t, row_size, order_relaxed, size);
    if
    :: linking && size < row_capacity ->
        store(t, row_slot(size), order_relaxed, linked(1));
        store(t, row_size, order_relaxed, size + 1)
    :: !linking && size > 0 ->
        // the edge to node 0 sits in slot 0 unless the rewrite dropped it; the last slot moves into its place
        load(t, row_slot(0), order_relaxed, first);
        if
        :: first == linked(0) ->
            load(t, row_slot(size - 1), order_relaxed, last);
            store(t, row_slot(0), order_relaxed, last);
            store(t, row_size, order_relaxed, size - 1)
        :: else
        fi
    :: else
    fi;
    editing = false;
    unlock_row(t);
    unlock_shared(t, observed);
refused:
    skip
}

proctype inserter(byte t) {
    int observed, size, displaced;
    lock_shared(t, observed);
    // the record, then back_link_ before the turn: an append when the row has room, else the
    // re-selection clears the row and writes it anew
    store(t, newcomer_key, order_relaxed, keyed(1));
    lock_row(t);
    editing = true;
    load(t, row_size, order_relaxed, size);
    if
    :: size < row_capacity ->
        store(t, row_slot(size), order_relaxed, linked(1));
        store(t, row_size, order_relaxed, size + 1)
    :: else ->
        store(t, row_size, order_relaxed, 0);
        store(t, row_slot(0), order_relaxed, linked(1));
        store(t, row_size, order_relaxed, 1)
    fi;
    editing = false;
    unlock_row(t);
    // the watermark turn that publishes it
    atomic {
        store(t, last_visible_id, watermark_order, 2);
        published = true
    };
    unlock_shared(t, observed)
}

// copy_row_, the cursor's seek and degree_of: one watermark snapshot, then the row under its mutex
proctype reader(byte t) {
    int watermark, seen, size, first, second, displaced;
    atomic {
        load(t, last_visible_id, order_acquire, watermark);
        // only `sc` reads the ghost, so under the other models it stays false and adds no states
        snapshot_published = memory == sc && published
    };
    // key_of: the same watermark gate, then a plain read
    if
    :: watermark > 1 -> load(t, newcomer_key, order_relaxed, seen); assert(seen == keyed(1))
    :: else
    fi;
    lock_row(t);
    copy_row(t, size, first, second);
    unlock_row(t);
    if :: size > 0 -> follow(t, first, watermark, seen) :: else fi;
    if :: size > 1 -> follow(t, second, watermark, seen) :: else fi
}

// health and plan_save: the unique side reads the row plain, every mutator released before it
proctype unique_taker(byte t) {
    int observed, size, first, second;
    bool succeeded;
    try_lock(t, shared_mutex, observed, succeeded);
    if
    :: succeeded ->
        if
        :: memory == sc -> assert(!editing)
        :: else
        fi;
        copy_row(t, size, first, second);
        unlock(t, shared_mutex, observed)
    :: else
    fi
}

// the words that do not start at zero take their value as the initial write, before any thread runs
init {
    atomic {
        seed(row_size, 1);
        seed(row_slot(0), linked(0));
        seed(last_visible_id, id_of(linked(1)));
        run linker(0); run inserter(1); run reader(2); run unique_taker(3)
    }
}
