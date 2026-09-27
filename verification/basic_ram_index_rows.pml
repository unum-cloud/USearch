/**
 *  The basic index's rows from `include/usearch/basic/ram_index.hpp`: one neighbor row under
 *  its `basic_mutex`, edited by the manual `link` and `unlink` and by an insert's `back_link_`
 *  behind the shared side of `basic_shared_mutex`, copied by the traversal's `copy_row_`, the
 *  cursor's `seek` and `degree_of` under the row mutex alone, and read plain by `health` and
 *  `plan_save` on the unique side. The gate inlines are the header's lines: the acquire add and
 *  the release fence of `lock_shared`, the release subtract of `unlock_shared`, the acquire
 *  compare-exchange from a free word of `try_lock`, and the release clear of `unlock`. The row
 *  mutex is the header's test-and-test-and-set: a peek at the flag, the acquire exchange, the
 *  release store.
 *
 *  Row 0 holds two edges at most, the model's bound rather than `link`'s, which grows a row past
 *  any extent, and starts with one, to node 0; node 1 is the newcomer. A linker
 *  links node 1 into the row, which `link` allows once the watermark covers it, or unlinks the
 *  edge to node 0, moving the last slot into its place. The inserter stores node 1's key, moves
 *  the watermark to 2 with a release, and back-links: an append when the row has room, else the
 *  re-selection that clears the row and writes it anew. A reader acquires the watermark, reads
 *  `key_of` for node 1 when it is published, copies the row under its mutex, and follows every
 *  copied edge below its snapshot to that node's key. A unique taker reads the row plain. The
 *  idempotence re-checks of `link` and `back_link_`, and the re-selection's off-lock scoring with
 *  its relock and revalidation, are left out: no ordering claim rests on them.
 *
 *  Invariants:
 *  - a copied row holds no unwritten slot, under the row mutex or the unique side;
 *  - a reader that acquired the watermark reads the published key, for `key_of` and for every
 *    edge it follows; under `-Dmemory=sequential` the ghost adds that the record was stored
 *    before the reader's snapshot, which is why the traversal skips ids at or above it;
 *  - `-Dwithout_watermark_release`: the watermark store is relaxed, and `key_of` reads no key;
 *  - `-Dwithout_row_lock_acquire`: the row lock's exchange is relaxed, and a copy taken under
 *    the lock holds the new size beside an unwritten slot;
 *  - `-Dwithout_row_unlock_release`: the row unlock is relaxed, and carries no edit to the next
 *    holder, the same torn copy;
 *  - `-Dwithout_unlock_shared_release`: `unlock_shared` is relaxed, and the unique side reads a
 *    row torn the same way through the gate;
 *  - `-Dwithout_watermark_filter`, under `-Dmemory=sequential`: the reader follows an edge to
 *    an id at or above its snapshot, and reads a record its snapshot never covered.
 *
 *  `-Dscenario=steps`: the load path. Two steps each store their key and add their count
 *  to `covered` with a release; the finisher, holding the unique side, acquires the tally,
 *  reads both keys, stores the claim and identity counters relaxed and the watermark with a
 *  release, and unlocks. A reader runs `watermark_` and `key_of`: a watermark it
 *  acquired at 2 must show the dispenser at 2 and node 1's key.
 *  - `-Dwithout_covered_release`: the tally add is relaxed, and the finisher reads no key; under
 *    `-Dmemory=far` the add is posted, and lands carrying nothing;
 *  - `-Dwithout_counters_before_watermark`: the watermark goes first, and a reader that acquired
 *    it reads the dispenser behind it, so `watermark_` reports a settled load contended.
 */
#include "weak_memory.pml"

// The scenarios: the rows under their mutex, or the load path's steps.
#define rows 1
#define steps 2
#ifndef scenario
#define scenario rows
#endif

// The threads, by role, apart from the processes that play them; the steps name themselves as they start.
#define linker_thread 0
#define inserter_thread 1
#define reader_thread 2
#define unique_taker_thread 3
#define finisher_thread 3

// The words, by the index member each stands for.
#define shared_mutex 0
#if scenario == rows
#define row_flag 1
#define row_size 2
#define row_slot(position) (3 + (position))
#define last_visible_id 5
#else
#define covered 1
#define claimed_count 2
#define next_dispensed_id 3
#define last_visible_id 4
#endif
#define key(id) (5 + (id))

// The gate's fields: two holder bits, the unique bit, the generation above.
#define holders_of(word) ((word) & 3)
#define unique_of(word) (((word) >> 2) & 1)
#define unique_bit 4
#define generation_unit 8
#define stepped(word) ((word) + generation_unit)

// The values: a slot holds an edge or nothing yet, a key word holds the record's key or nothing yet.
#define hole 0
#define linked(id) (1 + (id))
#define id_of(slot) ((slot) - 1)
#define keyed(id) (10 + (id))
#define row_capacity 2

#ifdef without_watermark_release
#define watermark_order order_relaxed
#else
#define watermark_order order_release
#endif
#ifdef without_row_lock_acquire
#define row_lock_order order_relaxed
#else
#define row_lock_order order_acquire
#endif
#ifdef without_row_unlock_release
#define row_unlock_order order_relaxed
#else
#define row_unlock_order order_release
#endif
#ifdef without_unlock_shared_release
#define unlock_shared_order order_relaxed
#else
#define unlock_shared_order order_release
#endif
#ifdef without_covered_release
#define covered_order order_relaxed
#else
#define covered_order order_release
#endif

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
// try_lock: only from a genuinely free word, a refusal is the answer
inline try_lock(t, observed, succeeded) {
    load(t, shared_mutex, order_relaxed, observed);
    succeeded = false;
    do
    :: holders_of(observed) == 0 && !unique_of(observed) ->
        compare_exchange(t, shared_mutex, order_acquire, observed, stepped(observed) | unique_bit, succeeded);
        if :: succeeded -> break :: else fi
    :: else -> break
    od
}
inline unlock(t, observed) { read_modify_write(t, shared_mutex, order_release, observed, observed - unique_bit) }

#if scenario == rows
bool editing;         // a mutator holds the row mutex
bool key_stored;      // the inserter wrote node 1's key
bool snapshot_keyed;  // it had, when the reader took its snapshot

// the words that do not start at zero take their value as the initial write, before any thread runs
#if memory == sequential
inline seed(l, value) { words[l] = value }
#else
inline seed(l, value) { history_value[write_index(l, 0)] = value }
#endif
init {
    atomic {
        seed(row_size, 1);
        seed(row_slot(0), linked(0));
        seed(last_visible_id, id_of(linked(1)));
        run linker(); run inserter(); run reader(); run unique_taker()
    }
}

// the row mutex: the peek, then the claiming exchange, one step so the exchange never reads the flag back set
inline lock_row(t, displaced) {
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
#ifdef without_watermark_filter
    :: slot == linked(1) ->
#else
    :: slot == linked(1) && id_of(slot) < watermark ->
#endif
#if memory == sequential
        assert(snapshot_keyed);
#endif
        load(t, key(1), order_relaxed, seen);
        assert(seen == keyed(1))
    :: else
    fi
}

proctype linker() {
    int observed, visible, size, first, last, displaced;
    bool linking;
    if :: linking = true :: linking = false fi;
    // link(0, 1) needs node 1 below the watermark; unlink(0, 0) always may
    load(linker_thread, last_visible_id, order_acquire, visible);
    if
    :: linking && id_of(linked(1)) >= visible -> goto refused
    :: else
    fi;
    lock_shared(linker_thread, observed);
    lock_row(linker_thread, displaced);
    editing = true;
    load(linker_thread, row_size, order_relaxed, size);
    if
    :: linking && size < row_capacity ->
        store(linker_thread, row_slot(size), order_relaxed, linked(1));
        store(linker_thread, row_size, order_relaxed, size + 1)
    :: !linking && size > 0 ->
        // the edge to node 0 sits in slot 0 unless the rewrite dropped it; the last slot moves into its place
        load(linker_thread, row_slot(0), order_relaxed, first);
        if
        :: first == linked(0) ->
            load(linker_thread, row_slot(size - 1), order_relaxed, last);
            store(linker_thread, row_slot(0), order_relaxed, last);
            store(linker_thread, row_size, order_relaxed, size - 1)
        :: else
        fi
    :: else
    fi;
    editing = false;
    unlock_row(linker_thread);
    unlock_shared(linker_thread, observed);
refused:
    skip
}

proctype inserter() {
    int observed, size, displaced;
    lock_shared(inserter_thread, observed);
    // the record, then the watermark turn that publishes it
    atomic {
        store(inserter_thread, key(1), order_relaxed, keyed(1));
        key_stored = true
    };
    store(inserter_thread, last_visible_id, watermark_order, 2);
    // back_link_: an append when the row has room, else the re-selection clears the row and writes it anew
    lock_row(inserter_thread, displaced);
    editing = true;
    load(inserter_thread, row_size, order_relaxed, size);
    if
    :: size < row_capacity ->
        store(inserter_thread, row_slot(size), order_relaxed, linked(1));
        store(inserter_thread, row_size, order_relaxed, size + 1)
    :: else ->
        store(inserter_thread, row_size, order_relaxed, 0);
        store(inserter_thread, row_slot(0), order_relaxed, linked(1));
        store(inserter_thread, row_size, order_relaxed, 1)
    fi;
    editing = false;
    unlock_row(inserter_thread);
    unlock_shared(inserter_thread, observed)
}

// copy_row_, the cursor's seek and degree_of: one watermark snapshot, then the row under its mutex
proctype reader() {
    int watermark, seen, size, first, second, displaced;
    atomic {
        load(reader_thread, last_visible_id, order_acquire, watermark);
        snapshot_keyed = key_stored
    };
    // key_of: the same watermark gate, then a plain read
    if
    :: watermark > 1 -> load(reader_thread, key(1), order_relaxed, seen); assert(seen == keyed(1))
    :: else
    fi;
    lock_row(reader_thread, displaced);
    copy_row(reader_thread, size, first, second);
    unlock_row(reader_thread);
    if :: size > 0 -> follow(reader_thread, first, watermark, seen) :: else fi;
    if :: size > 1 -> follow(reader_thread, second, watermark, seen) :: else fi
}

// health and plan_save: the unique side reads the row plain, every mutator released before it
proctype unique_taker() {
    int observed, size, first, second;
    bool succeeded;
    try_lock(unique_taker_thread, observed, succeeded);
    if
    :: succeeded ->
#if memory == sequential
        assert(!editing);
#endif
        copy_row(unique_taker_thread, size, first, second);
        unlock(unique_taker_thread, observed)
    :: else
    fi
}
#else
#define step_count 2
byte steps_started;

// load_plan_t::load: the step's records, then its share of the tally
active [step_count] proctype step() {
    byte me;
    atomic { me = steps_started; steps_started++ };
    store(me, key(me), order_relaxed, keyed(me));
    add_no_return(me, covered, covered_order, 1);
    landed(me)
}

// load_plan_t::finish under plan_load's hold: the tally, then claims, identity, watermark last
active proctype finisher() {
    int observed, tally, first, second;
    bool succeeded;
    try_lock(finisher_thread, observed, succeeded);
    assert(succeeded);
    (newest_value(covered) == step_count);
    load(finisher_thread, covered, order_acquire, tally);
    if
    :: tally == step_count ->
        load(finisher_thread, key(0), order_relaxed, first);
        load(finisher_thread, key(1), order_relaxed, second);
        assert(first == keyed(0) && second == keyed(1));
#ifdef without_counters_before_watermark
        store(finisher_thread, last_visible_id, order_release, step_count);
        store(finisher_thread, claimed_count, order_relaxed, step_count);
        store(finisher_thread, next_dispensed_id, order_relaxed, step_count)
#else
        store(finisher_thread, claimed_count, order_relaxed, step_count);
        store(finisher_thread, next_dispensed_id, order_relaxed, step_count);
        store(finisher_thread, last_visible_id, order_release, step_count)
#endif
    :: else
    fi;
    unlock(finisher_thread, observed)
}

// watermark_ and key_of: a watermark acquired at the row count shows the dispenser there and the keys
active proctype reader() {
    int watermark, dispensed, seen;
    load(reader_thread, last_visible_id, order_acquire, watermark);
    if
    :: watermark == step_count ->
        load(reader_thread, next_dispensed_id, order_acquire, dispensed);
        assert(dispensed == step_count);
        load(reader_thread, key(1), order_relaxed, seen);
        assert(seen == keyed(1))
    :: else
    fi
}
#endif
