/**
 *  The sharded dense index from `include/usearch/dense/ram_indexes.hpp`: rows admitted into each
 *  shard's key plane through `fetch_add_if_at_most`, a refusal past admission handing them back
 *  with a release subtract, each landed row's key stored before its shard's watermark turn, and
 *  the readers that rest on the watermark - the search merge reading `keys[node]`, the monolith
 *  save taking one `plan_save` per shard, and the load path landing keys under one `finish`.
 *
 *  Two shards with planes of two rows. `-Dscenario=adding`, the default: two batch adders each
 *  bringing one row per shard, one single-row adder probing the shards from its routing base,
 *  and a searcher; the dispenser is a ghost, a row's publication is its watermark turn. The
 *  batch adder admits shard by shard, unwinds every shard admitted so far on a refusal, and may
 *  have its claims or fan-out refused after admission, which unwinds them all. A checker outside
 *  the memory model compares each plane's admission word with its dispenser once every adder is
 *  done. `-Dscenario=saving`: one batch adder, a saver running `plan_save` per shard, `finish`,
 *  then the plain key copy of `keys[0..visible)`, and a searcher; the dispenser is a word, since
 *  the cut compares it. `-Dscenario=loading`: a loader running `plan_load` per shard, the keys at
 *  the base, then `finish`, against a searcher on the still-empty index.
 *
 *  Invariants:
 *  - the plane: a dispensed id is below its shard's plane, and once every adder is done each
 *    plane's admission word equals its dispenser. `-Dwithout_withdraw` skips the hand-back after
 *    a refused fan-out, and the admission word stays above the dispenser;
 *  - publication: every id below a watermark a reader acquired has its key. The key lands before
 *    the watermark turn; `-Dwithout_key_before_watermark` moves it after, and a searcher reads an
 *    unset key under every model; `-Dwithout_watermark_release` drops the release from the
 *    watermark store, and the searcher reads it under the view model;
 *  - the cut: a settled `plan_save` copies no unset key below its cut, and, sequentially, a plan
 *    is refused exactly when a row is between its dispense and its publication, or was published
 *    between the plan's two loads. `-Dwhole_batches` asks the cut to capture whole batches, shard
 *    zero and shard one together, and fails under every model: the cut is per shard;
 *  - the load path: keys land before `finish` publishes the rows. `-Dwithout_keys_before_finish`
 *    moves them after, and the searcher reads an unset key under the sequential and view models.
 *
 *  `-Dwithout_single_consume` drops the `admission.consume()` the single-row `add` makes after
 *  `insert_reserved`, the header before the fix: the landed row is handed back and the
 *  admission word falls below the dispenser; with two batches behind it the plane admits one
 *  row more than it holds, and the next dispensed id names a key slot past the plane.
 */
#include "weak_memory.pml"

// The scenarios: the adders against the planes, the monolith save's cut, or the load path.
#define adding 1
#define saving 2
#define loading 3
#ifndef scenario
#define scenario adding
#endif

// The threads, by role, apart from the processes that play them; the batch adders name themselves as they start.
#if scenario == adding
#define single_adder_thread 2
#define searcher_thread 3
#elif scenario == saving
#define batch_adder_thread 0
#define saver_thread 1
#define searcher_thread 2
#else
#define loader_thread 0
#define searcher_thread 1
#endif

// The sizes: two shards, each with a key plane of two rows.
#define shards 2
#define plane 2

// The words, by the index member each stands for, per shard.
#if scenario == adding
#define routed_rows 0
#define admitted(shard) (1 + (shard))
#define last_visible_id(shard) (3 + (shard))
#define key(shard, id) (5 + (shard) * 2 + (id))
#else
#define last_visible_id(shard) (0 + (shard))
#define next_visible_id(shard) (2 + (shard))
#define key(shard, id) (4 + (shard) * 2 + (id))
#endif

// The values a key slot carries: unset, or the key of the thread that landed the row.
#define unset 0
#define key_of(thread) (10 + (thread))

#ifdef without_watermark_release
#define watermark_order order_relaxed
#else
#define watermark_order order_release
#endif

byte adders_started;
#if memory == sequential
byte adders_done; // the checker waits on it
#define done() adders_done++
#else
#define done() skip
#endif
byte dispensed[shards]; // each shard's `next_visible_id_`, a ghost where no plan reads it
byte in_flight[shards]; // rows dispensed and not yet published, per shard
bool locators_of_key[shards * plane]; // the reverse plane: which rows were recorded
#define locator(shard, id) ((shard) * plane + (id))

#if scenario == adding
inline dispense(t, shard, id) {
    atomic {
        id = dispensed[shard];
        dispensed[shard]++;
        in_flight[shard]++;
        assert(id < plane) // admission is what keeps an id inside the key plane
    }
}
#else
inline dispense(t, shard, id) {
    atomic {
        read_modify_write(t, next_visible_id(shard), order_acq_rel, id, id + 1);
        in_flight[shard]++
    }
}
#endif

// insert_reserved's tail, for one row of one shard: the key before the turn, the turn strictly
// in dispensed order, the watermark released, then the reverse plane's entry
inline land_row(t, shard, id) {
    // on_assigned fires right after the dispense: one step for the two
    atomic {
        dispense(t, shard, id);
#ifndef without_key_before_watermark
        store(t, key(shard, id), order_relaxed, key_of(t))
#endif
    };
    // the turn's acquire load is what merges the predecessor's publication into this one
    (newest_value(last_visible_id(shard)) == id);
    do
    :: load(t, last_visible_id(shard), order_acquire, watermark);
       if
       :: watermark == id -> break
       :: else
       fi
    od;
    atomic {
        store(t, last_visible_id(shard), watermark_order, id + 1);
        in_flight[shard]--;
        locators_of_key[locator(shard, id)] = true
    }
#ifdef without_key_before_watermark
    ;
    store(t, key(shard, id), order_relaxed, key_of(t))
#endif
}

// shard_t::admit for one row: `fetch_add_if_at_most` against the plane
inline admit(t, shard, before) {
    atomic {
        read_modify_write_if(t, admitted(shard), order_relaxed, before + 1 <= plane, before, before + 1);
        assert(newest_value(admitted(shard)) <= plane)
    }
}

// admission_t's destructor: every shard admitted so far, from shard zero, hands its row back
inline withdraw_up_to(t, count) {
    back = 0;
    do
    :: back == count -> break
    :: else -> add_no_return(t, admitted(back), order_release, 0 - 1); back++
    od
}

#if scenario == adding
active [2] proctype batch_adder() {
    byte me, shard, back;
    int routed, rotation, before, id, watermark;
    atomic { me = adders_started; adders_started++ };
    read_modify_write(me, routed_rows, order_relaxed, routed, routed + shards);
    rotation = routed % shards;
    // admit_: shard by shard ahead of the claims, or into none
    shard = 0;
    do
    :: shard == shards -> break
    :: else ->
        admit(me, shard, before);
        if
        :: before + 1 <= plane -> shard++
        :: else -> withdraw_up_to(me, shard); goto refused
        fi
    od;
    // a claim or the fan-out refused: ForkUnion runs every task or none, so the admissions go back whole
    if
#ifdef without_withdraw
    :: goto refused
#else
    :: withdraw_up_to(me, shards); goto refused
#endif
    :: skip
    fi;
    land_row(me, rotation, id);
    land_row(me, (rotation + 1) % shards, id);
refused:
    done()
}

active proctype single_adder() {
    byte probe, shard;
    int routed, before, id, watermark;
    read_modify_write(single_adder_thread, routed_rows, order_relaxed, routed, routed + 1);
    // the routing tally names the first shard to try; a full one passes the row on to the next
    probe = 0;
    do
    :: probe == shards -> goto refused
    :: else ->
        shard = (routed + probe) % shards;
        admit(single_adder_thread, shard, before);
        if
        :: before + 1 <= plane -> break
        :: else -> probe++
        fi
    od;
    // reserve_traversal or the claim refused: only this shard's row goes back
    if
    :: add_no_return(single_adder_thread, admitted(shard), order_release, 0 - 1); goto refused
    :: skip
    fi;
    land_row(single_adder_thread, shard, id);
#ifdef without_single_consume
    add_no_return(single_adder_thread, admitted(shard), order_release, 0 - 1);
#endif
refused:
    done()
}

#if memory == sequential
active proctype checker() {
    byte shard;
    (adders_done == 3);
    for (shard : 0 .. shards - 1) {
        assert(newest_value(admitted(shard)) == dispensed[shard]);
        assert(!locators_of_key[locator(shard, 0)] || newest_value(key(shard, 0)) != unset);
        assert(!locators_of_key[locator(shard, 1)] || newest_value(key(shard, 1)) != unset)
    }
}
#endif
#endif

#if scenario == saving
active proctype batch_adder() {
    int id, watermark;
    land_row(batch_adder_thread, 0, id);
    land_row(batch_adder_thread, 1, id)
}

active proctype saver() {
    byte shard, id;
    int visible[shards], dispensed_now, again, copy;
    // plan_save per shard: the cut, refused while an insert is mid-flight
    for (shard : 0 .. shards - 1) {
        load(saver_thread, last_visible_id(shard), order_acquire, visible[shard]);
        atomic {
            load(saver_thread, next_visible_id(shard), order_acquire, dispensed_now);
#if memory == sequential
            assert((visible[shard] != dispensed_now) ==
                   (in_flight[shard] != 0 || newest_value(last_visible_id(shard)) != visible[shard]));
#endif
        };
        if
        :: visible[shard] != dispensed_now -> goto contended
        :: else
        fi
    };
    // finish per shard: the fence, the relaxed re-check, then the plain key copy at the shard's base
    for (shard : 0 .. shards - 1) {
        fence_acquire(saver_thread);
        load(saver_thread, last_visible_id(shard), order_relaxed, again);
        if
        :: again != visible[shard] -> goto contended
        :: else
        fi;
        load(saver_thread, next_visible_id(shard), order_relaxed, again);
        if
        :: again != visible[shard] -> goto contended
        :: else
        fi;
        id = 0;
        do
        :: id == visible[shard] -> break
        :: else -> load(saver_thread, key(shard, id), order_relaxed, copy); assert(copy != unset); id++
        od
    };
#ifdef whole_batches
    assert(visible[0] == visible[1]);
#endif
contended:
    skip
}
#endif

#if scenario == loading
active proctype loader() {
    byte shard, row;
    int visible, dispensed_now;
    for (shard : 0 .. shards - 1) {
        // plan_load: the cut, never refused on an idle index
        load(loader_thread, last_visible_id(shard), order_acquire, visible);
        load(loader_thread, next_visible_id(shard), order_acquire, dispensed_now);
        assert(visible == dispensed_now);
        // the keys at the base and the reverse plane, then the publication
#ifndef without_keys_before_finish
        for (row : 0 .. plane - 1) {
            store(loader_thread, key(shard, visible + row), order_relaxed, key_of(loader_thread));
            locators_of_key[locator(shard, visible + row)] = true
        };
#endif
        add_no_return(loader_thread, next_visible_id(shard), order_release, plane);
        store(loader_thread, last_visible_id(shard), watermark_order, visible + plane);
#ifdef without_keys_before_finish
        for (row : 0 .. plane - 1) {
            store(loader_thread, key(shard, visible + row), order_relaxed, key_of(loader_thread));
            locators_of_key[locator(shard, visible + row)] = true
        };
#endif
    };
    landed(loader_thread)
}
#endif

// the search merge: every id below the watermark it acquired resolves through the key plane
active proctype searcher() {
    byte shard, id;
    int watermark, seen;
    for (shard : 0 .. shards - 1) {
        load(searcher_thread, last_visible_id(shard), order_acquire, watermark);
        id = 0;
        do
        :: id == watermark -> break
        :: else -> load(searcher_thread, key(shard, id), order_relaxed, seen); assert(seen != unset); id++
        od
    }
}
