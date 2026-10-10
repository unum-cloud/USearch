/**
 *  The sharded dense index from `include/usearch/dense/ram_indexes.hpp`: each row claimed from
 *  its shard's own capacity counter, a refusal past the claim releasing it with a release
 *  subtract, each shard's key column covered to what the shard hosts after every claim, each
 *  landed row's key stored before its shard's watermark turn, and the readers that rest on the
 *  watermark - the search merge reading `keys[node]`, the monolith save taking one `plan_save`
 *  per shard, and the load path landing keys under one `finish`.
 *
 *  `-Dscenario=adding`, the default: two shards of two rows each, two batch adders each bringing
 *  one row per shard, one single-row adder probing the shards from its routing base, and a
 *  searcher; the dispenser is a ghost, a row's publication is its watermark turn. The batch
 *  adder claims shard by shard, releases every shard claimed so far on a refusal, and may have
 *  its fan-out refused after the claims, which releases them all. A checker outside the memory
 *  model compares each shard's claim word with its dispenser once every adder is done.
 *  `-Dscenario=covering`: one shard whose rows and key chunks hold one row apiece, three adders
 *  claiming a row each, covering the key column to the shard's capacity, then dispensing, and a
 *  searcher. `-Dscenario=saving`: one batch adder, a saver running `plan_save` per shard,
 *  `finish`, then the plain key copy of `keys[0..visible)`, and a searcher; the dispenser is a
 *  word, since the watermark compares it. `-Dscenario=loading`: a loader running `plan_load` per
 *  shard, the keys at the base, then `finish`, against a searcher on the still-empty index.
 *  `-Dscenario=committed`: one batch's commit landing a row on each shard, its ranges dispensed
 *  in one step as the dispensing mutex does, every shard's turn awaited, then its watermark
 *  stores bracketed by the index's commit sequence word, a release add making it odd before them
 *  and another making it even after; and a searcher that acquires the word, snapshots both
 *  shards' watermarks, re-reads the word, and retries while it is odd or changed.
 *  `-Dconcurrent_commits` runs a second such commit beside the first, and `-Dcounted_sequence`
 *  counts the commits in flight in the word's low part rather than in its parity, as the header
 *  does: a commit adds one before its stores and `sequence_unit - 1` after, and the searcher
 *  retries while the low part is not zero.
 *
 *  The shard's claim is `claim_capacity` seen from outside: it grows the shard on demand and
 *  refuses only past the rows one shard's ids reach, the model's row limit.
 *  `dense_ram_index_capacity.pml` covers the counter itself: the subtract, the election, the
 *  deficit handed back and waited out.
 *
 *  Invariants:
 *  - the row limit: a dispensed id is below its shard's row limit, and once every adder is done
 *    each shard's claim word equals its dispenser. `-Dwithout_release` skips the release after a
 *    refused fan-out, and the claim word stays above the dispenser;
 *  - the cover: every key slot an insert writes or a searcher reads lies in a chunk the key
 *    column published. The id an adder dispenses may lie past its own claim, in a chunk only a
 *    later claimer covered, but that claimer dispensed first, and the dispenser's acquire-release
 *    carries its cover along: `-Dwithout_cover` skips the cover, `-Dwithout_dispense_acq_rel`
 *    makes the dispense relaxed, and each writes a key into a chunk it never saw published;
 *  - publication: every id below a watermark a reader acquired has its key. The key lands before
 *    the watermark turn; `-Dwithout_key_before_watermark` moves it after, and a searcher reads an
 *    unset key under every model; `-Dwithout_watermark_release` drops the release from the
 *    watermark store, and the searcher reads it under the view model;
 *  - the watermark: a settled `plan_save` copies no unset key below its watermark, and,
 *    sequentially, a plan is refused exactly when a row is between its dispense and its
 *    publication, or was published between the plan's two loads. `-Dwhole_batches` asks the
 *    watermark to capture whole batches, shard zero and shard one together, and fails under
 *    every model: the watermark is per shard;
 *  - the load path: keys land before `finish` publishes the rows. `-Dwithout_keys_before_finish`
 *    moves them after, and the searcher reads an unset key under the sequential and view models;
 *  - the commit: a snapshot the searcher accepts holds each commit on both shards or on neither.
 *    `-Dwithout_commit_sequence` leaves the word out, and the searcher takes a snapshot between
 *    the commit's two watermark stores. `-Dconcurrent_commits` fails with the parity: one
 *    commit's add makes the word odd and the other's makes it even again while both are inside
 *    their brackets, so a searcher accepts a snapshot holding one shard's part.
 *    `-Dcounted_sequence` passes there, since the word returns to a multiple of the unit only
 *    once both are done. `-Dwithout_ordered_dispense` lets each shard dispense on its own, so two
 *    commits' ranges cross between the shards, each waits for a turn only the other's store
 *    gives, and Spin reports the deadlock.
 *
 *  `-Dwithout_claim_consume` drops the `claim.consume()` the shard's `insert` makes once the
 *  row is dispensed against: the landed row is released and the claim word falls below the
 *  dispenser; with two batches behind it the shard claims one row more than it holds, and the
 *  next dispensed id passes the row limit.
 */
#include "weak_memory.pml"

// The scenarios: the adders against the shards, the key column's cover, the monolith save's
// watermark, the load path, or a commit across the shards.
#define adding 1
#define covering 2
#define saving 3
#define loading 4
#define committed 5
#ifndef scenario
#define scenario adding
#endif

// The threads, by role, apart from the processes that play them; the batch adders name themselves as they start.
#if scenario == adding
#define single_adder_thread 2
#define searcher_thread 3
#elif scenario == covering
#define searcher_thread 3
#elif scenario == saving
#define batch_adder_thread 0
#define saver_thread 1
#define searcher_thread 2
#elif scenario == committed
#define searcher_thread 2
#else
#define loader_thread 0
#define searcher_thread 1
#endif

// The sizes: the shards, and the rows one shard's ids reach - its row limit.
#if scenario == covering
#define shards 1
#define row_limit 3
#else
#define shards 2
#define row_limit 2
#endif

// The words, by the index member each stands for, per shard.
#if scenario == adding
#define routed_rows 0
#define claimed(shard) (1 + (shard))
#define last_visible_id(shard) (3 + (shard))
#define key(shard, id) (5 + (shard) * 2 + (id))
#elif scenario == covering
#define last_visible_id(shard) 0
#define next_dispensed_id(shard) 1
#define key(shard, id) (2 + (id))
#define claimed_rows 5
#define key_chunk(id) (6 + (id))
#elif scenario == committed
#define commit_sequence 0
#define last_visible_id(shard) (1 + (shard))
#define key(shard, id) (3 + (shard) * 2 + (id))
#else
#define last_visible_id(shard) (0 + (shard))
#define next_dispensed_id(shard) (2 + (shard))
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
byte dispensed[shards]; // each shard's `next_dispensed_id_`, a ghost where no plan reads it
byte in_flight[shards]; // rows dispensed and not yet published, per shard
bool locators_of_key[shards * row_limit]; // which rows the locator map recorded
#define locator(shard, id) ((shard) * row_limit + (id))

#if scenario == adding || scenario == committed
inline dispense(t, shard, id) {
    atomic {
        id = dispensed[shard];
        dispensed[shard]++;
        in_flight[shard]++;
        assert(id < row_limit) // the claim is what keeps an id inside the row limit
    }
}
#else
#ifdef without_dispense_acq_rel
#define dispense_order order_relaxed
#else
#define dispense_order order_acq_rel
#endif
inline dispense(t, shard, id) {
    atomic {
        read_modify_write(t, next_dispensed_id(shard), dispense_order, id, id + 1);
        in_flight[shard]++
    }
}
#endif

// `insert`'s tail, for one row of one shard: the key before the turn, the turn strictly
// in dispensed order, the watermark released, then the locator map's entry
inline land_row(t, shard, id) {
    // on_assigned fires right after the dispense: one step for the two
    atomic {
        dispense(t, shard, id);
#if scenario == covering
        // the slot's chunk, loaded relaxed as `chunked_column::operator[]` does
        load(t, key_chunk(id), order_relaxed, chunk);
        assert(chunk != 0);
#endif
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

// the shard's own row claim up to its row limit - `claim_capacity` seen from outside: one
// unconditional add whose value says what happened - a fit; the crossing, which elects this
// claimer and the row limit then refuses; or a peer's refusal still in flight, handed back and
// waited out. `before + 1 <= row_limit` afterwards names a fit
inline claim(t, shard, before) {
    do
    :: read_modify_write(t, claimed(shard), order_acquire, before, before + 1);
       if
       :: before + 1 <= row_limit -> break
       :: before == row_limit -> add_no_return(t, claimed(shard), order_release, 0 - 1); break
       :: else -> add_no_return(t, claimed(shard), order_release, 0 - 1); (newest_value(claimed(shard)) <= row_limit)
       fi
    od
}

// the claims' destructors: every shard claimed so far, from shard zero, releases its row
inline release_up_to(t, count) {
    back = 0;
    do
    :: back == count -> break
    :: else -> add_no_return(t, claimed(back), order_release, 0 - 1); back++
    od
}

#if scenario == adding
active [2] proctype batch_adder() {
    byte me, shard, back;
    int routed, rotation, before, id, watermark;
    atomic { me = adders_started; adders_started++ };
    read_modify_write(me, routed_rows, order_relaxed, routed, routed + shards);
    rotation = routed % shards;
    // claim: shard by shard, or into none
    shard = 0;
    do
    :: shard == shards -> break
    :: else ->
        claim(me, shard, before);
        if
        :: before + 1 <= row_limit -> shard++
        :: else -> release_up_to(me, shard); goto refused
        fi
    od;
    // the fan-out refused: ForkUnion runs every task or none, so the claims go back whole
    if
#ifdef without_release
    :: goto refused
#else
    :: release_up_to(me, shards); goto refused
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
        claim(single_adder_thread, shard, before);
        if
        :: before + 1 <= row_limit -> break
        :: else -> probe++
        fi
    od;
    // reserve_traversal refused: only this shard's row goes back
    if
    :: add_no_return(single_adder_thread, claimed(shard), order_release, 0 - 1); goto refused
    :: skip
    fi;
    land_row(single_adder_thread, shard, id);
#ifdef without_claim_consume
    add_no_return(single_adder_thread, claimed(shard), order_release, 0 - 1);
#endif
refused:
    done()
}

#if memory == sequential
active proctype checker() {
    byte shard;
    (adders_done == 3);
    for (shard : 0 .. shards - 1) {
        assert(newest_value(claimed(shard)) == dispensed[shard]);
        assert(!locators_of_key[locator(shard, 0)] || newest_value(key(shard, 0)) != unset);
        assert(!locators_of_key[locator(shard, 1)] || newest_value(key(shard, 1)) != unset)
    }
}
#endif
#endif

#if scenario == covering
// `chunked_column::cover`: every chunk below @p rows, published with a release unless a peer did
inline cover(t, rows) {
    covered = 0;
    do
    :: covered == rows -> break
    :: else ->
        load(t, key_chunk(covered), order_acquire, chunk);
        if
        :: chunk == 0 ->
            chunk = 0;
            compare_exchange(t, key_chunk(covered), order_release, chunk, 1, published)
        :: else
        fi;
        covered++
    od
}

// the single-row insert: the claim, the cover to what the shard hosts after it, then the row
active [3] proctype adder() {
    byte me;
    int first, hosted, covered, chunk, id, watermark;
    bool published;
    atomic { me = adders_started; adders_started++ };
    read_modify_write(me, claimed_rows, order_acquire, first, first + 1);
    load(me, claimed_rows, order_acquire, hosted);
#ifndef without_cover
    cover(me, hosted);
#endif
    land_row(me, 0, id)
}
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
    // plan_save per shard: the watermark, refused while an insert is mid-flight
    for (shard : 0 .. shards - 1) {
        load(saver_thread, last_visible_id(shard), order_acquire, visible[shard]);
        atomic {
            load(saver_thread, next_dispensed_id(shard), order_acquire, dispensed_now);
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
        load(saver_thread, next_dispensed_id(shard), order_relaxed, again);
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
        // plan_load: the watermark, never refused on an idle index
        load(loader_thread, last_visible_id(shard), order_acquire, visible);
        load(loader_thread, next_dispensed_id(shard), order_acquire, dispensed_now);
        assert(visible == dispensed_now);
        // the keys at the base and the locator map, then the publication
#ifndef without_keys_before_finish
        for (row : 0 .. row_limit - 1) {
            store(loader_thread, key(shard, visible + row), order_relaxed, key_of(loader_thread));
            locators_of_key[locator(shard, visible + row)] = true
        };
#endif
        add_no_return(loader_thread, next_dispensed_id(shard), order_release, row_limit);
        store(loader_thread, last_visible_id(shard), watermark_order, visible + row_limit);
#ifdef without_keys_before_finish
        for (row : 0 .. row_limit - 1) {
            store(loader_thread, key(shard, visible + row), order_relaxed, key_of(loader_thread));
            locators_of_key[locator(shard, visible + row)] = true
        };
#endif
    };
    landed(loader_thread)
}
#endif

#if scenario == committed
#ifdef concurrent_commits
#define commits 2
#else
#define commits 1
#endif
// the word's step after the turns: the parity's, or the counted word's carry into its high part
#ifdef counted_sequence
#define sequence_unit 4
#else
#define sequence_unit 2
#endif
byte commits_started;
bool commit_dispensed[commits];
byte commit_part[commits * shards]; // the ID each commit landed on each shard
#define part(commit, shard) commit_part[(commit) * shards + (shard)]

// a batch's commit: a row on each shard, both keys before the turns, every turn awaited outside
// the bracket of the commit sequence word, and the watermark stores inside it
active [commits] proctype committer() {
    byte me, shard;
    int id, watermark, sequence;
    atomic { me = commits_started; commits_started++ };
#ifdef without_ordered_dispense
    // each shard dispenses on its own, so two commits' IDs may cross between the shards
    for (shard : 0 .. shards - 1) {
        atomic { dispense(me, shard, id); part(me, shard) = id }
    };
#else
    // the dispensing mutex: every shard's range in one step, so no two commits cross between the shards
    atomic {
        for (shard : 0 .. shards - 1) {
            dispense(me, shard, id);
            part(me, shard) = id
        }
    };
#endif
    commit_dispensed[me] = true;
    for (shard : 0 .. shards - 1) {
        store(me, key(shard, part(me, shard)), order_relaxed, key_of(me))
    };
    // the waits hold no bracket open: no later range on a shard can take its turn first
    for (shard : 0 .. shards - 1) {
        (newest_value(last_visible_id(shard)) == part(me, shard));
        do
        :: load(me, last_visible_id(shard), order_acquire, watermark);
           if
           :: watermark == part(me, shard) -> break
           :: else
           fi
        od
    };
#ifndef without_commit_sequence
    read_modify_write(me, commit_sequence, order_release, sequence, sequence + 1);
#endif
    for (shard : 0 .. shards - 1) {
        atomic {
            store(me, last_visible_id(shard), watermark_order, part(me, shard) + 1);
            in_flight[shard]--
        }
    };
#ifndef without_commit_sequence
    read_modify_write(me, commit_sequence, order_release, sequence, sequence + sequence_unit - 1)
#endif
}

// a search over both shards: the word acquired, the shards' watermarks snapshotted, the word
// re-read, and a retry while it is odd or moved; an accepted snapshot holds each commit whole
active proctype searcher() {
    byte shard, id, commit;
    int before, after, watermarks[shards], seen;
retry:
    load(searcher_thread, commit_sequence, order_acquire, before);
    for (shard : 0 .. shards - 1) {
        load(searcher_thread, last_visible_id(shard), order_acquire, watermarks[shard])
    };
    load(searcher_thread, commit_sequence, order_relaxed, after);
#ifndef without_commit_sequence
    if
    :: before % sequence_unit != 0 || after != before ->
        (newest_value(commit_sequence) % sequence_unit == 0);
        goto retry
    :: else
    fi;
#endif
    for (commit : 0 .. commits - 1) {
        assert(!commit_dispensed[commit] ||
               (watermarks[0] > part(commit, 0)) == (watermarks[1] > part(commit, 1)))
    };
    for (shard : 0 .. shards - 1) {
        id = 0;
        do
        :: id == watermarks[shard] -> break
        :: else -> load(searcher_thread, key(shard, id), order_relaxed, seen); assert(seen != unset); id++
        od
    }
}
#else
// the search merge: every id below the watermark it acquired resolves through the key column
active proctype searcher() {
    byte shard, id;
    int watermark, seen;
    for (shard : 0 .. shards - 1) {
        load(searcher_thread, last_visible_id(shard), order_acquire, watermark);
        id = 0;
        do
        :: id == watermark -> break
        :: else ->
#if scenario == covering
            load(searcher_thread, key_chunk(id), order_relaxed, seen);
            assert(seen != 0);
#endif
            load(searcher_thread, key(shard, id), order_relaxed, seen);
            assert(seen != unset);
            id++
        od
    }
}
#endif
