/**
 *  `dense_vram_indexes` from `include/usearch/dense/vram_indexes.cuh`: the sharded CUDA index,
 *  whose batches spread rows over the shards, each shard growing as it claims, and write every
 *  key before the shard's claim; whose one mutation gate answers `contended_k` to a second
 *  batch, and so serializes the key columns' covers too; whose batch turns every shard's
 *  watermark inside one commit of a sequence word, so a reader spanning the shards snapshots each
 *  batch whole or not at all; and whose load lands under the same gate and in the same
 *  key-then-publication order.
 *
 *  Two mutators, one searcher, and under `-Dscenario=loading` one loader beside one mutator, over
 *  two shards of two rows each, one per claimer. A mutator takes the gate with an acquire
 *  exchange, then per shard reads the shard's claim as the first node, stores the key relaxed,
 *  and runs the shard's insert claiming the next row. A first node that the shard's own claim
 *  does not confirm is a claimer outside the gate, and the batch stops, publishing nothing and
 *  leaving a kept failure on every shard it claimed on. A batch that claimed on both shards
 *  commits, `insert_plan_t::finish`: a release add of one to the sequence word, each shard's
 *  turn, a wait until its watermark reaches the part's first node or the shard keeps a failure
 *  and a release store past it, then a release add of the unit less one. The searcher snapshots:
 *  an acquire of the word, a wait while a commit is in flight, an acquire of each shard's
 *  watermark, a relaxed re-read of the word and a retry when it moved; then it reads every key
 *  below each snapshotted watermark. The loader takes the gate, checks under it that no row is
 *  visible, writes each shard's key at the append base, and runs the shard plan's `finish`: the
 *  claim, then the turn.
 *
 *  Invariants:
 *  - ownership: every key a searcher reads below a watermark it snapshotted is the key of the
 *    thread that claimed the row, never unset and never another's. `-Dwithout_gate` lets the
 *    mutators interleave their claims, and one writes its key over a row the other's claim
 *    confirmed; `-Dwithout_key_before_turn` writes the keys after the commit and the searcher
 *    reads an unset key under every memory model; `-Dscenario=loading -Dwithout_load_gate` runs
 *    the load outside the gate, the header before the fix, and a batch's key lands on a row the
 *    load claims; `-Dwithout_load_key_before_finish` publishes the loaded shard before its key;
 *  - wholeness: a snapshot the searcher accepts holds each committed batch on both shards or on
 *    neither. `-Dwithout_commit_sequence` reads the watermarks unbracketed, and a snapshot taken
 *    between a commit's two turns holds one shard's part. `-Dwithout_watermark_release` turns
 *    relaxed, and under views the same snapshot passes the re-read: the turn's release is what
 *    carries the commit's opening add to it, besides the keys.
 */
#include "weak_memory.pml"

// The knob's values are integers, so a typo fails the range check below.
#define batches 1
#define loading 2
#ifndef scenario
#define scenario batches
#endif
#if scenario < batches || scenario > loading
#error "scenario is batches or loading"
#endif

// The threads, by role, apart from the processes that play them; the mutators name themselves.
#define searcher_thread 2
#define loader_thread 3

// The words: the gate, each shard's watermark, the commit sequence word, and each shard's keys. A
// shard's claim is read only under the gate or the shard's own mutex and publishes nothing a
// searcher reads, since the watermark bounds every walk, so it is a plain counter here.
#define mutating 0
#define last_visible_id(shard) (1 + (shard))
#define commit_sequence 3
#define key(shard, id) (4 + (shard) * plane + (id))

// The sizes and values: two shards of two rows, one per claimer, a batch of one row per shard, the
// keys, and the sequence word's unit, past the commits in flight it counts. A load runs beside one
// batch, so no shard takes a third row.
#define shards 2
#define plane 2
#define unset 0
#define key_of(t) (10 + (t))
#define sequence_unit 4
#if scenario == loading
#define mutators 1
#else
#define mutators 2
#endif

#ifdef without_watermark_release
#define watermark_order order_relaxed
#else
#define watermark_order order_release
#endif

byte mutators_started;
byte claims[shards];        // per shard, its claim: `claimed_rows_`
bool kept[shards];          // per shard, a failure kept by a batch that stopped with rows claimed there
byte owner[shards * plane]; // one past the thread that claimed the row; zero while unclaimed
#define owner_of(shard, id) owner[(shard) * plane + (id)]
bool committing[mutators];  // per mutator, whether its batch reached its commit
byte part[mutators * shards]; // per mutator and shard, the row its batch landed at
#define part_of(t, shard) part[(t) * shards + (shard)]

// the gate: an acquire exchange, and a contended answer when it was held: `plan_insert` and `gate_load_`
inline take_gate(t, seen) { read_modify_write(t, mutating, order_acquire, seen, 1) }
inline drop_gate(t) { store(t, mutating, order_release, 0) }

// the shard's insert claiming the next row: `dense_vram_index::insert`
inline claim_row(t, shard, first_id) {
    atomic {
        first_id = claims[shard];
        claims[shard]++;
        owner_of(shard, first_id) = (t) + 1
    }
}

// the shard's turn from `first` past one row, once every row claimed before has turned, or never once
// the shard keeps a failure, since the rows before never turn: `publish_`
inline take_turn(t, shard, first) {
    (newest_value(last_visible_id(shard)) == first || kept[shard]);
    do
    :: load(t, last_visible_id(shard), order_acquire, watermark);
       if
       :: watermark == first -> store(t, last_visible_id(shard), watermark_order, first + 1); break
       :: watermark != first && kept[shard] -> break
       :: else
       fi
    od
}

active [mutators] proctype mutator() {
    byte me, shard;
    int seen, first_node, first_id, watermark, sequence;
    atomic { me = mutators_started; mutators_started++ };
#ifndef without_gate
    take_gate(me, seen);
    if :: seen == 1 -> goto contended :: else fi;
#endif
    for (shard : 0 .. shards - 1) {
        // the first node from the shard's claim, then the key, then the shard's insert: `insert_plan_t::insert`
        first_node = claims[shard];
        assert(first_node < plane);
#ifndef without_key_before_turn
        store(me, key(shard, first_node), order_relaxed, key_of(me));
#endif
        claim_row(me, shard, first_id);
        part_of(me, shard) = first_node;
        // a first node the shard did not confirm is a claimer outside the gate: the batch publishes nothing,
        // and every shard it claimed on keeps the failure
        if
        :: first_id != first_node ->
            atomic {
                kept[shard] = true;
                kept[0] = true
            };
            goto stopped
        :: else
        fi
    };
    // the commit: every shard's turn bracketed by the sequence word, `insert_plan_t::finish`
    committing[me] = true;
    read_modify_write(me, commit_sequence, order_release, sequence, sequence + 1);
    for (shard : 0 .. shards - 1) {
        take_turn(me, shard, part_of(me, shard))
    };
    read_modify_write(me, commit_sequence, order_release, sequence, sequence + sequence_unit - 1);
#ifdef without_key_before_turn
    for (shard : 0 .. shards - 1) {
        store(me, key(shard, part_of(me, shard)), order_relaxed, key_of(me))
    };
#endif
stopped:
#ifndef without_gate
    drop_gate(me);
#endif
contended:
    skip
}

// a search: the snapshot of every shard's watermark between two equal reads of the sequence word with no
// commit in flight, then every key below each watermark
active proctype searcher() {
    byte shard, id, batch;
    int before, after, seen, watermarks[shards];
retry:
    load(searcher_thread, commit_sequence, order_acquire, before);
#ifndef without_commit_sequence
    if
    :: before % sequence_unit != 0 ->
        (newest_value(commit_sequence) % sequence_unit == 0);
        goto retry
    :: else
    fi;
#endif
    for (shard : 0 .. shards - 1) {
        load(searcher_thread, last_visible_id(shard), order_acquire, watermarks[shard])
    };
    load(searcher_thread, commit_sequence, order_relaxed, after);
#ifndef without_commit_sequence
    if
    :: after != before -> goto retry
    :: else
    fi;
#endif
    for (batch : 0 .. mutators - 1) {
        assert(!committing[batch] ||
               (watermarks[0] > part_of(batch, 0)) == (watermarks[1] > part_of(batch, 1)))
    };
    for (shard : 0 .. shards - 1) {
        id = 0;
        do
        :: id < watermarks[shard] ->
            load(searcher_thread, key(shard, id), order_relaxed, seen);
            assert(owner_of(shard, id) != 0 && seen == key_of(owner_of(shard, id) - 1));
            id++
        :: else -> break
        od
    }
}

#if scenario == loading
active proctype loader() {
    byte shard;
    int seen, first_id, total, watermark;
#ifndef without_load_gate
    take_gate(loader_thread, seen);
    if :: seen == 1 -> goto contended :: else fi;
#endif
    // empty under the hold, as a load requires: `gate_load_` reads `size()`
    load(loader_thread, last_visible_id(0), order_acquire, seen);
    load(loader_thread, last_visible_id(1), order_acquire, total);
    if :: seen + total != 0 -> goto refused :: else fi;
    // each shard: the key at the append base, then the plan's finish claiming and turning: `load_plan_t::finish`
    for (shard : 0 .. shards - 1) {
#ifdef without_load_key_before_finish
        claim_row(loader_thread, shard, first_id);
        take_turn(loader_thread, shard, first_id);
        store(loader_thread, key(shard, first_id), order_relaxed, key_of(loader_thread));
#else
        first_id = claims[shard]; // plan_load's base, read under the shard's mutex
        store(loader_thread, key(shard, first_id), order_relaxed, key_of(loader_thread));
        claim_row(loader_thread, shard, first_id);
        take_turn(loader_thread, shard, first_id);
#endif
    };
refused:
#ifndef without_load_gate
    drop_gate(loader_thread);
#endif
contended:
    skip
}
#endif
