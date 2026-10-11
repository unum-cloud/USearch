/**
 *  `dense_vram_nvidia_indexes` from `include/usearch/dense/vram_nvidia_indexes.cuh`: the sharded CUDA index,
 *  whose batches spread rows over the shards, each shard growing as it claims, and write every key
 *  before the shard's claim; whose one mutation gate answers `contended_k` to a second batch, and
 *  so serializes the key columns' covers too; whose batch turns every shard's watermark inside one
 *  commit of a sequence word, so a reader spanning the shards snapshots each batch whole or not at
 *  all; and whose load lands under the same gate and in the same key-then-publication order.
 *
 *  Two shards of two rows each, one per claimer. A mutator takes the gate with an acquire
 *  exchange, then per shard reads the shard's claim as the first node, stores the key relaxed, and
 *  runs the shard's insert claiming the next row. A first node that the shard's own claim does not
 *  confirm is a claimer outside the gate, and the batch stops, publishing nothing and leaving a
 *  kept failure on every shard it claimed on. A batch that claimed on both shards commits,
 *  `insert_plan_t::finish`: a release add of one to the sequence word, each shard's turn, a wait
 *  until its watermark reaches the part's first node or the shard keeps a failure and a release
 *  store past it, then a release add of the unit less one. The searcher snapshots: an acquire of
 *  the word, a wait while a commit is in flight, an acquire of each shard's watermark, a relaxed
 *  re-read of the word and a retry when it moved; then it reads every key below each snapshotted
 *  watermark.
 *
 *  Every key a searcher reads below a watermark it snapshotted is the key of the thread that
 *  claimed the row, never unset and never another's, and a snapshot it accepts holds each
 *  committed batch on both shards or on neither. Every scenario beside this file defines
 *  `mutators`, the batches it runs as threads `0` and up.
 */
#include "../weak_memory.pml"

// The words: the gate, each shard's watermark, `commit_sequence_`, and each shard's keys. A shard's
// claim is read only under the gate or the shard's own mutex and publishes nothing a searcher
// reads, since the watermark bounds every walk, so it is a plain counter here.
#define mutating 0
#define last_visible_id(shard) (1 + (shard))
#define sequence_word 3
#define key(shard, id) (4 + (shard) * plane + (id))

// The sizes and values: two shards of two rows, one per claimer, a batch of one row per shard, the
// keys, and the sequence word's unit, past the commits in flight it counts. A load runs beside one
// batch, so no shard takes a third row.
#define shards 2
#define plane 2
#define unset 0
#define key_of(t) (10 + (t))
#define sequence_unit 4

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef gate
#define gate true
#endif
#ifndef key_before_turn
#define key_before_turn true
#endif
#ifndef watermark_order
#define watermark_order order_release
#endif
#ifndef commit_sequence
#define commit_sequence true
#endif
#ifndef load_gate
#define load_gate true
#endif
#ifndef load_key_before_finish
#define load_key_before_finish true
#endif

byte claims[shards];          // per shard, its claim: `claimed_rows_`
bool kept[shards];            // per shard, a failure kept by a batch that stopped with rows claimed there
byte owner[shards * plane];   // one past the thread that claimed the row; zero while unclaimed
#define owner_of(shard, id) owner[(shard) * plane + (id)]
bool committing[mutators];    // per mutator, whether its batch reached its commit
byte part[mutators * shards]; // per mutator and shard, the row its batch landed at
#define part_of(t, shard) part[(t) * shards + (shard)]

// the gate: an acquire exchange, and a contended answer when it was held: `plan_insert` and `gate_load_`
inline take_gate(t, seen) { read_modify_write(t, mutating, order_acquire, seen, 1) }
inline drop_gate(t) { store(t, mutating, order_release, 0) }

// the shard's insert claiming the next row: `dense_vram_nvidia_index::insert`
inline claim_row(t, shard, first_id) {
    atomic {
        first_id = claims[shard];
        claims[shard]++;
        owner_of(shard, first_id) = (t) + 1
    }
}

// the shard's turn from `first` past one row, once every row claimed before has turned, or never
// once the shard keeps a failure, since the rows before never turn: `publish_`
inline take_turn(t, shard, first, watermark) {
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

// a batch: the gate, per shard the first node, the key and the claim, then the commit across the
// shards inside the sequence word: `insert_plan_t::insert` and `insert_plan_t::finish`
inline insert_batch(t) {
    byte shard;
    int seen, first_node, first_id, watermark, sequence;
    if
    :: gate ->
        take_gate(t, seen);
        if :: seen == 1 -> goto contended :: else fi
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        first_node = claims[shard];
        assert(first_node < plane);
        if
        :: key_before_turn -> store(t, key(shard, first_node), order_relaxed, key_of(t))
        :: else
        fi;
        claim_row(t, shard, first_id);
        part_of(t, shard) = first_node;
        // a first node the shard did not confirm is a claimer outside the gate: the batch publishes
        // nothing, and every shard it claimed on keeps the failure
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
    committing[t] = true;
    read_modify_write(t, sequence_word, order_release, sequence, sequence + 1);
    for (shard : 0 .. shards - 1) {
        take_turn(t, shard, part_of(t, shard), watermark)
    };
    read_modify_write(t, sequence_word, order_release, sequence, sequence + sequence_unit - 1);
    if
    :: !key_before_turn ->
        for (shard : 0 .. shards - 1) {
            store(t, key(shard, part_of(t, shard)), order_relaxed, key_of(t))
        }
    :: else
    fi;
stopped:
    if
    :: gate -> drop_gate(t)
    :: else
    fi;
contended:
    skip
}

// a search: the snapshot of every shard's watermark between two equal reads of the sequence word
// with no commit in flight, then every key below each watermark
inline search(t) {
    byte shard, id, batch;
    int before, after, seen, watermarks[shards];
retry:
    load(t, sequence_word, order_acquire, before);
    if
    :: commit_sequence && before % sequence_unit != 0 ->
        (newest_value(sequence_word) % sequence_unit == 0);
        goto retry
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermarks[shard])
    };
    load(t, sequence_word, order_relaxed, after);
    if
    :: commit_sequence && after != before -> goto retry
    :: else
    fi;
    for (batch : 0 .. mutators - 1) {
        assert(!committing[batch] ||
               (watermarks[0] > part_of(batch, 0)) == (watermarks[1] > part_of(batch, 1)))
    };
    for (shard : 0 .. shards - 1) {
        id = 0;
        do
        :: id < watermarks[shard] ->
            load(t, key(shard, id), order_relaxed, seen);
            assert(owner_of(shard, id) != 0 && seen == key_of(owner_of(shard, id) - 1));
            id++
        :: else -> break
        od
    }
}
