/**
 *  @file verification/dense_vram_indexes/protocol.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief `dense::vram_nvidia_indexes` from `include/usearch/dense/vram_nvidia_indexes.cuh`: the
 *      sharded CUDA index.
 *
 *  Batches claim each shard's share in shard order by one compare-exchange of its claim word from
 *  the watermark, so a shard holds one batch in flight and its map one writer. A batch uploads its
 *  keys on the launch stream ahead of the launch, files them after the drain, and turns every
 *  watermark inside one commit of the sequence word; a failed drain keeps its failure on every shard
 *  instead. Loads and saves close every shard first, and reopen them the last first.
 *
 *  - Every key a searcher reads is its claimer's, and the map holds every visible row once.
 *  - A snapshot it accepts holds each committed batch on both shards or on neither.
 */
#include <weak_memory.pml>

// The words: each shard's watermark, `commit_sequence_`, each shard's keys, host and device as one
// word the launch stream orders, and each shard's map. A shard's claim, its closed state and its
// kept failure publish nothing a searcher reads, since the watermark bounds every walk, so they are
// ghosts here, each step on them one compare-exchange.
#define last_visible_id(shard) (shard)
#define sequence_word 2
#define key(shard, id) (3 + (shard) * shard_rows + (id))
#define map_slot(shard, position) (7 + (shard) * shard_rows + (position))

// The sizes and values: two shards of two rows, one per claimer, a batch of one row per shard, the
// keys, and the sequence word's unit, past the commits in flight it counts.
#define shards 2
#define shard_rows 2
#define unset 0
#define key_of(t) (10 + (t))
#define sequence_unit 4

// A map slot is the key beside its node in one word, published with one release.
#define empty 0
#define filed(key, id) ((key) * shard_rows + (id) + 1)
#define key_in(slot) (((slot) - 1) / shard_rows)
#define node_in(slot) (((slot) - 1) % shard_rows)

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef quiescent_claim
#define quiescent_claim true
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
#ifndef load_closes
#define load_closes true
#endif
#ifndef load_key_before_finish
#define load_key_before_finish true
#endif
#ifndef save_closes
#define save_closes true
#endif
#ifndef claim_order
#define claim_order order_acquire
#endif
#ifndef keys_after_drain
#define keys_after_drain true
#endif
#ifndef keys_once_covered
#define keys_once_covered true
#endif

byte claims[shards];          // per shard, its claim: `claimed_rows_`
bool closed[shards];          // per shard, whether its dispenser holds the closed state
byte owner[shards * shard_rows];   // one past the thread that took the row; zero while untaken
#define owner_of(shard, id) owner[(shard) * shard_rows + (id)]
bool keyed[shards * shard_rows];   // per row, whether the shard's map files it
byte done;                    // the threads finished
bool committing[mutators];    // per mutator, whether its batch reached its commit
byte part[mutators * shards]; // per mutator and shard, the row its batch landed at
#define part_of(t, shard) part[(t) * shards + (shard)]

// a row of `shard` taken by thread `t`, never one another thread took
inline take_row(t, shard, id) {
    assert(owner_of(shard, id) == 0);
    owner_of(shard, id) = (t) + 1
}

// the shard's claim of one row: the watermark acquired, which carries the map's filings to the
// claim's holder, then the compare-exchange from it to one past it, refused while a batch is in
// flight there or the dispenser is closed
inline claim_row(t, shard, first_id, claimed, visible) {
    load(t, last_visible_id(shard), claim_order, visible);
    atomic {
        if
        :: !closed[shard] && (claims[shard] == visible || !quiescent_claim) ->
            first_id = claims[shard];
            claims[shard]++;
            take_row(t, shard, first_id);
            claimed = true
        :: else -> claimed = false
        fi
    }
}

// a share given back by a refused plan: its row untaken, the claim back at its watermark
inline unclaim_row(shard, first_id) {
    atomic {
        owner_of(shard, first_id) = 0;
        claims[shard]--
    }
}

// the shard's turn from `first` past one row, once every row claimed before has turned: `publish_`
inline take_turn(t, shard, first, watermark) {
    do
    :: load(t, last_visible_id(shard), order_acquire, watermark);
        if
        :: watermark == first -> store(t, last_visible_id(shard), watermark_order, first + 1); break
        :: else -> (newest_value(last_visible_id(shard)) != watermark)
        fi
    od
}

// the first `count` shards reopened, the last first, each at its watermark
inline reopen_shards(count) {
    do
    :: count == 0 -> break
    :: else ->
        atomic {
            count--;
            claims[count] = newest_value(last_visible_id(count));
            closed[count] = false
        }
    od
}

// every shard's dispenser closed in shard order, each by one compare-exchange from the watermark
// acquired, so no batch is in flight there; a refusal reopens the ones closed before it, the last
// first, and leaves `count` below `shards`
inline close_shards(t, count, visible) {
    count = 0;
    do
    :: count == shards -> break
    :: else ->
        load(t, last_visible_id(count), order_acquire, visible);
        atomic {
            if
            :: !closed[count] && claims[count] == visible ->
                closed[count] = true;
                count++
            :: else -> break
            fi
        }
    od;
    if
    :: count != shards -> reopen_shards(count)
    :: else
    fi
}

// `keyed_nodes::publish` of one row by the shard map's one writer: relaxed peeks until an empty
// slot, then the slot with the release
inline publish_key(t, shard, id, position, seen) {
    position = 0;
    do
    :: assert(position < shard_rows);
        load(t, map_slot(shard, position), order_relaxed, seen);
        if
        :: seen == empty -> break
        :: else -> position++
        fi
    od;
    load(t, key(shard, id), order_relaxed, seen);
    atomic {
        assert(newest_value(map_slot(shard, position)) == empty && !keyed[(shard) * shard_rows + (id)]);
        store(t, map_slot(shard, position), order_release, filed(seen, id));
        keyed[(shard) * shard_rows + (id)] = true
    }
}

// a batch: per shard in order the claim of its row, the earlier shares given back on a refusal;
// then the keys onto the launch stream ahead of the launch, the drain, which may fail and keep its
// failure on every shard, every key filed, and the commit across the shards inside the sequence
// word: `plan_insert`, `insert_plan_t::insert` and `insert_plan_t::finish`
inline insert_batch(t) {
    byte shard, position;
    bool claimed;
    int first_id, watermark, sequence, seen;
    shard = 0;
    do
    :: shard == shards -> break
    :: else ->
        claim_row(t, shard, first_id, claimed, watermark);
        if
        :: claimed -> part_of(t, shard) = first_id; shard++
        :: else ->
            do
            :: shard == 0 -> break
            :: else -> shard--; unclaim_row(shard, part_of(t, shard))
            od;
            goto contended
        fi
    od;
    if
    :: key_before_turn ->
        for (shard : 0 .. shards - 1) {
            store(t, key(shard, part_of(t, shard)), order_relaxed, key_of(t))
        };
        if
        :: !keys_after_drain ->
            for (shard : 0 .. shards - 1) {
                publish_key(t, shard, part_of(t, shard), position, seen)
            }
        :: else
        fi
    :: else
    fi;
    // the drain: a failed launch keeps its failure and its claim on every shard, which never turns
    // again
    if
    :: true
    :: goto contended
    fi;
    if
    :: key_before_turn && keys_after_drain ->
        for (shard : 0 .. shards - 1) {
            publish_key(t, shard, part_of(t, shard), position, seen)
        }
    :: else
    fi;
    committing[t] = true;
    read_modify_write(t, sequence_word, order_release, sequence, sequence + 1);
    for (shard : 0 .. shards - 1) {
        take_turn(t, shard, part_of(t, shard), watermark)
    };
    read_modify_write(t, sequence_word, order_release, sequence, sequence + sequence_unit - 1);
    if
    :: !key_before_turn ->
        for (shard : 0 .. shards - 1) {
            store(t, key(shard, part_of(t, shard)), order_relaxed, key_of(t));
            publish_key(t, shard, part_of(t, shard), position, seen)
        }
    :: else
    fi;
contended:
    done++
}

// a search: the snapshot of every shard's watermark between two equal reads of the sequence word
// with no commit in flight, then every key below each watermark, and `count` over each shard's map
// finding each of them once
inline search(t) {
    byte shard, id, batch, position, found;
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
        od;
        found = 0;
        for (position : 0 .. shard_rows - 1) {
            load(t, map_slot(shard, position), order_acquire, seen);
            if
            :: seen != empty && node_in(seen) < watermarks[shard] -> found++
            :: else
            fi
        };
        assert(found == watermarks[shard])
    };
    done++
}

// once every thread is done: every filed row lies below its shard's watermark under its claimer's
// key, so no row of a failed, refused or given-back batch or load is keyed
inline audit_keys() {
    byte shard, position;
    atomic {
        for (shard : 0 .. shards - 1) {
            for (position : 0 .. shard_rows - 1) {
                assert(newest_value(map_slot(shard, position)) == empty ||
                       (node_in(newest_value(map_slot(shard, position))) < newest_value(last_visible_id(shard)) &&
                        owner_of(shard, node_in(newest_value(map_slot(shard, position)))) != 0 &&
                        key_in(newest_value(map_slot(shard, position))) ==
                            key_of(owner_of(shard, node_in(newest_value(map_slot(shard, position)))) - 1)))
            }
        }
    }
}
