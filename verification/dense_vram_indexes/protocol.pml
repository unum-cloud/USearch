/**
 *  @file verification/dense_vram_indexes/protocol.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief `dense::vram_nvidia_indexes` from `include/usearch/dense/vram_nvidia_indexes.cuh`: the
 *      sharded CUDA index.
 *
 *  Its batches claim every shard's share in shard order, each claim one compare-exchange of the
 *  shard's `claimed_rows_` from its watermark, so a shard holds one batch in flight at most; they
 *  write every key before the turn, and turn every shard's watermark inside one commit of a
 *  sequence word. Its load and save close every shard's dispenser in shard order first, and reopen
 *  them at the end, the last shard first. Two shards of two rows each, one per claimer.
 *
 *  - Every key a searcher reads is the key of the thread that claimed the row, never unset.
 *  - A snapshot it accepts holds each committed batch on both shards or on neither.
 */
#include <weak_memory.pml>

// The words: each shard's watermark, `commit_sequence_`, and each shard's keys. A shard's claim and
// its closed state publish nothing a searcher reads, since the watermark bounds every walk, so they
// are ghosts here, each step on them one compare-exchange.
#define last_visible_id(shard) (shard)
#define sequence_word 2
#define key(shard, id) (3 + (shard) * shard_rows + (id))

// The sizes and values: two shards of two rows, one per claimer, a batch of one row per shard, the
// keys, and the sequence word's unit, past the commits in flight it counts.
#define shards 2
#define shard_rows 2
#define unset 0
#define key_of(t) (10 + (t))
#define sequence_unit 4

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

byte claims[shards];          // per shard, its claim: `claimed_rows_`
bool closed[shards];          // per shard, whether its dispenser holds the closed state
byte owner[shards * shard_rows];   // one past the thread that took the row; zero while untaken
#define owner_of(shard, id) owner[(shard) * shard_rows + (id)]
bool committing[mutators];    // per mutator, whether its batch reached its commit
byte part[mutators * shards]; // per mutator and shard, the row its batch landed at
#define part_of(t, shard) part[(t) * shards + (shard)]

// a row of `shard` taken by thread `t`, never one another thread took
inline take_row(t, shard, id) {
    assert(owner_of(shard, id) == 0);
    owner_of(shard, id) = (t) + 1
}

// the shard's claim of one row: from its watermark to one past it, refused while a batch is in
// flight there or the dispenser is closed
inline claim_row(t, shard, first_id, claimed) {
    atomic {
        if
        :: !closed[shard] && (claims[shard] == newest_value(last_visible_id(shard)) || !quiescent_claim) ->
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

// a batch: per shard in order the claim of its row, the earlier shares given back on a refusal;
// then every key and the commit across the shards inside the sequence word: `plan_insert`,
// `insert_plan_t::insert` and `insert_plan_t::finish`
inline insert_batch(t) {
    byte shard;
    bool claimed;
    int first_id, watermark, sequence;
    shard = 0;
    do
    :: shard == shards -> break
    :: else ->
        claim_row(t, shard, first_id, claimed);
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
            store(t, key(shard, part_of(t, shard)), order_relaxed, key_of(t))
        }
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
