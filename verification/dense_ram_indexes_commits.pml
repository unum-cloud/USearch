/**
 *  @brief A batch's commit across the shards, behind one counted sequence word.
 *
 *  `include/usearch/dense/ram_indexes.hpp` makes a batch visible whole across shards. A commit
 *  lands a row on each shard, its ranges dispensed in one step as the dispensing mutex does, stores
 *  both keys, waits for every shard's turn, then stores the watermarks between one add on the
 *  index's commit sequence word and a second add of the unit less one, so the low part counts the
 *  commits between their stores. A searcher acquires the word, snapshots both shards' watermarks,
 *  re-reads the word, and retries while the low part is not zero or the word moved. An accepted
 *  snapshot holds each commit on both shards or on neither, and every ID below it has its key.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 counted_sequence=false
 *  @verify pass sc,rc11 commits=2
 *  @verify fail sc,rc11 commits=2 counted_sequence=false: a parity word made odd before the stores
 *      and even after holds for one commit and fails for two: one commit's add makes it odd, the
 *      other's even again while both are inside their brackets, and the searcher accepts a
 *      snapshot holding one shard's part; the counted word returns to a multiple of the unit only
 *      once both are done
 *  @verify fail sc,rc11 commit_sequence=false: the sequence word brackets the watermark stores;
 *      without it, the searcher takes a snapshot between the commit's two stores
 *  @verify stuck sc,rc11 commits=2 dispensing_mutex=false: the waits sit outside the word's
 *      bracket, which is safe only because the dispensing mutex keeps two commits' ranges in one
 *      order on every shard; dispensing each shard on its own, the ranges cross between the
 *      shards, and each commit waits for a turn only the other's store gives
 */
#define thread_count 3
#define location_count 7
#define history_depth 5
#define shards 2
#include "weak_memory.pml"
#include "watermark_turn.pml"

// The words, by the index member each stands for, per shard; the first is `commit_sequence_`.
#define sequence_word 0
#define last_visible_id(shard) (1 + (shard))
#define key(shard, id) (3 + (shard) * 2 + (id))

// The values a key slot carries: unset, or the key of the thread that landed the row.
#define unset 0
#define key_of(thread) (10 + (thread))

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef commits
#define commits 1
#endif
#if commits < 1 || commits > 2
#error "commits is 1 or 2"
#endif
#ifndef counted_sequence
#define counted_sequence true
#endif
#ifndef commit_sequence
#define commit_sequence true
#endif
#ifndef dispensing_mutex
#define dispensing_mutex true
#endif

// the word's step after the turns: the counted word's carry into its high part, or the parity's
#define sequence_unit (counted_sequence -> 4 : 2)

byte dispensed[shards];              // each shard's `next_dispensed_id_`, a ghost, since no plan reads it
bool commit_dispensed[commits];      // the commit has its IDs
byte commit_part[commits * shards];  // the ID each commit landed on each shard
#define part(commit, shard) commit_part[(commit) * shards + (shard)]

inline dispense(shard, id) {
    id = dispensed[shard];
    dispensed[shard]++
}

// a batch's commit: a row on each shard, both keys before the turns, every turn awaited outside the
// bracket of the commit sequence word, and the watermark stores inside it
proctype committer(byte t) {
    byte shard;
    int id, watermark, sequence;
    if
    :: dispensing_mutex ->
        // every shard's range in one step, so no two commits cross between the shards
        atomic {
            for (shard : 0 .. shards - 1) {
                dispense(shard, id);
                part(t, shard) = id
            }
        }
    :: else ->
        for (shard : 0 .. shards - 1) {
            atomic { dispense(shard, id); part(t, shard) = id }
        }
    fi;
    commit_dispensed[t] = true;
    for (shard : 0 .. shards - 1) {
        store(t, key(shard, part(t, shard)), order_relaxed, key_of(t))
    };
    // the waits hold no bracket open: no later range on a shard can take its turn first
    for (shard : 0 .. shards - 1) {
        await_turn(t, last_visible_id(shard), part(t, shard), watermark)
    };
    if
    :: commit_sequence -> read_modify_write(t, sequence_word, order_release, sequence, sequence + 1)
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        store(t, last_visible_id(shard), order_release, part(t, shard) + 1)
    };
    if
    :: commit_sequence -> read_modify_write(t, sequence_word, order_release, sequence, sequence + sequence_unit - 1)
    :: else
    fi
}

// a search over both shards: the word acquired, the shards' watermarks snapshotted, the word
// re-read, and a retry while it counts a commit in flight or moved
proctype searcher(byte t) {
    byte shard, id, commit;
    int before, after, watermarks[shards], seen;
retry:
    load(t, sequence_word, order_acquire, before);
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermarks[shard])
    };
    load(t, sequence_word, order_relaxed, after);
    if
    :: commit_sequence && (before % sequence_unit != 0 || after != before) ->
        (newest_value(sequence_word) % sequence_unit == 0);
        goto retry
    :: else
    fi;
    for (commit : 0 .. commits - 1) {
        assert(!commit_dispensed[commit] ||
               (watermarks[0] > part(commit, 0)) == (watermarks[1] > part(commit, 1)))
    };
    for (shard : 0 .. shards - 1) {
        id = 0;
        do
        :: id == watermarks[shard] -> break
        :: else -> load(t, key(shard, id), order_relaxed, seen); assert(seen != unset); id++
        od
    }
}

init {
    atomic {
        run committer(0);
        if
        :: commits == 2 -> run committer(1)
        :: else
        fi;
        run searcher(2)
    }
}
