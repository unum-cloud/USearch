/**
 *  @file verification/dense_ram_indexes_commits.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A batch's commit across the shards, behind one counted sequence word.
 *
 *  `include/usearch/dense/ram_indexes.hpp` makes a batch visible whole across shards. A commit
 *  lands a row on each shard, stores both keys, waits for every shard's turn, then stores the
 *  watermarks between one add on the index's commit sequence word and a second add of the unit
 *  less one, so the low part counts the commits between their stores. A closer stands for a
 *  refine, an absorb, a clear or a checkpoint load of the sharded index.
 *
 *  - An accepted snapshot holds each commit on both shards or on neither.
 *  - Every ID below an accepted snapshot has its key.
 *  - Holding them all, the closer finds no commit in flight on any shard.
 *  - Once everyone is done every dispenser is open and level with its watermark.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 closer=true
 *  @verify pass sc,rc11 commits=2 closer=true
 *  @verify fail sc closer=true close_in_shard_order=false: the commit is refused with a range it can
 *      neither publish nor give back
 *  @verify fail sc closer=true reopen_in_reverse=false: a commit dispensing on the first is refused
 *      on the second with that range stranded
 *  @verify pass sc,rc11 counted_sequence=false
 *  @verify pass sc,rc11 commits=2
 *  @verify fail sc,rc11 commits=2 counted_sequence=false: the searcher accepts a snapshot holding
 *      one shard's part
 *  @verify fail sc,rc11 commit_sequence=false: without it, the searcher takes a snapshot between the
 *      commit's two stores
 *  @verify stuck sc,rc11 commits=2 dispensing_mutex=false: dispensing each shard on its own, the
 *      ranges cross between the shards, and each commit waits for a turn only the other's store gives
 *  @verify fail rc11 closing_order=order_relaxed: the searcher accepts a closed word beside one
 *      shard's new watermark and the other's old one
 *  @verify fail rc11 before_order=order_relaxed: the unmoved word accepts the half
 *  @verify fail rc11 size_order=order_relaxed: the searcher re-reads the word as it was before the
 *      commit opened, and accepts the half
 */
#define thread_count 3
#define location_count 7
#define history_depth 5
#define shards 2
#include <weak_memory.pml>
#include <watermark_turn.pml>

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
#ifndef closing_order
#define closing_order order_release
#endif
#ifndef before_order
#define before_order order_acquire
#endif
#ifndef size_order
#define size_order order_acquire
#endif
#ifndef closer
#define closer false
#endif
#ifndef close_in_shard_order
#define close_in_shard_order true
#endif
#ifndef reopen_in_reverse
#define reopen_in_reverse true
#endif

// the word's step after the turns: the counted word's carry into its high part, or the parity's
#define sequence_unit (counted_sequence -> 4 : 2)

byte dispensed[shards];              // each shard's `next_dispensed_id_`, a ghost, since no plan reads it
bool closed[shards];                 // each shard's dispenser holds the closed state
bool dispensing;                     // the dispensing mutex is held
byte done;                           // the commits and the closer, once finished
bool commit_dispensed[commits];      // the commit has its IDs
byte commit_part[commits * shards];  // the ID each commit landed on each shard
bool commit_got[commits * shards];   // the commit dispensed its range on the shard
#define part(commit, shard) commit_part[(commit) * shards + (shard)]
#define got(commit, shard) commit_got[(commit) * shards + (shard)]

inline dispense(shard, id) {
    id = dispensed[shard];
    dispensed[shard]++
}

// a batch's commit: a row on each shard, both keys before the turns, every turn awaited outside the
// bracket of the commit sequence word, and the watermark stores inside it
proctype committer(byte t) {
    byte shard;
    int id, watermark, sequence;
    bool refused;
    // every shard's range under the mutex, so no two commits cross between the shards; a closed
    // shard refuses its dispense, and every shard takes one row, so only the first can refuse
    if
    :: dispensing_mutex -> atomic { !dispensing -> dispensing = true }
    :: else
    fi;
    shard = 0;
    do
    :: shard == shards -> break
    :: else ->
        atomic {
            if
            :: closed[shard] -> assert(shard == 0); refused = true
            :: else -> dispense(shard, id); part(t, shard) = id; got(t, shard) = true
            fi
        };
        if
        :: refused -> break
        :: else -> shard++
        fi
    od;
    if
    :: dispensing_mutex -> dispensing = false
    :: else
    fi;
    if
    :: refused -> goto finished
    :: else
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
        end_turn(t, last_visible_id(shard), order_release, part(t, shard) + 1)
    };
    if
    :: commit_sequence -> read_modify_write(t, sequence_word, closing_order, sequence, sequence + sequence_unit - 1)
    :: else
    fi;
finished:
    done++
}

// `close_` on every shard in turn, a refusal reopening those already closed; holding every shard,
// no commit sits between its dispense and its turn on any; then every shard reopens, the last first
proctype closing() {
    byte index, shard, commit;
    bool refused;
    index = 0;
    do
    :: index == shards || refused -> break
    :: else ->
        shard = (close_in_shard_order -> index : shards - 1 - index);
        atomic {
            if
            :: dispensed[shard] == newest_value(last_visible_id(shard)) -> closed[shard] = true
            :: else -> refused = true
            fi
        };
        index++
    od;
    if
    :: !refused ->
        atomic {
            for (commit : 0 .. commits - 1) {
                for (shard : 0 .. shards - 1) {
                    assert(!got(commit, shard) || newest_value(last_visible_id(shard)) > part(commit, shard))
                }
            }
        }
    :: else
    fi;
    for (index : 0 .. shards - 1) {
        closed[(reopen_in_reverse -> shards - 1 - index : index)] = false
    };
    done++
}

// once everyone is done: every dispenser open and level with its watermark
proctype checker() {
    byte shard;
    (done == commits + 1);
    for (shard : 0 .. shards - 1) {
        assert(!closed[shard] && dispensed[shard] == newest_value(last_visible_id(shard)))
    }
}

// a search over both shards: the word acquired, the shards' watermarks snapshotted, the word
// re-read, and a retry while it counts a commit in flight or moved
proctype searcher(byte t) {
    byte shard, id, commit;
    int before, after, watermarks[shards], seen;
retry:
    load(t, sequence_word, before_order, before);
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), size_order, watermarks[shard])
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
        run searcher(2);
        if
        :: closer -> run closing(); run checker()
        :: else
        fi
    }
}
