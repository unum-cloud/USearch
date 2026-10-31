/**
 *  @file verification/dense_ram_indexes_commits.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A batch's commit across the shards, behind one counted sequence word.
 *
 *  `include/usearch/dense/ram_indexes.hpp` makes a batch visible whole across shards. A commit places
 *  a row and its key on each shard, awaits every turn, files each shard's key into its map while
 *  holding that turn, then stores the watermarks between two adds on the commit sequence word. The
 *  looker counts the first commit's key as `count` does, under `read_until_settled`; a closer,
 *  standing for a refine, an absorb or a clear, runs in its place.
 *
 *  - An accepted snapshot holds each commit on both shards or on neither.
 *  - Every ID below an accepted snapshot has its key.
 *  - A count the looker accepts holds the commit's key on every shard or on none.
 *  - Holding them all, the closer finds no commit in flight on any shard.
 *  - Once everyone is done every dispenser is open and level with its watermark.
 *
 *  @budget 4096
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 closer=true
 *  @verify pass sc,rc11 commits=2 closer=true
 *  @verify fail sc closer=true close_in_shard_order=false: the commit is refused with a range it can
 *      neither publish nor give back
 *  @verify fail sc closer=true reopen_in_reverse=false: a commit dispensing on the first is refused
 *      on the second with that range stranded
 *  @verify pass sc,rc11 sequence=parity
 *  @verify pass sc,rc11 commits=2
 *  @verify fail sc,rc11 commits=2 sequence=parity: the searcher accepts a snapshot holding one
 *      shard's part
 *  @verify fail sc,rc11 sequence=none: without it, the looker probes one shard before the commit's
 *      watermark stores and the other after, and counts the key on one shard alone
 *  @verify stuck sc,rc11 commits=2 dispensing_mutex=false: dispensing each shard on its own, the
 *      ranges cross between the shards, and each commit waits for a turn only the other's store gives
 *  @verify fail rc11 closing_order=order_relaxed: the searcher accepts a closed word beside one
 *      shard's new watermark and the other's old one
 *  @verify fail rc11 before_order=order_relaxed: the unmoved word accepts the half
 *  @verify fail rc11 size_order=order_relaxed: the searcher re-reads the word as it was before the
 *      commit opened, and accepts the half
 */
#define thread_count 4
#define location_count 11
#define history_depth 5
#define shards 2
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for, per shard; the first is `commit_sequence_`.
#define sequence_word 0
#define last_visible_id(shard) (1 + (shard))
#define key(shard, id) (3 + (shard) * 2 + (id))
#define map_slot(shard, position) (7 + (shard) * 2 + (position))

// The values a key slot carries: unset, or the key of the thread that placed the row. A map slot
// is the key beside its node in one word, published with one release: its tag's ordering against
// its slot is `ram_tables.pml`'s concern.
#define unset 0
#define key_of(thread) (10 + (thread))
#define empty 0
#define filed(key, id) ((key) * 2 + (id) + 1)
#define key_in(slot) (((slot) - 1) / 2)
#define node_in(slot) (((slot) - 1) % 2)

// The sequence word's forms: the counted word `begin_commit` and `end_commit` move, the parity a
// single commit in flight would allow, or no bracket at all.
#define none 0
#define parity 1
#define counted 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef commits
#define commits 1
#endif
#if commits < 1 || commits > 2
#error "commits is 1 or 2"
#endif
#ifndef sequence
#define sequence counted
#endif
#if sequence != none && sequence != parity && sequence != counted
#error "sequence is none, parity or counted"
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
#define sequence_unit (sequence == counted -> 4 : 2)

byte dispensed[shards];              // each shard's `next_dispensed_id_`, a ghost, since no plan reads it
bool closed[shards];                 // each shard's dispenser holds the closed state
bool dispensing;                     // the dispensing mutex is held
byte done;                           // the commits and the closer, once finished
bool commit_dispensed[commits];      // the commit has its IDs
byte commit_part[commits * shards];  // the ID each commit placed on each shard
bool commit_got[commits * shards];   // the commit dispensed its range on the shard
#define part(commit, shard) commit_part[(commit) * shards + (shard)]
#define got(commit, shard) commit_got[(commit) * shards + (shard)]

inline dispense(shard, id) {
    id = dispensed[shard];
    dispensed[shard]++
}

// `keyed_nodes::publish` of one row by its turn holder, the shard map's one writer: relaxed peeks
// until an empty slot, then the slot with the release
inline publish_keys(t, shard, id, position, seen) {
    position = 0;
    do
    :: assert(position < commits);
        load(t, map_slot(shard, position), order_relaxed, seen);
        if
        :: seen == empty -> break
        :: else -> position++
        fi
    od;
    load(t, key(shard, id), order_relaxed, seen);
    atomic {
        assert(newest_value(map_slot(shard, position)) == empty);
        store(t, map_slot(shard, position), order_release, filed(seen, id))
    }
}

// a batch's commit: a row on each shard, both keys before the turns, every turn awaited outside the
// bracket of the commit sequence word, each shard's keys filed while its turn is held, and the
// watermark stores inside the bracket
proctype committer(byte t) {
    byte shard, position;
    int id, watermark, opened, seen;
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
    for (shard : 0 .. shards - 1) {
        publish_keys(t, shard, part(t, shard), position, seen)
    };
    if
    :: sequence != none -> read_modify_write(t, sequence_word, order_release, opened, opened + 1)
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        end_turn(t, last_visible_id(shard), order_release, part(t, shard) + 1)
    };
    if
    :: sequence != none -> read_modify_write(t, sequence_word, closing_order, opened, opened + sequence_unit - 1)
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
    :: sequence != none && (before % sequence_unit != 0 || after != before) ->
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

// `count` of the first commit's key: `read_until_settled`'s commit loop around every shard's
// `keyed_nodes::count`, each the watermark acquired, the chain probed with acquires and the nodes at
// or above it dropped; no window opens here, so its epoch loop is left out
proctype looker(byte t) {
    byte shard, position;
    int before, after, watermark, seen, found;
retry:
    found = 0;
    load(t, sequence_word, before_order, before);
    if
    :: sequence != none && before % sequence_unit != 0 ->
        (newest_value(sequence_word) % sequence_unit == 0);
        goto retry
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), size_order, watermark);
        position = 0;
        do
        :: position == commits -> break
        :: else ->
            load(t, map_slot(shard, position), order_acquire, seen);
            if
            :: seen == empty -> break
            :: else
            fi;
            if
            :: key_in(seen) == key_of(0) && node_in(seen) < watermark -> found++
            :: else
            fi;
            position++
        od
    };
    load(t, sequence_word, order_relaxed, after);
    if
    :: sequence != none && after != before -> goto retry
    :: else
    fi;
    assert(found == 0 || found == shards)
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
        :: else -> run looker(3)
        fi
    }
}
