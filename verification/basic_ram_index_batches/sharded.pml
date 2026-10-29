/**
 *  @file verification/basic_ram_index_batches/sharded.pml
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief A basic batch across two shards publishes whole, even when two batches cross between them.
 *
 *  `basic::ram_indexes::plan_insert` and `insert_plan_t::finish` from
 *  `include/usearch/basic/ram_indexes.hpp`. Each batch dispenses one row on each shard, a shard at
 *  a time with no lock across them, so two batches can take opposite turn orders on the two
 *  shards. `finish` opens the commit sequence word, takes each shard's turn in placement order,
 *  filing its key in that shard's map inside the turn, then closes the word. The looker counts the
 *  first batch's key as `count` does, under `read_until_settled`.
 *
 *  - No batch waits forever, crossed or not.
 *  - A count the looker accepts holds the first batch's key on both shards or on neither.
 *
 *  @verify pass sc,rc11
 *  @verify stuck sc await_every_turn=true: awaiting both turns before filing, as the dense plural
 *      does under its dispensing mutex, each crossed batch waits for a turn only the other's store
 *      gives
 *  @verify fail sc,rc11 sequence=false: without it, the looker probes one shard before the batch's
 *      turn and the other after, and counts its key on one shard alone
 */
#define thread_count 3
#define location_count 7
#define history_depth 5
#define shards 2
#define batches 2
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The words, by the index member each stands for: the set's `commit_sequence_`, then each shard's
// watermark and its map's seat per node, which the one writer of its turn files.
#define sequence_word 0
#define last_visible_id(shard) (1 + (shard))
#define seat(shard, id) (3 + (shard) * batches + (id))

// The values: a seat holds nothing yet or the key of the batch that filed it; the counted sequence
// word's low part, below `sequence_unit`, counts the commits in flight.
#define empty 0
#define key_of(batch) (1 + (batch))
#define sequence_unit 4

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef await_every_turn
#define await_every_turn false
#endif
#ifndef sequence
#define sequence true
#endif

byte dispensed[shards];          // each shard's `next_dispensed_id_`, a ghost, since no reader loads it
byte parts[batches * shards];    // the ID each batch dispensed on each shard
#define part(batch, shard) parts[(batch) * shards + (shard)]

// one batch: a dispense per shard, then `finish`, the bracket around every turn, each shard's key
// filed by its turn holder between the wait and the watermark's release
proctype committer(byte t) {
    byte shard;
    int opened, watermark;
    for (shard : 0 .. shards - 1) {
        atomic { part(t, shard) = dispensed[shard]; dispensed[shard]++ }
    };
    if
    :: sequence -> read_modify_write(t, sequence_word, order_release, opened, opened + 1)
    :: else
    fi;
    if
    :: await_every_turn ->
        for (shard : 0 .. shards - 1) { await_turn(t, last_visible_id(shard), part(t, shard), watermark) };
        for (shard : 0 .. shards - 1) {
            store(t, seat(shard, part(t, shard)), order_release, key_of(t));
            end_turn(t, last_visible_id(shard), order_release, part(t, shard) + 1)
        }
    :: else ->
        for (shard : 0 .. shards - 1) {
            await_turn(t, last_visible_id(shard), part(t, shard), watermark);
            store(t, seat(shard, part(t, shard)), order_release, key_of(t));
            end_turn(t, last_visible_id(shard), order_release, part(t, shard) + 1)
        }
    fi;
    if
    :: sequence -> read_modify_write(t, sequence_word, order_release, opened, opened + sequence_unit - 1)
    :: else
    fi
}

// `count` of the first batch's key: `read_until_settled`'s commit loop around each shard's `count`,
// which acquires the watermark, then every seat, and keeps a hit only below the watermark; no window
// opens here, so its epoch loop is left out
proctype looker(byte t) {
    byte shard, id;
    int before, after, watermark, seen, found;
retry:
    found = 0;
    load(t, sequence_word, order_acquire, before);
    if
    :: sequence && before % sequence_unit != 0 ->
        (newest_value(sequence_word) % sequence_unit == 0);
        goto retry
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermark);
        for (id : 0 .. batches - 1) {
            load(t, seat(shard, id), order_acquire, seen);
            if
            :: seen == key_of(0) && id < watermark -> found++
            :: else
            fi
        }
    };
    load(t, sequence_word, order_relaxed, after);
    if
    :: sequence && after != before -> goto retry
    :: else
    fi;
    assert(found == 0 || found == shards)
}

init { atomic { run committer(0); run committer(1); run looker(2) } }
