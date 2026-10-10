/**
 *  The sharded dense index's plans from `include/usearch/dense/ram_indexes.hpp`: the monolith save
 *  taking one `plan_save` per shard, and the load path landing every shard's keys under one
 *  `finish`, beside the search merge reading `keys[node]` below each shard's watermark.
 *
 *  Two shards of two rows each. A shard's dispenser is a word here, since the plans compare the
 *  watermark with it.
 */
#include "../weak_memory.pml"
#include "../watermark_turn.pml"

// The sizes: the shards, and the rows one shard's IDs reach.
#define shards 2
#define row_limit 2

// The words, by the index member each stands for, per shard.
#define last_visible_id(shard) (shard)
#define next_dispensed_id(shard) (2 + (shard))
#define key(shard, id) (4 + (shard) * row_limit + (id))

// The values a key slot carries: unset, or the key of the thread that landed the row.
#define unset 0
#define key_of(thread) (10 + (thread))

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef whole_batches
#define whole_batches false
#endif
#ifndef keys_before_finish
#define keys_before_finish true
#endif

// `insert`'s tail, for one row of one shard: the dispense with `on_assigned`'s key in one step,
// then the turn
inline land_row(t, shard, id) {
    int watermark;
    atomic {
        read_modify_write(t, next_dispensed_id(shard), order_acq_rel, id, id + 1);
        store(t, key(shard, id), order_relaxed, key_of(t))
    };
    take_turn(t, last_visible_id(shard), id, id + 1, order_release, watermark)
}

// the load path's keys at the shard's base, relaxed
inline land_keys(t, shard, base) {
    byte row;
    for (row : 0 .. row_limit - 1) {
        store(t, key(shard, base + row), order_relaxed, key_of(t))
    }
}

// the search merge: every ID below the watermark it acquired resolves through the key column
inline search(t) {
    byte shard, id;
    int watermark, seen;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermark);
        id = 0;
        do
        :: id == watermark -> break
        :: else -> load(t, key(shard, id), order_relaxed, seen); assert(seen != unset); id++
        od
    }
}
