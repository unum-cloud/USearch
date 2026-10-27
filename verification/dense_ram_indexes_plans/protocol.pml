/**
 *  @file verification/dense_ram_indexes_plans/protocol.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The sharded dense index's plans from `include/usearch/dense/ram_indexes.hpp`.
 *
 *  The monolith save taking one `plan_save` per shard, and the load path landing every shard's keys
 *  under one `finish`, beside the search merge reading `keys[node]` below each shard's watermark.
 *  Two shards of two rows each. A shard's dispenser is a word here, since the plans compare the
 *  watermark with it.
 */
#include <weak_memory.pml>
#include <watermark_turn.pml>

// The sizes: the shards, and the rows one shard's IDs reach.
#define shards 2
#define row_limit 2

// The words, by the index member each stands for, per shard; the last is `commit_sequence_`.
#define last_visible_id(shard) (shard)
#define next_dispensed_id(shard) (2 + (shard))
#define key(shard, id) (4 + (shard) * row_limit + (id))
#define sequence_word 8

// The values a key slot carries: unset, or the key of the thread that landed the row.
#define unset 0
#define key_of(thread) (10 + (thread))

// The counted sequence's carry: opening a bracket adds one, closing it adds the unit less one.
#define sequence_unit 4

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef plans_before_finish
#define plans_before_finish true
#endif
#ifndef keys_before_finish
#define keys_before_finish true
#endif

// a planned batch of one row per shard: `insert` dispenses every shard's range before any row
// lands, then the keys; `finish` awaits every turn, then stores the watermarks inside the bracket
// of the commit sequence word
inline insert_batch(t) {
    byte shard;
    int first[shards], watermark, sequence;
    for (shard : 0 .. shards - 1) {
        read_modify_write(t, next_dispensed_id(shard), order_acq_rel, first[shard], first[shard] + 1)
    };
    for (shard : 0 .. shards - 1) {
        store(t, key(shard, first[shard]), order_relaxed, key_of(t))
    };
    for (shard : 0 .. shards - 1) {
        await_turn(t, last_visible_id(shard), first[shard], watermark)
    };
    read_modify_write(t, sequence_word, order_release, sequence, sequence + 1);
    for (shard : 0 .. shards - 1) {
        end_turn(t, last_visible_id(shard), order_release, first[shard] + 1)
    };
    read_modify_write(t, sequence_word, order_release, sequence, sequence + sequence_unit - 1)
}

// one shard's `plan_save`: the watermark into @p planned, refused while a row is between its
// dispense and its publication; under `sc` it refuses exactly then, or when a row published
// between its two loads
inline plan_shard(t, shard, planned) {
    int dispensed_now;
    load(t, last_visible_id(shard), order_acquire, planned);
    atomic {
        load(t, next_dispensed_id(shard), order_acquire, dispensed_now);
        assert(memory != sc || (planned != dispensed_now) ==
               (newest_value(next_dispensed_id(shard)) != newest_value(last_visible_id(shard))
                || newest_value(last_visible_id(shard)) != planned))
    };
    if
    :: planned != dispensed_now -> goto contended
    :: else
    fi
}

// one shard's `finish`: the acquire fence, the relaxed re-check against @p planned, then the plain
// key copy of `keys[0..planned)` at the shard's base, which holds no unset key
inline finish_shard(t, shard, planned) {
    byte id;
    int again;
    fence_acquire(t);
    load(t, last_visible_id(shard), order_relaxed, again);
    if
    :: again != planned -> goto contended
    :: else
    fi;
    load(t, next_dispensed_id(shard), order_relaxed, again);
    if
    :: again != planned -> goto contended
    :: else
    fi;
    do
    :: id == planned -> break
    :: else -> load(t, key(shard, id), order_relaxed, again); assert(again != unset); id++
    od
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
