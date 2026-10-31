/**
 *  @file verification/dense_ram_indexes_plans/protocol.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The sharded dense index's plans from `include/usearch/dense/ram_indexes.hpp`.
 *
 *  The monolith save taking one `plan_save` per shard, and the load path placing every shard's keys
 *  through its plan, filed by one publisher per shard before `finish`, beside the search merge
 *  reading `keys[node]` and counting each shard's map below its watermark. Two shards of two rows
 *  each. A shard's dispenser is a word here, since the plans compare the watermark with it.
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
#define map_slot(shard, position) (9 + (shard) * row_limit + (position))

// The values a key slot carries: unset, or the key of the thread that placed the row. A map slot
// is the key beside its node in one word, published with one release: its tag's ordering against
// its slot is `ram_tables.pml`'s concern.
#define unset 0
#define key_of(thread) (10 + (thread))
#define empty 0
#define filed(key, id) ((key) * row_limit + (id) + 1)
#define key_in(slot) (((slot) - 1) / row_limit)
#define node_in(slot) (((slot) - 1) % row_limit)

// The counted sequence's carry: opening a bracket adds one, closing it adds the unit less one.
#define sequence_unit 4

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef plans_before_finish
#define plans_before_finish true
#endif
#ifndef keys_before_finish
#define keys_before_finish true
#endif
#ifndef keys_once_covered
#define keys_once_covered true
#endif

bool keyed[shards * row_limit]; // a ghost: the node is filed in its shard's map

// `keyed_nodes::publish` of one node by the shard map's one writer: relaxed peeks until an empty
// slot, then the slot with the release
inline publish_key(t, shard, id, position, seen) {
    position = 0;
    do
    :: assert(position < row_limit);
        load(t, map_slot(shard, position), order_relaxed, seen);
        if
        :: seen == empty -> break
        :: else -> position++
        fi
    od;
    load(t, key(shard, id), order_relaxed, seen);
    atomic {
        assert(newest_value(map_slot(shard, position)) == empty && !keyed[(shard) * row_limit + (id)]);
        store(t, map_slot(shard, position), order_release, filed(seen, id));
        keyed[(shard) * row_limit + (id)] = true
    }
}

// a planned batch of one row per shard: `insert` dispenses every shard's range before any row
// places, then the keys; `finish` awaits every turn, files each shard's key while its turn is held,
// then stores the watermarks inside the bracket of the commit sequence word
inline insert_batch(t) {
    byte shard, position;
    int first[shards], watermark, sequence, seen;
    for (shard : 0 .. shards - 1) {
        read_modify_write(t, next_dispensed_id(shard), order_acq_rel, first[shard], first[shard] + 1)
    };
    for (shard : 0 .. shards - 1) {
        store(t, key(shard, first[shard]), order_relaxed, key_of(t))
    };
    for (shard : 0 .. shards - 1) {
        await_turn(t, last_visible_id(shard), first[shard], watermark)
    };
    for (shard : 0 .. shards - 1) {
        publish_key(t, shard, first[shard], position, seen)
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

// the load plan's `copy_keys` of @p rows rows at the shard's base, relaxed, each step the one
// writer of its rows
inline copy_keys(t, shard, base, rows) {
    byte row;
    row = 0;
    do
    :: row == rows -> break
    :: else -> store(t, key(shard, base + row), order_relaxed, key_of(t)); row++
    od
}

// the load plan's `publish_keys`: its shard's one publisher files every row of the plan
inline publish_keys(t, shard, base, position, seen) {
    byte filing;
    for (filing : 0 .. row_limit - 1) {
        publish_key(t, shard, base + filing, position, seen)
    }
}

// the search merge: every ID below the watermark it acquired resolves through the key column, and
// `count` over the map finds each of them once, dropping the nodes at or above the watermark
inline search(t) {
    byte shard, id, position, found;
    int watermark, seen;
    for (shard : 0 .. shards - 1) {
        load(t, last_visible_id(shard), order_acquire, watermark);
        id = 0;
        do
        :: id == watermark -> break
        :: else -> load(t, key(shard, id), order_relaxed, seen); assert(seen != unset); id++
        od;
        found = 0;
        for (position : 0 .. row_limit - 1) {
            load(t, map_slot(shard, position), order_acquire, seen);
            if
            :: seen != empty && node_in(seen) < watermark -> found++
            :: else
            fi
        };
        assert(found == watermark)
    }
}

// once every thread is done: every filed node lies below its shard's watermark, under the key its
// row carries, so no entry of a refused or abandoned plan is keyed
inline audit_keys() {
    byte shard, position;
    atomic {
        for (shard : 0 .. shards - 1) {
            for (position : 0 .. row_limit - 1) {
                assert(newest_value(map_slot(shard, position)) == empty ||
                       (node_in(newest_value(map_slot(shard, position))) < newest_value(last_visible_id(shard)) &&
                        key_in(newest_value(map_slot(shard, position))) ==
                            newest_value(key(shard, node_in(newest_value(map_slot(shard, position)))))))
            }
        }
    }
}
