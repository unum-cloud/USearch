/**
 *  @file verification/dense_ram_indexes_gate/leading.pml
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief The MaxSim scan's `leads_` over a snapshot: each key nominated at exactly one row.
 *
 *  Two shards and two entries of key K. A batch spanning both lands key L on shard 0 and K on shard
 *  1 and commits them inside the sequence bracket; a single insert lands K on shard 0. Each files
 *  its keys into the shard's map while holding its turn. The scanner takes the snapshot under
 *  `read_committed` and asks `leads_` of every row below it: no earlier shard holds the key below
 *  its watermark, and no lower node of the row's shard does.
 *
 *  - An accepted snapshot nominates each visible key at exactly one row.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc filter=none: probing shard 0's map without its watermark, the scanner finds the
 *      single insert's K filed above the snapshot, so the visible K on shard 1 does not lead and
 *      the key goes unnominated
 */
#define thread_count 3
#define location_count 12
#define history_depth 3
#include <dense_ram_indexes_gate/protocol.pml>

// The words past the protocol's: `commit_sequence_`, each shard's key column, and K's chain in
// each shard's map.
#define commit_sequence 5
#define key(shard, id) (6 + (shard) * 2 + (id))
#define map_slot(shard) (10 + (shard))

// The values: the keys, a slot as one past its node, and the counted sequence word's unit.
#define unset 0
#define key_k 1
#define key_l 2
#define empty 0
#define sequence_unit 4

// one ID of the shard, its claim and dispense as one step
inline take_id(shard, id) { atomic { id = dispensed[shard]; dispensed[shard]++ } }

// L on shard 0 and K on shard 1: both turns held, K filed on shard 1, then the watermarks inside
// the bracket; L's chain is not K's, so its filing is not modelled
proctype spanning(byte t) {
    int first, second, watermark, opened;
    take_id(0, first);
    take_id(1, second);
    store(t, key(0, first), order_relaxed, key_l);
    store(t, key(1, second), order_relaxed, key_k);
    await_turn(t, watermark(0), first, watermark);
    await_turn(t, watermark(1), second, watermark);
    store(t, map_slot(1), order_release, second + 1);
    read_modify_write(t, commit_sequence, order_release, opened, opened + 1);
    end_turn(t, watermark(0), order_release, first + 1);
    end_turn(t, watermark(1), order_release, second + 1);
    read_modify_write(t, commit_sequence, order_release, opened, opened + sequence_unit - 1)
}

// K on shard 0, filed inside its turn
proctype single(byte t) {
    int id, watermark;
    take_id(0, id);
    store(t, key(0, id), order_relaxed, key_k);
    await_turn(t, watermark(0), id, watermark);
    store(t, map_slot(0), order_release, id + 1);
    end_turn(t, watermark(0), order_release, id + 1)
}

// `keyed_nodes::contains` and `find_each` on K's chain: the slot acquired, and a node at or above
// the snapshot's watermark dropped unless the filter is off
#define kept(seen, visible) ((seen) != empty && (filter == none || (seen) - 1 < (visible)))

// the snapshot under `read_committed`, then `leads_` for every row below it holding K
proctype scanner(byte t) {
    byte shard, node, leaders, visible;
    int before, after, watermarks[2], seen;
    bool leads;
retry:
    load(t, commit_sequence, order_acquire, before);
    if
    :: before % sequence_unit != 0 ->
        (newest_value(commit_sequence) % sequence_unit == 0);
        goto retry
    :: else
    fi;
    load(t, watermark(0), order_acquire, watermarks[0]);
    load(t, watermark(1), order_acquire, watermarks[1]);
    load(t, commit_sequence, order_relaxed, after);
    if
    :: after != before -> goto retry
    :: else
    fi;
    for (shard : 0 .. 1) {
        node = 0;
        do
        :: node == watermarks[shard] -> break
        :: else ->
            load(t, key(shard, node), order_relaxed, seen);
            assert(seen != unset);
            if
            :: seen == key_k ->
                visible++;
                leads = true;
                if
                :: shard == 1 ->
                    load(t, map_slot(0), order_acquire, seen);
                    leads = !kept(seen, watermarks[0])
                :: else
                fi;
                load(t, map_slot(shard), order_acquire, seen);
                leads = leads && !(kept(seen, watermarks[shard]) && seen - 1 < node);
                if
                :: leads -> leaders++
                :: else
                fi
            :: else
            fi;
            node++
        od
    };
    assert(visible == 0 || leaders == 1)
}

init { atomic { run spanning(0); run single(1); run scanner(2) } }
