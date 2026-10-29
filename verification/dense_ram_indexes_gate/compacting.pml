/**
 *  @file verification/dense_ram_indexes_gate/compacting.pml
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief An in-place compaction moving an entry between shards, beside a looker counting its key:
 *      the landing's `finish` and `forget_moved_` both run inside the window.
 *
 *  Shard 0 holds key A at row 0 and shard 1 key B at row 0; each shard's map holds B's chain in one
 *  slot. The compactor takes the unique side, closes both shards, lands B onto shard 0 at row 1 and
 *  files it there, above the watermark. Inside its window and the commit bracket it runs `finish`,
 *  which moves shard 0's watermark past B, then `forget_moved_` on shard 1: the watermark to zero,
 *  then the map cleared. The looker sums both shards' `count` of B under `read_until_settled`, the
 *  commit sequence's loop inside the epoch's.
 *
 *  - A validated count sees the moved entry exactly once.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc keys_inside_window=false: forgetting the moved shard past the window, a count
 *      that starts after the window finds B under both shards' watermarks
 *  @verify fail sc count_bracket=false: without the epoch and the commit sequence, a count reads
 *      shard 0's watermark before `finish` and shard 1's after `forget_moved_`, and misses B
 */
#define thread_count 2
#define location_count 8
#define history_depth 3
#include <dense_ram_indexes_gate/protocol.pml>

// The words past the protocol's: B's chain in each shard's map, and `commit_sequence_`.
#define map_slot(shard) (5 + (shard))
#define commit_sequence 7

// The values: the key, a slot as its key above its node, zero when empty, and the counted sequence
// word's unit.
#define key_b 2
#define empty 0
#define filed(key, node) ((key) * 4 + (node) + 1)
#define key_in(slot) ((slot) / 4)
#define node_in(slot) ((slot) % 4 - 1)
#define sequence_unit 4

// `forget_` of a moved shard: the watermark to zero with a release, then `keyed_nodes::forget`
inline forget_moved(t, shard) {
    store(t, watermark(shard), order_release, 0);
    atomic { claimed[shard] = 0; dispensed[shard] = 0 };
    store(t, map_slot(shard), order_relaxed, empty)
}

proctype compactor(byte t) {
    int observed, seen, epoch_seen, opened;
    byte shard;
    bool succeeded, refused;
    try_lock(t, gate, observed, succeeded);
    if
    :: succeeded ->
        close_shards(t, 2, refused);
        if
        :: !refused ->
            // the landing's steps, then its one publisher filing B above shard 0's watermark
            store(t, map_slot(0), order_release, filed(key_b, 1));
            begin_window(t, epoch_seen);
            read_modify_write(t, commit_sequence, order_release, opened, opened + 1);
            atomic { claimed[0]++; dispensed[0]++ };
            store(t, watermark(0), order_release, 2);
            if
            :: keys_inside_window -> forget_moved(t, 1)
            :: else
            fi;
            read_modify_write(t, commit_sequence, order_release, opened, opened + sequence_unit - 1);
            end_window(t, epoch_seen);
            if
            :: !keys_inside_window -> forget_moved(t, 1)
            :: else
            fi;
            reopen_shards(2)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

// `count(B)` under `read_until_settled`: the epoch, then the commit sequence acquired, a commit in
// flight waited out, and per shard the watermark acquired, B's chain acquired, and a node at or
// above the watermark dropped; a moved sequence re-reads, then a moved epoch retries
proctype looker(byte t) {
    int epoch_seen, again, before, after, watermark, seen;
    byte shard, found;
    bool valid;
    do
    :: if
        :: count_bracket -> read_epoch(t, epoch_seen)
        :: else
        fi;
        do
        :: if
            :: count_bracket -> load(t, commit_sequence, order_acquire, before)
            :: else
            fi;
            if
            :: before % sequence_unit != 0 -> (newest_value(commit_sequence) % sequence_unit == 0)
            :: else ->
                found = 0;
                for (shard : 0 .. 1) {
                    load(t, watermark(shard), order_acquire, watermark);
                    load(t, map_slot(shard), order_acquire, seen);
                    if
                    :: key_in(seen) == key_b && node_in(seen) < watermark -> found++
                    :: else
                    fi
                };
                if
                :: count_bracket -> load(t, commit_sequence, order_relaxed, after)
                :: else
                fi;
                if
                :: after == before -> break
                :: else
                fi
            fi
        od;
        valid = true;
        if
        :: count_bracket -> still_valid(t, epoch_seen, again, valid)
        :: else
        fi;
        if
        :: valid -> assert(found == 1); break
        :: else
        fi
    od
}

init {
    atomic {
        seed(watermark(0), 1);
        seed(watermark(1), 1);
        seed(map_slot(1), filed(key_b, 0));
        claimed[0] = 1;
        dispensed[0] = 1;
        claimed[1] = 1;
        dispensed[1] = 1;
        run compactor(0);
        run looker(1)
    }
}
