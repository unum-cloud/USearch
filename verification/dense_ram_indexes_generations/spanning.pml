/**
 *  @brief A planned batch spanning the open generation's last room and a fresh generation, beside a
 *      single-row insert and a snapshot.
 *
 *  The first generation's shard has one row of room. The planner reads `room()`, opens the next
 *  generation for the row that does not fit, claims a row on each shard, dispenses both ranges in
 *  one step, and commits them inside the bracket of the commit sequence word. The single-row insert
 *  probes the first shard and, finding it full, opens the next generation and claims there. When
 *  the insert takes the last row first, the planner's claim on the first shard is refused, or its
 *  whole batch goes to the fresh one. A snapshot acquires the word, reads `size()` past it, touches
 *  and acquires every shard's watermark it counts, and retries when the word moved; it holds the
 *  batch on every shard it landed in or on none.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 count_inside_sequence=false: `snapshot` reads `size()` past the sequence
 *      word, so a commit the word shows landed counts every generation it landed in; reading the
 *      count first, the snapshot counts the first shard alone, then accepts a word the batch's
 *      commit already closed, and holds the batch's row on the old shard without the one on the
 *      fresh shard it never counted
 */
#define thread_count 3
#define location_count 6
#define history_depth 3
#include "protocol.pml"

proctype planner(byte t) {
    int count, seen, watermark, sequence;
    byte split, shard;
    bool won, granted, opened;
    plan_batch(t, 2)
}

proctype inserter(byte t) {
    int count, seen, id, watermark;
    byte shard;
    bool won, granted, opened;
    insert_row(t, false)
}

// `snapshot`: the word acquired and waited out while a bracket is open, `size()` past it, every
// counted shard touched and its watermark acquired, and a retry when the word moved
proctype reader(byte t) {
    int count, seen, before, after, watermarks[shards];
    byte shard;
retry:
    if
    :: !count_inside_sequence -> published_shards(t)
    :: else
    fi;
    load(t, sequence_word, order_acquire, before);
    if
    :: before % sequence_unit != 0 ->
        wait_on(t, sequence_word, before, newest_value(sequence_word) % sequence_unit == 0);
        goto retry
    :: else
    fi;
    if
    :: count_inside_sequence -> published_shards(t)
    :: else
    fi;
    for (shard : 0 .. shards - 1) {
        if
        :: shard < count -> touch(t, shard); load(t, last_visible_id(shard), order_acquire, watermarks[shard])
        :: else -> watermarks[shard] = 0
        fi
    };
    load(t, sequence_word, order_relaxed, after);
    if
    :: after != before -> goto retry
    :: else
    fi;
    for (shard : 1 .. shards - 1) {
        assert(!batch_dispensed || batch_share[0] == 0 || batch_share[shard] == 0 ||
               (watermarks[0] > batch_part[0]) == (watermarks[shard] > batch_part[shard]))
    }
}

// once both adders are done: the generation open, and every shard's claims landed
proctype checker() {
    byte shard;
    (done == 2);
    assert(newest_value(shards_count) == 2 && constructed[1]);
    for (shard : 0 .. shards - 1) {
        assert(claimed[shard] == dispensed[shard] && newest_value(last_visible_id(shard)) == dispensed[shard])
    }
}

init {
    atomic {
        seed(shards_count, 1);
        seed(built(0), 1);
        seed(last_visible_id(0), row_limit - 1);
        claimed[0] = row_limit - 1;
        dispensed[0] = row_limit - 1;
        constructed[0] = true;
        run planner(0);
        run inserter(1);
        run reader(2);
        run checker()
    }
}
