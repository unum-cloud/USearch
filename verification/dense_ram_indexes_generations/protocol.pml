/**
 *  The shard set's generations from `include/usearch/utilities/ram_shards.hpp`: once the open
 *  generation's shards reach the bound, `open_generation_` appends one shard per placement, and only
 *  the newest generation takes insertions. One caller opens it: a compare-exchange takes
 *  `shards_count_` from the generation's first shard to the same count with the opening bit set,
 *  the caller builds the shards, and a release store publishes the count past them, or, when a build
 *  fails, puts the count back. A caller refused by the bit waits on the word and tries again; one
 *  that finds the count already past the generation's first shard answers success, its failed
 *  compare-exchange acquiring the store that moved it. `size()` acquires the count and masks the
 *  bit, so every shard it counts is whole.
 *
 *  One placement, so a generation is one shard of `row_limit` rows, and the model holds the first
 *  generation and the next. A shard's construction is one word, written relaxed before the count's
 *  release and read relaxed by every thread that touches the shard; its claim word and dispenser
 *  are ghosts, each step on them one atomic step, `claim_capacity` seen from outside. The opener's
 *  wait is the one `wait_type_` makes with no tag, capped: the spin waiter `ram_indexes` defaults to
 *  returns at once, and ForkUnion's monitored waiters arm the count's line, re-check it, and wake on
 *  any store to it or at the cap.
 *
 *  Every scenario beside this file checks that each generation is built once, that every shard a
 *  thread touches is constructed, and that no opener waits forever.
 */
#include "../weak_memory.pml"
#include "../watermark_turn.pml"

// The sizes: the shards the model holds, and the rows one shard takes under the bound.
#define shards 2
#define row_limit 2

// The words, by the index member each stands for; the last is `commit_sequence_`.
#define shards_count 0
#define built(shard) (1 + (shard))
#define last_visible_id(shard) (3 + (shard))
#define sequence_word 5

// The values: the count's opening bit, above every count the model reaches, and the counted
// sequence's carry, which opening a bracket adds one toward and closing it the unit less one.
#define opening 8
#define counted(word) ((word) % opening)
#define sequence_unit 4

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef opening_bit
#define opening_bit true
#endif
#ifndef publish_order
#define publish_order order_release
#endif
#ifndef failure_order
#define failure_order order_acquire
#endif
#ifndef restore_count
#define restore_count true
#endif
#ifndef opening_cap
#define opening_cap true
#endif
#ifndef count_wakes
#define count_wakes true
#endif
#ifndef count_inside_sequence
#define count_inside_sequence true
#endif

byte claimed[shards];     // each shard's claim word, a ghost
byte dispensed[shards];   // each shard's `next_dispensed_id_`, a ghost, since no plan reads it
bool constructed[shards]; // a builder holds the shard constructed, so a second build is caught
byte done;                // the adders, once finished
bool batch_dispensed;     // the planned batch's ranges are dispensed
byte batch_share[shards]; // the rows the batch lands on each shard
byte batch_part[shards];  // the ID its range starts at on each shard

// The inlines below name their callers' locals: `count`, `seen`, `won`, `granted`, `opened`,
// `shard`, `split`, `id`, `watermark` and `sequence`.

// `size()`: the count acquired, the opening bit masked
inline published_shards(t) {
    load(t, shards_count, order_acquire, count);
    count = counted(count)
}

// any access to a shard's members, which reads its construction
inline touch(t, shard) {
    load(t, built(shard), order_relaxed, seen);
    assert(seen == 1)
}

// the shard's `claim(rows)`: granted while the rows fit under the bound, else `overflow_k`
inline claim(shard, rows, granted) {
    atomic {
        granted = claimed[shard] + rows <= row_limit;
        if
        :: granted -> claimed[shard] = claimed[shard] + rows
        :: else
        fi
    }
}

// a store to the count, waking every opener armed on its line, as a monitored line does
inline store_count(t, value) {
    atomic {
        store(t, shards_count, publish_order, value);
        if
        :: count_wakes -> wake(shards_count)
        :: else
        fi
    }
}

// `open_generation_` for the generation from shard @p first: the opening bit taken by one
// compare-exchange, the count found already past @p first, or the bit's holder waited out and the
// exchange tried again; then the build, which may fail when @p may_fail, and the count's store.
// @p opened says whether the generation is open on return
inline open_generation(t, first, may_fail, opened) {
    do
    :: seen = first;
        if
        :: opening_bit ->
            atomic {
                compare_exchange(t, shards_count, order_acquire, seen, first + opening, won);
                if
                :: !won && acquires(failure_order) -> load(t, shards_count, order_acquire, seen)
                :: else
                fi
            }
        :: else -> load(t, shards_count, order_acquire, seen); won = seen == first
        fi;
        if
        :: won || counted(seen) != first -> break
        :: else -> wait_on(t, shards_count, seen, opening_cap && newest_value(shards_count) != seen)
        fi
    od;
    opened = true;
    if
    :: won && may_fail -> opened = false
    :: skip
    fi;
    if
    :: won && opened ->
        atomic { assert(!constructed[first]); constructed[first] = true };
        store(t, built(first), order_relaxed, 1);
        store_count(t, first + 1)
    :: won && !opened && restore_count -> store_count(t, first)
    :: else
    fi
}

// the single-row `insert`: the open generation's shard probed, the next generation opened when it
// is full and probed in turn, a refused build refusing the insert; then the shard's own `insert`,
// whose dispense consumes the claim and whose turn publishes the row
inline insert_row(t, may_fail) {
    do
    :: published_shards(t);
        shard = count - 1;
        touch(t, shard);
        claim(shard, 1, granted);
        if
        :: granted -> break
        :: else ->
            open_generation(t, count, may_fail, opened);
            if
            :: !opened -> break
            :: else
            fi
        fi
    od;
    if
    :: granted ->
        atomic { id = dispensed[shard]; dispensed[shard]++ };
        take_turn(t, last_visible_id(shard), id, id + 1, order_release, watermark)
    :: else
    fi;
    done++
}

// `plan_insert` of @p rows, then its `insert` and `finish`: `room()` read off the open generation's
// shard, the next generation opened for the rows past it, and both shards' claims in placement order
// right after `open_generation_` answers, with no `size()` between, a refusal releasing the claim
// granted; then both ranges dispensed in one step under the dispensing mutex, every turn awaited,
// and the watermarks stored inside the bracket of the commit sequence word
inline plan_batch(t, rows) {
    published_shards(t);
    touch(t, count - 1);
    atomic {
        split = row_limit - claimed[count - 1];
        if
        :: split > rows -> split = rows
        :: else
        fi;
        batch_share[count - 1] = split;
        if
        :: split < rows -> batch_share[count] = rows - split
        :: else
        fi
    };
    if
    :: split < rows -> open_generation(t, count, false, opened)
    :: else
    fi;
    granted = true;
    shard = count - 1;
    do
    :: shard == count + (split < rows -> 1 : 0) || !granted -> break
    :: else ->
        touch(t, shard);
        claim(shard, batch_share[shard], granted);
        if
        :: !granted && shard == count -> atomic { claimed[count - 1] = claimed[count - 1] - split }
        :: else
        fi;
        shard++
    od;
    if
    :: granted ->
        atomic {
            for (shard : 0 .. shards - 1) {
                batch_part[shard] = dispensed[shard];
                dispensed[shard] = dispensed[shard] + batch_share[shard]
            };
            batch_dispensed = true
        };
        for (shard : 0 .. shards - 1) {
            if
            :: batch_share[shard] > 0 -> await_turn(t, last_visible_id(shard), batch_part[shard], watermark)
            :: else
            fi
        };
        read_modify_write(t, sequence_word, order_release, sequence, sequence + 1);
        for (shard : 0 .. shards - 1) {
            if
            :: batch_share[shard] > 0 ->
                end_turn(t, last_visible_id(shard), order_release, batch_part[shard] + batch_share[shard])
            :: else
            fi
        };
        read_modify_write(t, sequence_word, order_release, sequence, sequence + sequence_unit - 1)
    :: else
    fi;
    done++
}
