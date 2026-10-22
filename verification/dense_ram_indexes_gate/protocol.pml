/**
 *  The sharded index's gate and epoch from `include/usearch/utilities/ram_shards.hpp` and
 *  `ram_indexes.hpp`: every mutator holds the gate's shared side for one call, every destructive
 *  window takes its unique side with `try_lock` and never waits, and every reader brackets its call
 *  with two loads of the epoch word instead of touching the gate. The gate is `basic::shared_mutex`,
 *  moved unchanged: the shared side is one acquire add of a holder and a generation step, backed off
 *  behind a unique holder, then a release fence; `unlock_shared` is a release subtract.
 *
 *  A destructive window, under the unique side, closes every shard it empties through the
 *  claim-aware `close`, makes the epoch odd with a relaxed add followed by a release fence, rewrites
 *  in place, keeping every byte a reader may reach allocated, and makes the epoch even with a
 *  release add. A reader acquires an even epoch, waiting out an odd one through the capped waiter,
 *  reads, issues an acquire fence, and re-reads the epoch relaxed; a moved epoch retries once, then
 *  answers `contended_k`. `close` moves no epoch: it changes only dispensers, which no reader reads.
 *
 *  A shard's claim word, dispenser and closed bit are ghosts, each step on them one atomic step,
 *  `claim_capacity` and `close_` seen from outside; its watermark is a word. A dispense asserts
 *  that a claim still covers it, the invariant `claim_capacity` keeps: claims die only before the
 *  dispense.
 */
#include "../weak_memory.pml"
#include "../watermark_turn.pml"
#include "../basic_gate.pml"

// The sizes: the shards a scenario may hold, and the rows one shard takes under the bound.
#define shards 2
#define row_limit 1

// The words every scenario shares, by the member each stands for; a scenario numbers its own from 5.
#define gate 0
#define epoch 1
#define shards_count 2
#define watermark(shard) (3 + (shard))

// The values: the count's opening and sealed bits, above every count the model reaches.
#define opening 4
#define sealed 8
#define counted(word) ((word) % opening)

// The knobs' values: a mutator's hold, when a plan closes its donors, and how in-place compaction
// rewrites the locators of the rows it moved.
#define none 0
#define shared 1
#define at_plan 0
#define at_run 1
#define replace 0
#define rebuild 1

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef claims_check
#define claims_check true
#endif
#ifndef insert_hold
#define insert_hold shared
#endif
#ifndef seal
#define seal true
#endif
#ifndef donor_close
#define donor_close at_plan
#endif
#ifndef rewrite
#define rewrite replace
#endif
#ifndef count_bracket
#define count_bracket true
#endif
#ifndef revalidate
#define revalidate true
#endif
#ifndef free_in_window
#define free_in_window false
#endif
#ifndef window_fence
#define window_fence true
#endif
#ifndef exit_fence
#define exit_fence true
#endif
#ifndef epoch_order
#define epoch_order order_acq_rel
#endif

byte claimed[shards];   // each shard's claim word, a ghost
byte dispensed[shards]; // each shard's `next_dispensed_id_`, a ghost
bool closed[shards];    // each shard's dispenser closed, a ghost

// The inlines below name their callers' locals: `shard`, `seen`, `count`, `won`, `observed` and
// `succeeded`.

// `lock_shared`: one acquire add of a holder and a generation step, handed back behind a unique
// holder and retried once the holder left, then the release fence. The back-off blocks on the
// unique bit as `basic_shared_mutex.pml` does, which checks its capped wait: a wait woken by every
// store would let two backed-off mutators hand the word back and forth without bound
inline lock_shared(t, observed) {
    do
    :: read_modify_write(t, gate, order_acquire, observed, observed + generation_unit + 1);
        if
        :: !unique_of(observed) -> break
        :: else ->
            read_modify_write(t, gate, order_release, observed, observed - 1);
            (!unique_of(newest_value(gate)))
        fi
    od;
    fence_release(t)
}

// `unlock_shared`: one release subtract
inline unlock_shared(t, observed) { read_modify_write(t, gate, order_release, observed, observed - 1) }

// a mutator's call: the shared side held throughout, or nothing under `insert_hold=none`
inline hold_shared(t, observed) {
    if
    :: insert_hold == shared -> lock_shared(t, observed)
    :: else
    fi
}
inline release_shared(t, observed) {
    if
    :: insert_hold == shared -> unlock_shared(t, observed)
    :: else
    fi
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

// a claim given back, by a plan that dies before its dispense
inline release_claim(shard, rows) { atomic { claimed[shard] = claimed[shard] - rows } }

// one ID dispensed, refused while the dispenser is closed; a claim must still cover it
inline dispense(shard, id, refused) {
    atomic {
        refused = closed[shard];
        if
        :: !refused ->
            id = dispensed[shard];
            dispensed[shard]++;
            assert(dispensed[shard] <= claimed[shard])
        :: else
        fi
    }
}

// `shard_set::close` under the unique side: the first @p count shards' dispensers closed in
// placement order, each refused while a batch sits between its dispense and its turn, as `close_`
// checks, or, under `claims_check`, while it holds claims it has not dispensed; a refusal reopens
// the shards closed before it, the last first
inline close_shards(t, count, refused) {
    refused = false;
    shard = 0;
    do
    :: shard == count || refused -> break
    :: else ->
        load(t, watermark(shard), order_acquire, seen);
        atomic {
            refused = dispensed[shard] != seen || (claims_check && claimed[shard] != dispensed[shard]);
            if
            :: !refused -> closed[shard] = true; shard++
            :: else
            fi
        }
    od;
    if
    :: refused ->
        do
        :: shard == 0 -> break
        :: else -> shard--; closed[shard] = false
        od
    :: else
    fi
}

// `close_gated_`: the unique side taken or refused, every shard the count publishes closed through
// the claim-aware `close`, and the count sealed on success
inline close_gated(t, refused) {
    try_lock(t, gate, observed, succeeded);
    refused = !succeeded;
    if
    :: succeeded ->
        load(t, shards_count, order_acquire, count);
        close_shards(t, counted(count), refused);
        if
        :: !refused && seal -> read_modify_write(t, shards_count, order_relaxed, seen, seen | sealed)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

// `open_generation_` for the generation from shard @p first: the opening bit taken by one
// compare-exchange, whose refusal acquires the word; a sealed count refuses, a count past @p first
// answers success, and a held bit is waited out; the opener publishes the count past the new shard
inline open_generation(t, first, refused) {
    refused = false;
    do
    :: seen = first;
        atomic {
            compare_exchange(t, shards_count, order_acquire, seen, first + opening, won);
            if
            :: !won -> load(t, shards_count, order_acquire, seen)
            :: else
            fi
        };
        if
        :: won -> break
        :: !won && (seen & sealed) -> refused = true; break
        :: !won && !(seen & sealed) && counted(seen) != first -> break
        :: !won && !(seen & sealed) && counted(seen) == first ->
            wait_on(t, shards_count, seen, newest_value(shards_count) != seen)
        fi
    od;
    if
    :: won ->
        atomic {
            store(t, shards_count, order_release, first + 1);
            wake(shards_count)
        }
    :: else
    fi
}

// `reopen`: the first @p count shards' dispensers reopened, the last first
inline reopen_shards(count) {
    shard = count;
    do
    :: shard == 0 -> break
    :: else -> shard--; closed[shard] = false
    od
}

// `forget_`: the shard emptied in place, its memory kept, inside a window
inline forget(t, shard) {
    atomic { claimed[shard] = 0; dispensed[shard] = 0 };
    store(t, watermark(shard), order_relaxed, 0)
}

// `begin_window`, under the unique side: the epoch odd by a relaxed add, then the release fence
// that orders every rewrite after it
inline begin_window(t, observed) {
    read_modify_write(t, epoch, order_relaxed, observed, observed + 1);
    if
    :: window_fence -> fence_release(t)
    :: else
    fi
}

// `end_window`: the epoch even by a release add, waking the readers waiting out the odd one
inline end_window(t, observed) {
    atomic {
        read_modify_write(t, epoch, epoch_order & order_release, observed, observed + 1);
        wake(epoch)
    }
}

// `read_epoch`: the epoch acquired, an odd one waited out through the capped waiter
inline read_epoch(t, epoch_seen) {
    do
    :: load(t, epoch, epoch_order & order_acquire, epoch_seen);
        if
        :: epoch_seen % 2 == 0 -> break
        :: else -> wait_on(t, epoch, epoch_seen, newest_value(epoch) != epoch_seen)
        fi
    od
}

// `still_valid`: the acquire fence, then the epoch re-read relaxed and compared
inline still_valid(t, epoch_seen, again, valid) {
    if
    :: exit_fence -> fence_acquire(t)
    :: else
    fi;
    load(t, epoch, order_relaxed, again);
    valid = again == epoch_seen
}

// a single-row `insert` into a donor, holding its gate shared: a claim on the open generation's
// shard, or, refused for room, the next generation opened and claimed in turn; a closed dispenser
// refuses it, giving the claim back, and a sealed count refuses the opening. Its callers' locals
// are `observed`, `count`, `seen`, `watermark`, `shard`, `id`, `granted`, `won` and `refused`
inline insert_row(t) {
    hold_shared(t, observed);
    do
    :: load(t, shards_count, order_acquire, count);
        shard = counted(count) - 1;
        claim(shard, 1, granted);
        if
        :: granted -> break
        :: else ->
            open_generation(t, shard + 1, refused);
            if
            :: refused -> break
            :: else
            fi
        fi
    od;
    if
    :: granted ->
        dispense(shard, id, refused);
        if
        :: refused -> release_claim(shard, 1)
        :: else -> take_turn(t, watermark(shard), id, id + 1, order_release, watermark)
        fi
    :: else
    fi;
    release_shared(t, observed)
}

// `plan_absorb` and its `run` for one donor: the donor closed at plan time or at `run`, the
// landings packed against its count and watermarks, and at `run` every row it holds one they
// packed. Its callers' locals are `observed`, `count`, `seen`, `shard`, `succeeded`, `refused`,
// `packed_count` and `packed_rows`
inline absorb(t) {
    if
    :: donor_close == at_plan -> close_gated(t, refused)
    :: else -> refused = false
    fi;
    if
    :: !refused ->
        load(t, shards_count, order_acquire, packed_count);
        packed_count = counted(packed_count);
        for (shard : 0 .. shards - 1) {
            if
            :: shard < packed_count -> load(t, watermark(shard), order_acquire, packed_rows[shard])
            :: else -> packed_rows[shard] = 0
            fi
        };
        if
        :: donor_close == at_run -> close_gated(t, refused)
        :: else
        fi;
        if
        :: !refused ->
            atomic {
                assert(counted(newest_value(shards_count)) == packed_count);
                for (shard : 0 .. shards - 1) {
                    assert(newest_value(watermark(shard)) == packed_rows[shard])
                }
            }
        :: else
        fi
    :: else
    fi
}
