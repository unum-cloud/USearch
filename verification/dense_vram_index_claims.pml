/**
 *  @brief The CUDA index's host side: the claim under its spin mutex, the planes grown before it,
 *      the watermark each batch turns after its drain as a search's bound, and the admission ramp.
 *
 *  From `include/usearch/dense/vram_index.cuh`: one claim per batch under `claiming_`, a
 *  `fu::spin_mutex_t`, IDs dense from `claimed_rows_`, the planes grown under the mutex and
 *  sentinel-filled before any launch, the claim stored only once the launch is accepted, the
 *  device's admission ramp on `published_rows`, a counter in device memory no host thread
 *  dereferences, and the watermark `last_visible_id_` turned once per batch.
 *
 *  Two inserters each submit one batch of two rows: the mutex, a relaxed read of the claim,
 *  `grow_rows_` when the planes are short, the launch, which carries the host's view to the device
 *  and is accepted or rejected, the release store of the claim on acceptance alone, the release
 *  unlock; then an accepted batch drains and takes its turn, `publish_`: a wait until the watermark
 *  reaches its first ID, then a release store of its end. A loader runs `plan_load` and `finish`:
 *  `lock_if_free`, `contended_k` when the claim is held, else the growth, the rows landed, the
 *  relaxed restore of `published_rows`, the release store of the claim, the unlock, and the turn;
 *  or it dies before `finish`. A searcher takes its bound from `size()`, an acquire load of the
 *  watermark, reads the planes' extent the way `submit_search_`'s kernel touches them, reads every
 *  row below the bound, and walks row 0's tail. The device serves the accepted batches one row at
 *  a time in any order, each row through the ramp: a relaxed load of `published_rows`, a wait
 *  while `id >= published + max(published, floor)`, one relaxed no-return add, and maybe a
 *  back-link appending the row into row 0's tail slot while it is empty, whatever batch row 0
 *  belongs to. The floor is one and a batch is two rows, so a batch served ahead of the one claimed
 *  before it waits on that one's adds. The planes' extent is `capacity()`, which the header derives
 *  from the planes' committed bytes and reads outside the claim: the one grower release-stores
 *  each plane's extent and `committed_bytes()` acquires it; the model folds the three planes into
 *  one word with the same orders.
 *
 *  The bound a searcher acquired lies inside the planes it sees mapped, and so does every ID the
 *  device is assigned. Every row below the bound landed. One grower and one claim holder run at a
 *  time, the loader's verdict names the claim's state, and the claimed ranges are disjoint and lie
 *  inside the planes when each claim is stored. `far` is not run: the posted add is x86's, and the
 *  ramp's add runs on the GPU.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: the turn's release store pairs with the acquire
 *      in `size()`, so a search's bound lies inside the planes the claim grew; relaxed, the searcher
 *      reads a bound past the planes' extent
 *  @verify fail rc11 size_order=order_relaxed: `size()` acquires the watermark the turn released;
 *      relaxed, the searcher reads a bound past the planes' extent
 *  @verify fail sc claiming_mutex=false: `claiming_` keeps the growers and the claims apart;
 *      without it, two batches claim the same IDs
 *  @verify stuck sc unclaim_on_rejection=false: the claim is stored only once the launch is
 *      accepted, so a rejected grid claims nothing; stored either way, the next batch's ramp waits
 *      for rows nobody publishes
 *  @verify fail sc base_under_claim=false: `plan_load` reads its base under the claim; read from
 *      `size()` before `lock_if_free`, a batch claimed between the two lines overlaps the loaded
 *      rows, and `finish` could lower the claim beneath it
 *  @verify stuck sc unlock_on_abandon=false: the load plan spends like its host twin, and one that
 *      dies before `finish` drops the claim mutex it holds; kept, the inserters spin behind it
 *      forever
 *  @verify fail sc drain_before_turn=false: a batch turns the watermark only after its drain and
 *      in claim order, so every row under a search's bound landed; turned first, the searcher reads
 *      a row still in flight
 *  @verify fail sc neighbor_filter=false: the walk follows no ID at or past the bound, since an
 *      in-flight batch's back-links name its rows inside visible ones; unfiltered, it reaches a row
 *      whose batch has not turned
 */
#define thread_count 5
#define location_count 6
#define history_depth 10
#define device_thread 4 // the thread every launch hands the host's view to
#include "weak_memory.pml"
#include "watermark_turn.pml"

// The words, by the index member each stands for; the planes' extent stands for `capacity()`, and
// row 0's tail for the slot a back-link appends into, zero while it names no row.
#define claiming 0
#define claimed_rows 1
#define mapped_rows 2
#define published_rows 3
#define last_visible_id 4
#define row_zero_tail 5

// The sizes.
#define rows_per_batch 2
#define rows_per_load 2
#define ramp_floor 1
#define claimable_rows 6

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef watermark_order
#define watermark_order order_release
#endif
#ifndef size_order
#define size_order order_acquire
#endif
#ifndef claiming_mutex
#define claiming_mutex true
#endif
#ifndef unclaim_on_rejection
#define unclaim_on_rejection true
#endif
#ifndef base_under_claim
#define base_under_claim true
#endif
#ifndef unlock_on_abandon
#define unlock_on_abandon true
#endif
#ifndef drain_before_turn
#define drain_before_turn true
#endif
#ifndef neighbor_filter
#define neighbor_filter true
#endif

byte inserters_done;
byte growing;       // threads inside grow_rows_
byte claim_holders; // threads holding claiming_
byte contended;     // loads refused by a held claim
bool claimed[3];    // per inserter and the loader, whether its range was claimed
int range_first[3]; // per inserter and the loader, its range's first ID
bool pending[2];    // per inserter, an accepted batch the device has rows left to serve
byte served[2];     // per inserter, the rows of its batch the device published
int refused[2];     // per inserter, the count its ramp last refused a row on; -1 when none
bool filled[claimable_rows]; // per row, its payload and links written, by the device or the loader

// spin_mutex::lock: a non-writing spin until the word looks free, then the acquire exchange
inline lock_claiming(t, observed) {
    do
    :: (newest_value(claiming) == 0);
       atomic {
           read_modify_write(t, claiming, order_acquire, observed, 1);
           if
           :: observed == 0 ->
               claim_holders++;
               assert(memory != sc || claim_holders == 1)
           :: else
           fi
       };
       if
       :: observed == 0 -> break
       :: else
       fi
    od
}
inline unlock_claiming(t) {
    atomic {
        claim_holders--;
        store(t, claiming, order_release, 0)
    }
}

// grow_rows_: the planes' extent through `capacity()`, an acquire, committed and sentinel-filled
// when short, and a stream synchronize before the claim may advance
inline grow(t, rows, mapped) {
    atomic {
        growing++;
        assert(memory != sc || growing == 1);
        load(t, mapped_rows, order_acquire, mapped)
    };
    atomic {
        if
        :: mapped < rows -> store(t, mapped_rows, order_release, rows); mapped = rows // the plane releases its extent
        :: else
        fi;
        growing--
    }
}

// a claim of `rows` IDs from `first`: inside the planes, and apart from every range claimed before
inline claim_range(who, first, rows) {
    assert(newest_value(mapped_rows) >= first + rows);
    assert(!claimed[0] || first >= range_first[0] + rows_per_batch || first + rows <= range_first[0]);
    assert(!claimed[1] || first >= range_first[1] + rows_per_batch || first + rows <= range_first[1]);
    assert(!claimed[2] || first >= range_first[2] + rows_per_load || first + rows <= range_first[2]);
    claimed[who] = true;
    range_first[who] = first
}

proctype inserter(byte t) {
    bool accepted;
    int observed, first_id, total, mapped, watermark;
    if
    :: claiming_mutex -> lock_claiming(t, observed)
    :: else
    fi;
    atomic {
        load(t, claimed_rows, order_relaxed, first_id);
        total = first_id + rows_per_batch
    };
    grow(t, total, mapped);
    // the launch: everything the host thread wrote or acquired is visible to the grid it launches
    atomic { merge_view(t, device_thread) };
    // the launch's verdict: accepted, the claim is stored and the grid publishes; rejected, it is un-claimed exactly
    if
    :: true ->
        atomic {
            claim_range(t, first_id, rows_per_batch);
            store(t, claimed_rows, order_release, total);
            pending[t] = true;
            refused[t] = -1;
            accepted = true
        }
    :: true ->
        if
        :: !unclaim_on_rejection ->
            atomic {
                claim_range(t, first_id, rows_per_batch);
                store(t, claimed_rows, order_release, total)
            }
        :: else
        fi
    fi;
    if
    :: claiming_mutex -> unlock_claiming(t)
    :: else
    fi;
    // the blocking insert: the drain, every row landed, then the turn
    if
    :: accepted ->
        if
        :: drain_before_turn ->
            (!pending[t]);
            take_turn(t, last_visible_id, first_id, total, watermark_order, watermark)
        :: else ->
            take_turn(t, last_visible_id, first_id, total, watermark_order, watermark);
            (!pending[t])
        fi
    :: else
    fi;
    inserters_done++
}

// plan_load and finish: the claim if free, the growth, the rows landed, the restored count, the
// claim's release, and the turn once the mutex is dropped
proctype loader(byte t) {
    int observed, base, total, mapped, watermark;
    if
    :: !base_under_claim -> load(t, claimed_rows, order_acquire, base)
    :: else
    fi;
    atomic {
        read_modify_write(t, claiming, order_acquire, observed, 1);
        if
        :: observed != 0 ->
            contended++;
            assert(memory != sc || claim_holders == 1)
        :: else ->
            assert(memory != sc || claim_holders == 0);
            claim_holders++
        fi
    };
    if
    :: observed != 0
    :: else ->
        atomic {
            if
            :: base_under_claim -> load(t, claimed_rows, order_acquire, base)
            :: else
            fi;
            total = base + rows_per_load
        };
        grow(t, total, mapped);
        // the plan's verdict: finished, the rows are claimed and restored; dead before `finish`,
        // the planes stay grown and nothing is claimed, so the next batch reuses the base
        if
        :: true ->
            atomic {
                claim_range(2, base, rows_per_load);
                filled[base] = true;
                filled[base + 1] = true
            };
            store(t, published_rows, order_relaxed, total);
            store(t, claimed_rows, order_release, total)
        :: true -> skip
        fi;
        if
        :: unlock_on_abandon -> unlock_claiming(t)
        :: else
        fi;
        if
        :: claimed[2] -> take_turn(t, last_visible_id, base, total, watermark_order, watermark)
        :: else
        fi
    fi
}

// submit_search_: the bound is size(), every row below it is read, and the walk follows row 0's
// tail, which a back-link of any batch may name, only below the bound
proctype searcher(byte t) {
    int bound, mapped, id;
    load(t, last_visible_id, size_order, bound);
    load(t, mapped_rows, order_acquire, mapped); // capacity() acquires it
    assert(bound <= mapped);
    atomic {
        for (id : 0 .. claimable_rows - 1) {
            assert(id >= bound || filled[id])
        }
    };
    if
    :: bound > 0 ->
        load(t, row_zero_tail, order_relaxed, id);
        assert(id == 0 || neighbor_filter && id >= bound || id < newest_value(last_visible_id))
    :: else
    fi
}

// one row of batch b through the admission ramp: refused, the count it read is remembered so the
// retry waits for the count to move, which is what the `__nanosleep` loop amounts to; served, the
// row lands and may append itself into row 0's tail while the slot is empty, whichever batch
// either row is in
inline serve_row(t, b, published, id, mapped) {
    atomic {
        (pending[b] && newest_value(published_rows) != refused[b]);
        id = range_first[b] + served[b];
        load(t, published_rows, order_relaxed, published);
        if
        :: id >= published + (published > ramp_floor -> published : ramp_floor) -> refused[b] = published
        :: else ->
            load(t, mapped_rows, order_relaxed, mapped);
            assert(id < mapped);
            filled[id] = true;
            add_no_return(t, published_rows, order_relaxed, 1);
            if
            :: id != 0 && filled[0] && newest_value(row_zero_tail) == 0 ->
                store(t, row_zero_tail, order_relaxed, id)
            :: true
            fi;
            served[b]++;
            refused[b] = -1;
            if
            :: served[b] == rows_per_batch -> pending[b] = false
            :: else
            fi
        fi
    }
}

// the grids of the accepted batches, their warps interleaved a row at a time
proctype device(byte t) {
    int published, id, mapped;
    do
    :: serve_row(t, 0, published, id, mapped)
    :: serve_row(t, 1, published, id, mapped)
    :: (inserters_done == 2 && !pending[0] && !pending[1]) -> break
    od
}

init { atomic { run inserter(0); run inserter(1); run searcher(2); run loader(3); run device(device_thread) } }
