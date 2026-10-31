/**
 *  @file verification/dense_vram_index_claims.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The CUDA index's host side: the claim as one compare-exchange from the watermark, the flat
 *      matrices grown by the claim holder alone, the watermark each batch turns after its drain as a
 *      search's bound, the closed dispenser, and the admission ramp.
 *
 *  From `include/usearch/dense/vram_nvidia_index.cuh`: `claimed_rows_` holds the rows claimed, or
 *  `node_id_unknown_k` while closed, and a claim is one compare-exchange of it from the watermark
 *  to the watermark past the batch, so it succeeds only when no batch is in flight and the
 *  dispenser is open. Two inserters each submit one batch of two rows. `far` is not run: the posted
 *  add is x86's, and the ramp's add runs on the GPU.
 *
 *  - The bound a searcher acquired lies inside the flat matrices it sees mapped.
 *  - Every row below the bound is written, and a row the walk reads was written in a batch that turned.
 *  - One claim holder and so one grower run at a time, and the claimed ranges are disjoint.
 *  - Every batch and load ends, and every launch lies inside the flat matrices.
 *  - The claim ends open and level with the watermark unless a failure was kept.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 watermark_order=order_relaxed: relaxed, the searcher reads a bound past their
 *      extent
 *  @verify fail rc11 size_order=order_relaxed: relaxed, the searcher reads a bound past the flat
 *      matrices' extent
 *  @verify fail sc quiescent_claim=false: compared against the claim it last read, two batches claim
 *      side by side and grow the flat matrices at once
 *  @verify fail sc unclaim_on_rejection=false: kept, the claim stays past the watermark for good,
 *      and every later batch, load and closer is refused
 *  @verify fail sc unclaim_on_abandon=false: kept, the claim stays past the watermark for good
 *  @verify fail sc drain_before_turn=false: turned first, the searcher reads a row still in flight
 *  @verify fail sc neighbor_filter=false: unfiltered, it reads a row whose batch has not turned
 */
#define thread_count 5
#define location_count 6
#define history_depth 7
#define device_thread 4 // the thread every launch hands the host's view to
#include <weak_memory.pml>

// The words, by the index member each stands for; the flat matrices' extent stands for `capacity()`, and
// row 0's tail for the slot a back-link appends into, zero while it names no row.
#define claimed_rows 0
#define mapped_rows 1
#define published_rows 2
#define last_visible_id 3
#define row_zero_tail 4
#define status 5

// The statuses `status_` keeps: none, or the first failure that left claimed rows unpublished.
#define success 0
#define device_error 1

// The sizes, and the closed claim: the all-ones ID in silicon, past every count here.
#define rows_per_batch 2
#define rows_per_load 2
#define ramp_floor 1
#define claimable_rows 6
#define closed_rows 15

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef watermark_order
#define watermark_order order_release
#endif
#ifndef size_order
#define size_order order_acquire
#endif
#ifndef quiescent_claim
#define quiescent_claim true
#endif
#ifndef unclaim_on_rejection
#define unclaim_on_rejection true
#endif
#ifndef unclaim_on_abandon
#define unclaim_on_abandon true
#endif
#ifndef drain_before_turn
#define drain_before_turn true
#endif
#ifndef neighbor_filter
#define neighbor_filter true
#endif

byte done;          // the inserters and the loader, once finished
byte growing;       // threads inside grow_rows_
byte holders;       // threads holding a claim or the closed dispenser
bool claimed[3];    // per inserter and the loader, whether it holds a range
int range_first[3]; // per inserter and the loader, its range's first ID
bool pending[2];    // per inserter, an accepted batch the device has rows left to serve
byte served[2];     // per inserter, the rows of its batch the device published
int refused[2];     // per inserter, the count its ramp last refused a row on; -1 when none
bool filled[claimable_rows]; // per row, its payload and links written, by the device or the loader

// grow_rows_: the flat matrices' extent through `capacity()`, an acquire, committed and sentinel-filled
// when short, and a stream synchronize before the launch
inline grow(t, rows, mapped) {
    atomic {
        growing++;
        assert(memory != sc || growing == 1);
        load(t, mapped_rows, order_acquire, mapped)
    };
    atomic {
        if
        :: mapped < rows -> store(t, mapped_rows, order_release, rows); mapped = rows // the flat matrix releases its extent
        :: else
        fi;
        growing--
    }
}

// a range of `rows` IDs from `first` taken by `who`: apart from every range held, one holder at a time
inline take_range(who, first, rows) {
    assert(memory != sc || holders == 0);
    assert(!claimed[0] || first >= range_first[0] + rows_per_batch || first + rows <= range_first[0]);
    assert(!claimed[1] || first >= range_first[1] + rows_per_batch || first + rows <= range_first[1]);
    assert(!claimed[2] || first >= range_first[2] + rows_per_load || first + rows <= range_first[2]);
    holders++;
    claimed[who] = true;
    range_first[who] = first
}

// a range given back, by its un-claim or its turn
inline drop_range(who) {
    holders--;
    claimed[who] = false
}

// publish_: a wait until the watermark reaches `from`, then the release store of `to`, which gives
// the range of `who` back; a failure an earlier batch kept ends the wait first, since that batch
// never turns
inline publish(t, who, from, to, watermark, kept) {
    do
    :: load(t, last_visible_id, order_acquire, watermark);
        if
        :: watermark == from ->
            atomic {
                store(t, last_visible_id, watermark_order, to);
                drop_range(who)
            };
            break
        :: else ->
            load(t, status, order_acquire, kept);
            if
            :: kept != success -> break
            :: else -> (newest_value(last_visible_id) != watermark || newest_value(status) != success)
            fi
        fi
    od
}

proctype inserter(byte t) {
    bool accepted, succeeded;
    int expected, first_id, total, mapped, watermark, kept;
    // the claim: from the watermark acquired, or from the claim as last read, never from a closed one
    if
    :: quiescent_claim -> load(t, last_visible_id, order_acquire, first_id)
    :: else -> load(t, claimed_rows, order_relaxed, first_id)
    fi;
    if
    :: first_id == closed_rows -> goto finished
    :: else
    fi;
    total = first_id + rows_per_batch;
    expected = first_id;
    atomic {
        compare_exchange(t, claimed_rows, order_acq_rel, expected, total, succeeded);
        if
        :: succeeded -> take_range(t, first_id, rows_per_batch)
        :: else
        fi
    };
    if
    :: !succeeded -> goto finished
    :: else
    fi;
    grow(t, total, mapped);
    // the launch: everything the host thread wrote or acquired is visible to the grid it launches
    atomic {
        assert(newest_value(mapped_rows) >= total);
        merge_view(t, device_thread)
    };
    // the launch's verdict: accepted, the grid publishes; rejected, the claim is stored back
    if
    :: true ->
        atomic {
            pending[t] = true;
            refused[t] = -1;
            accepted = true
        }
    :: true ->
        if
        :: unclaim_on_rejection ->
            atomic {
                store(t, claimed_rows, order_release, first_id);
                drop_range(t)
            }
        :: else
        fi
    fi;
    // the blocking insert: the drain, every row written, then the turn; the first inserter's drain may
    // fail instead, keeping its failure and leaving its rows invisible
    if
    :: accepted && drain_before_turn ->
        (!pending[t]);
        if
        :: true -> publish(t, t + 0, first_id, total, watermark, kept)
        :: t == 0 -> read_modify_write(t, status, order_acq_rel, kept, device_error)
        fi
    :: accepted && !drain_before_turn ->
        publish(t, t + 0, first_id, total, watermark, kept);
        (!pending[t])
    :: else
    fi;
finished:
    done++
}

// plan_load and finish: on an open dispenser the rows claimed like a batch, or the dispenser closed
// as the sharded index does; then the growth, the rows written, the restored count, the turn, and the
// reopening of a closed dispenser; or the plan dies before `finish` and gives back what it took
proctype loader(byte t) {
    bool closing, succeeded, reopened;
    int expected, base, total, mapped, watermark, kept;
    load(t, last_visible_id, order_acquire, base);
    total = base + rows_per_load;
    expected = base;
    atomic {
        if
        :: closing = false; compare_exchange(t, claimed_rows, order_acq_rel, expected, total, succeeded)
        :: closing = true; compare_exchange(t, claimed_rows, order_acq_rel, expected, closed_rows, succeeded)
        fi;
        if
        :: succeeded -> take_range(2, base, rows_per_load)
        :: else
        fi
    };
    if
    :: !succeeded -> goto finished
    :: else
    fi;
    grow(t, total, mapped);
    if
    :: true ->
        atomic {
            filled[base] = true;
            filled[base + 1] = true
        };
        store(t, published_rows, order_relaxed, total);
        publish(t, 2, base, total, watermark, kept)
    :: closing -> drop_range(2)
    :: !closing && unclaim_on_abandon ->
        atomic {
            store(t, claimed_rows, order_release, base);
            drop_range(2)
        }
    :: !closing && !unclaim_on_abandon -> goto finished
    fi;
    if
    :: closing ->
        load(t, last_visible_id, order_acquire, watermark);
        expected = closed_rows;
        compare_exchange(t, claimed_rows, order_acq_rel, expected, watermark, reopened)
    :: else
    fi;
finished:
    done++
}

// submit_search_: the bound is size(), every row below it is read, and the walk follows row 0's
// tail, which a back-link of any batch may name, only below the bound; a row the walk reads was written
// in a batch that turned
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
        if
        :: id != 0 && (id < bound || !neighbor_filter) -> assert(filled[id] && id < newest_value(last_visible_id))
        :: else
        fi
    :: else
    fi
}

// one row of batch b through the admission ramp: refused, the count it read is remembered so the
// retry waits for the count to move, which is what the `__nanosleep` loop amounts to; served, the
// row is written and may append itself into row 0's tail while the slot is empty, whichever batch
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
    :: (done == 3 && !pending[0] && !pending[1]) -> break
    od
}

// once everyone is done: the claim open and level with the watermark, unless a failure was kept
proctype checker() {
    (done == 3);
    assert(memory != sc || newest_value(status) != success ||
           newest_value(claimed_rows) == newest_value(last_visible_id))
}

init { atomic { run inserter(0); run inserter(1); run searcher(2); run loader(3); run device(device_thread); run checker() } }
