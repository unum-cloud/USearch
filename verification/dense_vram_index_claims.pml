/**
 *  The CUDA index's host side from `include/usearch/dense/vram_index.cuh`: one claim per batch
 *  under `claiming_`, a `fu::spin_mutex`, ids dense from `claimed_rows_`, the planes grown under
 *  the mutex and sentinel-filled before any launch, the claim stored only once the launch is
 *  accepted, and the device's admission ramp on `published_rows`, a counter in device memory
 *  no host thread dereferences.
 *
 *  Two inserters each submit one batch of two rows: the mutex, a relaxed read of the claim,
 *  `grow_rows_` when the planes are short, the launch, which carries the host's view to the
 *  device and is accepted or rejected, the release store of the claim on acceptance alone, the
 *  release unlock. A loader runs `plan_load` and `finish`: `lock_if_free`, `contended_k` when
 *  the claim is held, else the growth, the relaxed restore of `published_rows`, the release
 *  store of the claim, the unlock. A searcher takes its sampling bound from `size()`, an acquire
 *  load of the claim, and reads the planes' extent the way `submit_search_`'s kernel touches
 *  them. The device serves the accepted batches one row at a time in any order, each row
 *  through the ramp: a relaxed load of `published_rows`, a wait while
 *  `id >= published + max(published, floor)`, one relaxed no-return add. The floor is one and
 *  a batch is two rows, so a batch served ahead of the one claimed before it waits on that
 *  one's adds. The planes' extent is `capacity()`, which the header derives from the planes'
 *  committed bytes: the one grower release-stores each plane's extent and `committed_bytes()`
 *  acquires it; the model folds the three planes into one word with the same orders.
 *
 *  Invariants:
 *  - bound: the sampling bound a searcher acquired lies inside the planes it sees mapped, and
 *    every id the device is assigned lies inside the planes it sees mapped. The claim's release
 *    pairs with `size()`'s acquire for the first; `-Dwithout_claim_release` and
 *    `-Dwithout_size_acquire` each drop one side, and either fails under the view model;
 *  - exclusion: one grower at a time, one claim holder at a time, and the loader's verdict names
 *    the claim's state; the claimed ranges are disjoint and lie inside the planes when each claim
 *    is stored. `-Dwithout_claiming_mutex` skips the mutex, and two batches claim the same ids.
 *    `-Dwithout_base_under_claim` reads the load's base before `lock_if_free`, as `plan_load`
 *    did before its re-read under the claim, and a batch claimed in between makes the load's
 *    range overlap it;
 *  - progress: a rejected launch publishes nothing, so its ids are not claimed.
 *    `-Dwithout_unclaim_on_rejection` stores the claim either way, and the next batch's ramp
 *    waits for rows nobody publishes: an invalid end state.
 *  `-Dmemory=far` is not run: the posted add is x86's, and the ramp's add runs on the GPU.
 */
#include "weak_memory.pml"

// The threads, by role, apart from the processes that play them; the inserters name themselves as they start.
#define searcher_thread 2
#define loader_thread 3
#define device_thread 4

// The words, by the index member each stands for; the planes' extent stands for `capacity()`.
#define claiming 0
#define claimed_rows 1
#define mapped_rows 2
#define published_rows 3

// The scenario's sizes.
#define rows_per_batch 2
#define rows_per_load 2
#define ramp_floor 1

#ifdef without_claim_release
#define claim_order order_relaxed
#else
#define claim_order order_release
#endif
#ifdef without_size_acquire
#define size_order order_relaxed
#else
#define size_order order_acquire
#endif

byte inserters_started, inserters_done;
byte growing;       // threads inside grow_rows_
byte claim_holders; // threads holding claiming_
byte contended;     // loads refused by a held claim
bool claimed[3];    // per inserter and the loader, whether its range was claimed
int range_first[3]; // per inserter and the loader, its range's first id
bool pending[2];    // per inserter, an accepted batch the device has rows left to serve
byte served[2];     // per inserter, the rows of its batch the device published
int refused[2];     // per inserter, the count its ramp last refused a row on; -1 when none

// the launch: everything the host thread wrote or acquired is visible to the grid it launches
#if memory != sequential
inline launch_edge(t) {
    atomic {
        for (view_location : 0 .. location_count - 1) {
            if
            :: current[thread_index(t, view_location)] > current[thread_index(device_thread, view_location)] ->
                current[thread_index(device_thread, view_location)] = current[thread_index(t, view_location)]
            :: else
            fi
        };
        view_location = 0
    }
}
#else
inline launch_edge(t) { skip }
#endif

// spin_mutex::lock: a non-writing spin until the word looks free, then the acquire exchange
inline lock_claiming(t, observed) {
    do
    :: (newest_value(claiming) == 0);
       atomic {
           read_modify_write(t, claiming, order_acquire, observed, 1);
           if
           :: observed == 0 ->
               claim_holders++;
#if memory == sequential
               assert(claim_holders == 1)
#endif
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
#if memory == sequential
        assert(growing == 1);
#endif
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

// a claim of `rows` ids from `first`: inside the planes, and apart from every range claimed before
inline claim_range(who, first, rows) {
    assert(newest_value(mapped_rows) >= first + rows);
    assert(!claimed[0] || first >= range_first[0] + rows_per_batch || first + rows <= range_first[0]);
    assert(!claimed[1] || first >= range_first[1] + rows_per_batch || first + rows <= range_first[1]);
    assert(!claimed[2] || first >= range_first[2] + rows_per_load || first + rows <= range_first[2]);
    claimed[who] = true;
    range_first[who] = first
}

active [2] proctype inserter() {
    byte me;
    int observed, first_id, total, mapped;
    atomic { me = inserters_started; inserters_started++ };
#ifndef without_claiming_mutex
    lock_claiming(me, observed);
#endif
    atomic {
        load(me, claimed_rows, order_relaxed, first_id);
        total = first_id + rows_per_batch
    };
    grow(me, total, mapped);
    launch_edge(me);
    // the launch's verdict: accepted, the claim is stored and the grid publishes; rejected, it is un-claimed exactly
    if
    :: true ->
        atomic {
            claim_range(me, first_id, rows_per_batch);
            store(me, claimed_rows, claim_order, total);
            pending[me] = true;
            refused[me] = -1
        }
    :: true ->
#ifdef without_unclaim_on_rejection
        atomic {
            claim_range(me, first_id, rows_per_batch);
            store(me, claimed_rows, claim_order, total)
        }
#else
        skip
#endif
    fi;
#ifndef without_claiming_mutex
    unlock_claiming(me);
#endif
    inserters_done++
}

// plan_load and finish: the claim if free, the growth, the restored count, the claim's release
active proctype loader() {
    int observed, base, total, mapped;
#ifdef without_base_under_claim
    load(loader_thread, claimed_rows, order_acquire, base);
#endif
    atomic {
        read_modify_write(loader_thread, claiming, order_acquire, observed, 1);
        if
        :: observed != 0 ->
            contended++;
#if memory == sequential
            assert(claim_holders == 1)
#endif
        :: else ->
#if memory == sequential
            assert(claim_holders == 0);
#endif
            claim_holders++
        fi
    };
    if
    :: observed != 0
    :: else ->
        atomic {
#ifndef without_base_under_claim
            load(loader_thread, claimed_rows, order_acquire, base);
#endif
            total = base + rows_per_load
        };
        grow(loader_thread, total, mapped);
        claim_range(2, base, rows_per_load);
        store(loader_thread, published_rows, order_relaxed, total);
        store(loader_thread, claimed_rows, claim_order, total);
        unlock_claiming(loader_thread)
    fi
}

// submit_search_: the sampling bound is size(), and the kernel touches rows below it
active proctype searcher() {
    int bound, mapped;
    load(searcher_thread, claimed_rows, size_order, bound);
    load(searcher_thread, mapped_rows, order_acquire, mapped); // capacity() acquires it
    assert(bound <= mapped)
}

// one row of batch b through the admission ramp: refused, the count it read is remembered so
// the retry waits for the count to move, which is what the `__nanosleep` loop amounts to
inline serve_row(b) {
    atomic {
        (pending[b] && newest_value(published_rows) != refused[b]);
        id = range_first[b] + served[b];
        load(device_thread, published_rows, order_relaxed, published);
        if
        :: id >= published + (published > ramp_floor -> published : ramp_floor) -> refused[b] = published
        :: else ->
            load(device_thread, mapped_rows, order_relaxed, mapped);
            assert(id < mapped);
            add_no_return(device_thread, published_rows, order_relaxed, 1);
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
active proctype device() {
    int published, id, mapped;
    do
    :: serve_row(0)
    :: serve_row(1)
    :: (inserters_done == 2 && !pending[0] && !pending[1]) -> break
    od
}
