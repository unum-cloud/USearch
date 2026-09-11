/**
 *  The CUDA index's row protocol from `include/usearch/dense/vram_index.cuh`: no watermark, one
 *  warp per node, slot zero of every row the combined lock and first-neighbor word. A warp is
 *  two lanes here, the leader and one follower, each a thread of the memory model with its own
 *  view; `__syncwarp` is the only step that merges views, `__shfl_sync` copies a value and
 *  merges nothing, which is what the PTX ISA promises of each.
 *
 *  Row 0 is published by an earlier batch, made visible to every lane by the launch edge. The
 *  inserter warp lands node 1: the follower writes the payload word (it stands for the vector
 *  row and the own tail slots, which follow the same path), the warp syncs, the leader
 *  release-stores slot zero - node 0 when the walk found it, a self-loop otherwise. It then
 *  back-links into row 0: the leader's bounded acquire exchange of the sentinel, broadcast by
 *  shuffle; a published-empty row takes the newcomer as its head, otherwise the follower sweeps
 *  slot one and the empty slot is appended into by the leader, or the full row is rewritten by
 *  the follower. A single-lane prior holder appends node 2 into the same row, so the inserter's
 *  sweep may follow an edit its leader's exchange acquired. The reader warp walks row 0: the
 *  leader acquires slot zero with one retry on the sentinel, broadcasts it, the follower loads
 *  slot one, the warp syncs on its staging and each lane reads its neighbor's payload plainly.
 *
 *  Invariants:
 *  - (a) an id a lane read from a published row names a node it sees published, with the
 *    payload it sees whole. The follower's slot-one load acquires, pairing with the release
 *    fence every lock holder places between its lock and its slot stores; `-Dwithout_tail_acquire`
 *    drops the acquire and `-Dwithout_tail_fence` the fence, and either fails: the follower
 *    reads an edit newer than the unlock its leader acquired, and the payload behind it stale;
 *  - (b) a lock holder's edit is whole to a reader that acquired its unlock: the follower's
 *    slot-one read is at least the write that unlock published. `-Dwithout_syncwarp_after_broadcast`
 *    reads slot one straight after the shuffle, as the header did before its `__syncwarp`, and
 *    it fails;
 *  - (c) an assigned id ends published, self-loop or not, whether or not its reverse edge landed;
 *  - (d) an append lands in a slot that is empty: the sweep that found it followed the unlock
 *    the exchange acquired. `-Dwithout_syncwarp_after_lock` sweeps straight after the lock's
 *    shuffle, as the header did before its `__syncwarp`, and it fails - the prior holder's link
 *    is overwritten.
 *  `-Dwithout_fixes` is the header before those four edits: all four drops at once.
 */
#include "weak_memory.pml"

// The threads, by role, apart from the processes that play them.
#define inserter_leader_thread 0
#define inserter_follower_thread 1
#define reader_leader_thread 2
#define reader_follower_thread 3
#define prior_holder_thread 4
#define inserter_warp 0
#define reader_warp 1

// The words: a row is its slot zero, the lock, and one tail slot; the newcomer also owns a payload word.
#define row_lock(id) ((id) * 2)
#define row_tail(id) ((id) * 2 + 1)
#define payload(id) ((id) * 2 + 1)

// The values rows carry: the sentinel is the all-ones id in silicon, and the initial payload
// word is what a stale read returns.
#define sentinel 0
#define node(id) (1 + (id))
#define written 7

#ifdef without_fixes
#define without_syncwarp_after_broadcast
#define without_syncwarp_after_lock
#define without_tail_acquire
#define without_tail_fence
#endif
#ifdef without_tail_acquire
#define tail_order order_relaxed
#else
#define tail_order order_acquire
#endif

bool launched;
byte links;      // the walk's selection: 1 links the newcomer to node 0, 0 publishes it empty
byte arrivals[2]; // per warp, the lanes' barrier arrivals so far
int broadcast_lock, broadcast_first; // the shuffled words: no view moves with them
bool lock_ready, first_ready;
bool slot1_empty; // the sweep's ballot
byte lock_skips, locked_drops;
#if memory != sequential
byte view_thread;         // scratch for the launch edge
byte published_at;        // the history index of the newcomer's publication
byte tail_at_unlock[history_depth]; // per slot-zero write, the slot-one write it published
byte leader_read;         // the slot-zero write the reader's leader acquired
#endif

// `__syncwarp`: the k-th barrier of warp w completes when both lanes arrived k times, and the
// last arrival merges both lanes' views both ways - `bar.warp.sync` orders memory among its
// participants. `acquired` moves too, so a later acquire fence on either lane folds what the
// other's relaxed loads picked up; nothing here exercises that.
#if memory != sequential
inline merge_views(a, b) {
    for (view_location : 0 .. location_count - 1) {
        if
        :: current[thread_index(a, view_location)] > current[thread_index(b, view_location)] ->
            current[thread_index(b, view_location)] = current[thread_index(a, view_location)]
        :: else -> current[thread_index(a, view_location)] = current[thread_index(b, view_location)]
        fi;
        if
        :: acquired[thread_index(a, view_location)] > acquired[thread_index(b, view_location)] ->
            acquired[thread_index(b, view_location)] = acquired[thread_index(a, view_location)]
        :: else -> acquired[thread_index(a, view_location)] = acquired[thread_index(b, view_location)]
        fi
    };
    view_location = 0
}
// the CUDA API edge: everything an earlier batch wrote is visible to every lane of this launch
inline launch_edge() {
    atomic {
        for (view_thread : 0 .. thread_count - 1) {
            for (view_location : 0 .. location_count - 1) {
                current[thread_index(view_thread, view_location)] = newest[view_location]
            }
        };
        view_thread = 0;
        view_location = 0
    }
}
#else
inline merge_views(a, b) { skip }
inline launch_edge() { skip }
#endif
inline syncwarp(w, a, b, phase) {
    phase = phase + 1;
    atomic {
        arrivals[w] = arrivals[w] + 1;
        if
        :: arrivals[w] == 2 * phase -> merge_views(a, b)
        :: else
        fi
    };
    (arrivals[w] >= 2 * phase)
}

// the release unlock, remembering which slot-one write it published
inline unlock(t, first_value) {
    atomic {
        store(t, row_lock(0), order_release, first_value);
#if memory != sequential
        tail_at_unlock[newest[row_lock(0)]] = newest[row_tail(0)]
#endif
    }
}

// a lane scoring node 1: it must see the node published and its payload as written
inline see_node1(t, seen) {
    atomic {
#if memory != sequential
        assert(current[thread_index(t, row_lock(1))] >= published_at);
#else
        assert(newest_value(row_lock(1)) != sentinel);
#endif
        load(t, payload(1), order_relaxed, seen);
        assert(seen == written)
    }
}

active proctype inserter_leader_lane() {
    int first_link, displaced;
    byte phase, attempt;
    // the host: row 0 published by an earlier batch, then the launch edge
    if
    :: true -> store(inserter_leader_thread, row_lock(0), order_relaxed, node(0)) // published-empty
    :: true -> store(inserter_leader_thread, row_lock(0), order_relaxed, node(3)) // headed by an older node
    fi;
    launch_edge();
    if
    :: true -> links = 0
    :: true -> links = 1
    fi;
    launched = true;
    // the newcomer: the lanes' payload writes, the syncwarp, the one release that publishes
    syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase);
    first_link = (links -> node(0) : node(1)); // a self-loop encodes published-empty
    atomic {
        store(inserter_leader_thread, row_lock(1), order_release, first_link);
#if memory != sequential
        published_at = newest[row_lock(1)]
#endif
    };
    if
    :: links == 1 ->
        // back_link_neighbors: the bounded lock, the shuffle, the sweep, the edit, the unlock
lock:
        read_modify_write_if(inserter_leader_thread, row_lock(0), order_acquire, displaced != sentinel, displaced, sentinel);
        if
        :: displaced == sentinel && attempt == 0 -> attempt = 1; goto lock
        :: else
        fi;
        atomic { broadcast_lock = displaced; lock_ready = true };
        if
        :: displaced == sentinel -> lock_skips++
        :: displaced == node(0) -> unlock(inserter_leader_thread, node(1)) // published-empty: the newcomer heads it
        :: else ->
#ifndef without_syncwarp_after_lock
            syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase);
#endif
            syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase); // after the sweep
#ifndef without_tail_fence
            fence_release(inserter_leader_thread);
#endif
            if
            :: slot1_empty ->
                atomic {
                    assert(newest_value(row_tail(0)) == sentinel); // (d)
                    store(inserter_leader_thread, row_tail(0), order_relaxed, node(1))
                };
                syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase)
            :: else -> syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase) // the rewrite
            fi;
            unlock(inserter_leader_thread, displaced)
        fi
    :: else
    fi;
    assert(newest_value(row_lock(1)) != sentinel) // (c)
}

active proctype inserter_follower_lane() {
    int displaced, slot1;
    byte phase;
    (launched);
    store(inserter_follower_thread, payload(1), order_relaxed, written);
    syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase);
    if
    :: links == 1 ->
        (lock_ready);
        displaced = broadcast_lock;
        if
        :: displaced == sentinel || displaced == node(0)
        :: else ->
#ifndef without_syncwarp_after_lock
            syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase);
#endif
            load(inserter_follower_thread, row_tail(0), tail_order, slot1); // the sweep of slot one, an acquire as every slot load
            slot1_empty = (slot1 == sentinel);
            syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase);
#ifndef without_tail_fence
            fence_release(inserter_follower_thread);
#endif
            if
            :: slot1_empty -> syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase)
            :: else -> // full: the re-selection keeps the newcomer in this lane's slot
                store(inserter_follower_thread, row_tail(0), order_relaxed, node(1));
                syncwarp(inserter_warp, inserter_leader_thread, inserter_follower_thread, phase)
            fi
        fi
    :: else
    fi
}

// another back-linker on row 0, one lane: its edit is what the inserter's sweep may miss
active proctype prior_holder_lane() {
    int displaced;
    byte attempt;
    (launched);
lock:
    read_modify_write_if(prior_holder_thread, row_lock(0), order_acquire, displaced != sentinel, displaced, sentinel);
    if
    :: displaced == sentinel && attempt == 0 -> attempt = 1; goto lock
    :: else
    fi;
    if
    :: displaced == sentinel -> lock_skips++
    :: displaced == node(0) -> unlock(prior_holder_thread, node(2))
    :: else ->
        // every lock holder's slot stores follow its release fence
#ifndef without_tail_fence
        fence_release(prior_holder_thread);
#endif
        store(prior_holder_thread, row_tail(0), order_relaxed, node(2));
        unlock(prior_holder_thread, displaced)
    fi
}

active proctype reader_leader_lane() {
    int first, seen;
    byte phase, attempt;
    (launched);
peek:
    atomic {
        load(reader_leader_thread, row_lock(0), order_acquire, first);
#if memory != sequential
        leader_read = current[thread_index(reader_leader_thread, row_lock(0))]
#endif
    };
    if
    :: first == sentinel && attempt == 0 -> attempt = 1; goto peek
    :: else
    fi;
    atomic { broadcast_first = first; first_ready = true };
    if
    :: first == sentinel -> locked_drops++
    :: else ->
#ifndef without_syncwarp_after_broadcast
        syncwarp(reader_warp, reader_leader_thread, reader_follower_thread, phase);
#endif
        syncwarp(reader_warp, reader_leader_thread, reader_follower_thread, phase); // the staging's syncwarp
        if
        :: first == node(1) -> see_node1(reader_leader_thread, seen)
        :: else
        fi
    fi
}

active proctype reader_follower_lane() {
    int first, tail, seen;
    byte phase;
    (first_ready);
    first = broadcast_first;
    if
    :: first == sentinel
    :: else ->
#ifndef without_syncwarp_after_broadcast
        syncwarp(reader_warp, reader_leader_thread, reader_follower_thread, phase);
#endif
        atomic {
            load(reader_follower_thread, row_tail(0), tail_order, tail);
#if memory != sequential
            assert(current[thread_index(reader_follower_thread, row_tail(0))] >= tail_at_unlock[leader_read]); // (b)
#endif
        };
        syncwarp(reader_warp, reader_leader_thread, reader_follower_thread, phase);
        if
        :: tail == node(1) -> see_node1(reader_follower_thread, seen)
        :: else
        fi
    fi
}
