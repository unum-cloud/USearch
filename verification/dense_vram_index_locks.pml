/**
 *  @file verification/dense_vram_index_locks.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The CUDA index's rows: publication by one release, the bounded slot-zero lock, the
 *      back-link's sweep and append, and a traversal warp whose follower lanes read behind a shuffle.
 *
 *  From `include/usearch/dense/vram_nvidia_index.cuh`: no device watermark, one warp per node, slot
 *  zero of every row the combined lock and first-neighbor word. A warp is two lanes here, the
 *  leader and one follower, each a thread of the memory model with its own view; `__syncwarp` is
 *  the only step that merges views, `__shfl_sync` copies a value and merges nothing, which is what
 *  the PTX ISA promises of each.
 *
 *  - An ID a lane read from a published row names a node it sees published, with its payload whole.
 *  - A lock holder's edit is whole to a reader that acquired its unlock.
 *  - An assigned ID ends published, self-loop or not, whether or not its reverse edge landed.
 *  - An append lands in a slot that is empty.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc syncwarp_after_broadcast=false syncwarp_after_lock=false tail_order=order_relaxed tail_fence=false
 *  @verify fail rc11 syncwarp_after_broadcast=false: loading their tail slots straight after it, a
 *      lane reads a slot older than the unlock its leader acquired, and the warp assembles a row no
 *      holder ever wrote
 *  @verify fail rc11 syncwarp_after_lock=false: sweeping the locked row straight after it, a stale
 *      sentinel names a slot the previous holder just filled, and the append overwrites that
 *      holder's link
 *  @verify fail rc11 tail_order=order_relaxed: relaxed, it returns an edit newer than the unlock its
 *      leader acquired, which carries nothing, and the ID it found names a node whose payload it
 *      then reads stale
 *  @verify fail rc11 publish_order=order_relaxed: relaxed, a peer insertion that draws the node as a
 *      landmark sees it published and reads its payload stale
 *  @verify fail rc11 tail_fence=false: without it, the follower reads an edit newer than the unlock
 *      its leader acquired, and the payload behind it stale
 *  @verify fail rc11 syncwarp_after_broadcast=false syncwarp_after_lock=false tail_order=order_relaxed tail_fence=false:
 *      under `rc11` a follower reads a node's payload stale behind an edit newer than the unlock
 *      its leader acquired
 */
#define thread_count 5
#define location_count 4
#define history_depth 6
#include <weak_memory.pml>

// The warps: warp w runs its leader lane as thread 2w and its follower as 2w + 1; the prior holder is thread 4.
#define inserter_warp 0
#define reader_warp 1
#define leader_of(w) ((w) * 2)
#define follower_of(w) ((w) * 2 + 1)

// The words: a row is its slot zero, the lock, and one tail slot; the newcomer also owns a payload word.
#define row_lock(id) ((id) * 2)
#define row_tail(id) ((id) * 2 + 1)
#define payload(id) ((id) * 2 + 1)

// The values rows carry: the sentinel is the all-ones ID in silicon, and the initial payload word
// is what a stale read returns.
#define sentinel 0
#define node(id) (1 + (id))
#define written 7

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef syncwarp_after_broadcast
#define syncwarp_after_broadcast true
#endif
#ifndef syncwarp_after_lock
#define syncwarp_after_lock true
#endif
#ifndef tail_order
#define tail_order order_acquire
#endif
#ifndef tail_fence
#define tail_fence true
#endif
#ifndef publish_order
#define publish_order order_release
#endif

bool launched;
byte links;       // the walk's selection: 1 links the newcomer to node 0, 0 publishes it empty
byte arrivals[2]; // per warp, the lanes' barrier arrivals so far
int broadcast_lock, broadcast_first; // the shuffled words: no view moves with them
bool lock_ready, first_ready;
bool slot1_empty; // the sweep's ballot
byte lock_skips, locked_drops;
byte published_at;                  // the history index of the newcomer's publication
byte tail_at_unlock[history_depth]; // per slot-zero write, the slot-one write it published
byte leader_read;                   // the slot-zero write the reader's leader acquired
hidden int drawn;                   // scratch: the landmark draw's slot zero, checked in the same step

// `__syncwarp`: the k-th barrier of warp w completes when both lanes arrived k times, and the last
// arrival merges both lanes' views both ways, since `bar.warp.sync` orders memory among its participants
inline syncwarp(w, phase) {
    phase = phase + 1;
    atomic {
        arrivals[w] = arrivals[w] + 1;
        if
        :: arrivals[w] == 2 * phase -> merge_view(leader_of(w), follower_of(w)); merge_view(follower_of(w), leader_of(w))
        :: else
        fi
    };
    (arrivals[w] >= 2 * phase)
}

// the release unlock, remembering which slot-one write it published
inline unlock(t, first_value) {
    atomic {
        store(t, row_lock(0), order_release, first_value);
        tail_at_unlock[newest_write(row_lock(0))] = newest_write(row_tail(0))
    }
}

// a lane scoring node 1: it must see the node published and its payload as written
inline see_node1(t, seen) {
    atomic {
        assert(memory != sc || newest_value(row_lock(1)) != sentinel);
        assert(seen_write(t, row_lock(1)) >= published_at);
        load(t, payload(1), order_relaxed, seen);
        assert(seen == written)
    }
}

proctype inserter_leader_lane(byte t) {
    int first_link, displaced;
    byte phase, attempt;
    // the host: row 0 published by an earlier batch, then the CUDA API edge to every lane of this launch
    if
    :: true -> store(t, row_lock(0), order_relaxed, node(0)) // published-empty
    :: true -> store(t, row_lock(0), order_relaxed, node(3)) // headed by an older node
    fi;
    atomic { see_newest(0); see_newest(1); see_newest(2); see_newest(3); see_newest(4) };
    if
    :: true -> links = 0
    :: true -> links = 1
    fi;
    launched = true;
    // the newcomer: the lanes' payload writes, the syncwarp, the one release that publishes
    syncwarp(inserter_warp, phase);
    first_link = (links -> node(0) : node(1)); // a self-loop encodes published-empty
    atomic {
        store(t, row_lock(1), publish_order, first_link);
        published_at = newest_write(row_lock(1))
    };
    if
    :: links == 1 ->
        // back_link_neighbors: the bounded lock, the shuffle, the sweep, the edit, the unlock
lock:
        read_modify_write_if(t, row_lock(0), order_acquire, displaced != sentinel, displaced, sentinel);
        if
        :: displaced == sentinel && attempt == 0 -> attempt = 1; goto lock
        :: else
        fi;
        atomic { broadcast_lock = displaced; lock_ready = true };
        if
        :: displaced == sentinel -> lock_skips++
        :: displaced == node(0) -> unlock(t, node(1)) // published-empty: the newcomer heads it
        :: else ->
            if
            :: syncwarp_after_lock -> syncwarp(inserter_warp, phase)
            :: else
            fi;
            syncwarp(inserter_warp, phase); // after the sweep
            if
            :: tail_fence -> fence_release(t)
            :: else
            fi;
            if
            :: slot1_empty ->
                atomic {
                    assert(newest_value(row_tail(0)) == sentinel); // the append lands in an empty slot
                    store(t, row_tail(0), order_relaxed, node(1))
                };
                syncwarp(inserter_warp, phase)
            :: else -> syncwarp(inserter_warp, phase) // the rewrite
            fi;
            unlock(t, displaced)
        fi
    :: else
    fi;
    assert(newest_value(row_lock(1)) != sentinel) // the newcomer ends published
}

proctype inserter_follower_lane(byte t) {
    int displaced, slot1;
    byte phase;
    (launched);
    store(t, payload(1), order_relaxed, written);
    syncwarp(inserter_warp, phase);
    if
    :: links == 1 ->
        (lock_ready);
        displaced = broadcast_lock;
        if
        :: displaced == sentinel || displaced == node(0)
        :: else ->
            if
            :: syncwarp_after_lock -> syncwarp(inserter_warp, phase)
            :: else
            fi;
            load(t, row_tail(0), tail_order, slot1); // the sweep of slot one, an acquire as every slot load
            slot1_empty = (slot1 == sentinel);
            syncwarp(inserter_warp, phase);
            if
            :: tail_fence -> fence_release(t)
            :: else
            fi;
            if
            :: slot1_empty -> syncwarp(inserter_warp, phase)
            :: else -> // full: the re-selection keeps the newcomer in this lane's slot
                store(t, row_tail(0), order_relaxed, node(1));
                syncwarp(inserter_warp, phase)
            fi
        fi
    :: else
    fi
}

// another insertion, one lane: its walk may draw node 1 as a landmark, gated on node 1's slot
// zero, and its back-link into row 0 is the edit the inserter's sweep may miss
proctype prior_holder_lane(byte t) {
    int displaced;
    byte attempt;
    (launched);
    atomic {
        load(t, row_lock(1), order_acquire, drawn);
        if
        :: drawn != sentinel -> see_node1(t, drawn)
        :: else
        fi
    };
lock:
    read_modify_write_if(t, row_lock(0), order_acquire, displaced != sentinel, displaced, sentinel);
    if
    :: displaced == sentinel && attempt == 0 -> attempt = 1; goto lock
    :: else
    fi;
    if
    :: displaced == sentinel -> lock_skips++
    :: displaced == node(0) -> unlock(t, node(2))
    :: else ->
        // every lock holder's slot stores follow its release fence
        if
        :: tail_fence -> fence_release(t)
        :: else
        fi;
        store(t, row_tail(0), order_relaxed, node(2));
        unlock(t, displaced)
    fi
}

proctype reader_leader_lane(byte t) {
    int first, seen;
    byte phase, attempt;
    (launched);
peek:
    atomic {
        load(t, row_lock(0), order_acquire, first);
        leader_read = seen_write(t, row_lock(0))
    };
    if
    :: first == sentinel && attempt == 0 -> attempt = 1; goto peek
    :: else
    fi;
    atomic { broadcast_first = first; first_ready = true };
    if
    :: first == sentinel -> locked_drops++
    :: else ->
        if
        :: syncwarp_after_broadcast -> syncwarp(reader_warp, phase)
        :: else
        fi;
        syncwarp(reader_warp, phase); // the staging's syncwarp
        if
        :: first == node(1) -> see_node1(t, seen)
        :: else
        fi
    fi
}

proctype reader_follower_lane(byte t) {
    int first, tail, seen;
    byte phase;
    (first_ready);
    first = broadcast_first;
    if
    :: first == sentinel
    :: else ->
        if
        :: syncwarp_after_broadcast -> syncwarp(reader_warp, phase)
        :: else
        fi;
        atomic {
            load(t, row_tail(0), tail_order, tail);
            assert(seen_write(t, row_tail(0)) >= tail_at_unlock[leader_read]) // the edit is whole
        };
        syncwarp(reader_warp, phase);
        if
        :: tail == node(1) -> see_node1(t, seen)
        :: else
        fi
    fi
}

init {
    atomic {
        run inserter_leader_lane(0);
        run inserter_follower_lane(1);
        run reader_leader_lane(2);
        run reader_follower_lane(3);
        run prior_holder_lane(4)
    }
}
