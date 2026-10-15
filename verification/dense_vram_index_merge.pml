/**
 *  @brief The CUDA index's refine merging a re-selected row into the live one beside a walking reader
 *      and an insertion's back-link, the two mutators kept apart by the closed dispenser.
 *
 *  From `include/usearch/dense/vram_index.cuh`: `vram_merge_back_links_one` writes every slot of a
 *  live row, slot zero included and never the sentinel, with relaxed stores in any order, so a walk
 *  reads each slot old or new. A refine closes the shard's `claimed_rows_` before its pass and
 *  reopens it after, refused while a batch is in flight, and a batch's claim is refused while it is
 *  closed, so an insertion's back-links never land in a row the merge rewrites.
 *
 *  Row 0 holds node 1 and an empty slot; nodes 1 and 2 are published below the watermark. The
 *  merger lane closes the dispenser, stages row 0 as its pass read it, waiting out a lock holder,
 *  and stores the re-selection back, the staged neighbors reordered and an empty slot filled with
 *  node 2. The back-linker lane is a batch inserting node 3: it claims, writes node 3's payload,
 *  takes row 0's bounded lock, sweeps slot one, appends node 3 into it when empty, unlocks, and
 *  turns the watermark past node 3. The reader lane acquires the watermark as its walk's bound,
 *  acquires slot zero with one retry on the sentinel, loads slot one, and reads the payload of
 *  every ID below the bound.
 *
 *  Slot zero reads the sentinel only while a lock holder has the row, every ID the reader admits
 *  names a node whose payload it sees whole, the row old, new or mixed, and a back-link the batch
 *  appended survives unless the merge saw it.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc refine_closes=false: a refine closes the dispenser, which refuses the batch's
 *      claim; without it, the batch appends its back-link into row 0 after the merge staged the row
 *      and before it stored the re-selection back, and the store drops the only in-link of a node
 *      the merge never saw
 */
#define thread_count 3
#define location_count 6
#define history_depth 4
#include "weak_memory.pml"

// The words: row 0's slot zero, the lock and first neighbor, and its one tail slot; the payloads of
// nodes 1 to 3; the shard's watermark.
#define row_lock 0
#define row_tail 1
#define payload(id) (1 + (id))
#define last_visible_id 5

// The values: the sentinel is the all-ones ID in silicon, a slot names a node by its ID, and the
// initial payload word is what a stale read returns.
#define sentinel 0
#define written 7


// The knob: the header's choice, overridden by a `@verify` line that replays its counterexample.
#ifndef refine_closes
#define refine_closes true
#endif

byte claimed;     // the shard's `claimed_rows_`, a ghost each step on which is one compare-exchange
bool closed;      // the dispenser holds the closed state
byte done;        // the merger and the back-linker, once finished
bool editing;     // the back-linker holds row 0's lock
bool linked;      // the back-linker appended node 3 into row 0
bool merge_saw;   // the merger staged node 3

// `vram_merge_back_links_one` between the closing and the reopening: the staged row, then every
// slot stored relaxed in either order, slot zero never the sentinel
proctype merger(byte t) {
    int visible, head, tail;
    // `close_`: from the watermark acquired, refused while a batch is in flight
    if
    :: refine_closes ->
        load(t, last_visible_id, order_acquire, visible);
        atomic {
            if
            :: !closed && claimed == visible -> closed = true
            :: else -> goto refused
            fi
        }
    :: else
    fi;
    (newest_value(row_lock) != sentinel);
    load(t, row_lock, order_relaxed, head);
    load(t, row_tail, order_relaxed, tail);
    if
    :: head == sentinel -> goto reopen // a lock holder, which the closure keeps out
    :: else
    fi;
    merge_saw = head == 3 || tail == 3;
    if
    :: tail == sentinel -> tail = 2
    :: else
    fi;
    if
    :: store(t, row_lock, order_relaxed, tail); store(t, row_tail, order_relaxed, head)
    :: store(t, row_tail, order_relaxed, head); store(t, row_lock, order_relaxed, tail)
    fi;
reopen:
    if
    :: refine_closes -> closed = false
    :: else
    fi;
refused:
    done++
}

// a batch inserting node 3: the claim, the payload, the bounded lock, the sweep and the append,
// the unlock, and the turn
proctype back_linker(byte t) {
    int displaced, slot1;
    byte attempt;
    atomic {
        if
        :: !closed && claimed == newest_value(last_visible_id) -> claimed = 4
        :: else -> goto refused
        fi
    };
    store(t, payload(3), order_relaxed, written);
lock:
    atomic {
        read_modify_write_if(t, row_lock, order_acquire, displaced != sentinel, displaced, sentinel);
        editing = displaced != sentinel
    };
    if
    :: displaced == sentinel && attempt == 0 -> attempt = 1; goto lock
    :: displaced == sentinel -> goto turn // the lock budget spent: the back-link is dropped
    :: else
    fi;
    fence_release(t);
    load(t, row_tail, order_acquire, slot1);
    if
    :: slot1 == sentinel ->
        atomic {
            store(t, row_tail, order_relaxed, 3);
            linked = true
        }
    :: else
    fi;
    atomic {
        store(t, row_lock, order_release, displaced);
        editing = false
    };
turn:
    store(t, last_visible_id, order_release, 4);
refused:
    done++
}

// a walk over row 0 under the bound it acquired: every ID below the bound read with its payload
proctype reader(byte t) {
    int bound, first, second, seen;
    byte attempt;
    load(t, last_visible_id, order_acquire, bound);
peek:
    atomic {
        load(t, row_lock, order_acquire, first);
        assert(memory != sc || first != sentinel || editing)
    };
    if
    :: first == sentinel && attempt == 0 -> attempt = 1; goto peek
    :: first == sentinel -> goto locked
    :: else
    fi;
    load(t, row_tail, order_acquire, second);
    if
    :: first < bound -> load(t, payload(first), order_relaxed, seen); assert(seen == written)
    :: else
    fi;
    if
    :: second != sentinel && second < bound -> load(t, payload(second), order_relaxed, seen); assert(seen == written)
    :: else
    fi;
locked:
    skip
}

// once both mutators are done: an appended back-link the merge never saw is still in the row
proctype checker() {
    (done == 2);
    assert(!linked || merge_saw || newest_value(row_lock) == 3 || newest_value(row_tail) == 3)
}

// row 0 holding node 1, nodes 1 and 2 published below the watermark, before any thread runs
init {
    atomic {
        seed(row_lock, 1);
        seed(payload(1), written);
        seed(payload(2), written);
        seed(last_visible_id, 3);
        claimed = 3;
        run merger(0); run reader(1); run back_linker(2); run checker()
    }
}
