/**
 *  @file verification/dense_ram_indexes_gate/clearing_locators.pml
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief An insert plan's `finish` beside a `clear`: the key filed inside the turn never names a
 *      row the clear forgot.
 *
 *  The one shard holds one row dispensed and landed, short of its watermark turn. `finish`, holding
 *  the gate's shared side, awaits the turn, files the row's key into shard 0's map, the map's one
 *  writer while it holds the turn, and stores the watermark. The clearer takes the unique side or is
 *  refused, closes the shard, which `close_` refuses while a dispensed row awaits its turn, and
 *  inside its window forgets the row and empties the map.
 *
 *  - Once both are done, the map names no row the index forgot.
 *
 *  @verify pass sc
 *  @verify pass sc insert_hold=none: the key files before the turn ends, and `close_` refuses until
 *      it ends, so the shared side no longer guards the map against a clear
 */
#define thread_count 2
#define location_count 6
#define history_depth 1
#include <dense_ram_indexes_gate/protocol.pml>

// The word past the protocol's: the key's slot in shard 0's map, zero when empty and one past its
// row.
#define map_slot 5

proctype finisher(byte t) {
    int observed, watermark;
    hold_shared(t, observed);
    await_turn(t, watermark(0), 0, watermark);
    store(t, map_slot, order_release, 1);
    end_turn(t, watermark(0), order_release, 1);
    release_shared(t, observed)
}

proctype clearer(byte t) {
    int observed, seen, epoch_seen;
    byte shard;
    bool succeeded, refused;
    try_lock(t, gate, observed, succeeded);
    if
    :: succeeded ->
        close_shards(t, 1, refused);
        if
        :: !refused ->
            begin_window(t, epoch_seen);
            forget(t, 0);
            store(t, map_slot, order_relaxed, 0);
            end_window(t, epoch_seen);
            reopen_shards(1)
        :: else
        fi;
        unlock(t, gate, observed)
    :: else
    fi
}

init {
    atomic {
        claimed[0] = 1;
        dispensed[0] = 1;
        run finisher(0);
        run clearer(1)
    };
    _nr_pr == 1;
    assert(newest_value(map_slot) == 0 || newest_value(map_slot) - 1 < newest_value(watermark(0)))
}
