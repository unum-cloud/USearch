/**
 *  @brief The basic index's public calls interleaved on one index, and a failed load's leftovers.
 *
 *  At the protocol layer: `insert` batches under the shared side of the gate, a certified `search`,
 *  `plan_save` and `health` on the unique side, and `plan_load` on the unique side of an empty
 *  index. The gate is the abstract shared mutex, `basic_shared_mutex.pml` covers the blind add and
 *  the back-off, and memory is sequentially consistent: this model composes the calls, the ones
 *  below it order the words.
 *
 *  Two adders each bring one or two values into a capacity of three, so an overflow is reachable;
 *  two loaders each bring two into an empty index, so they and the adders race for the first
 *  publication, and a load may fail after wiring its rows and registering its keys.
 *
 *  Invariants, checked in every reachable state by the monitor and at the calls' exits:
 *  - `visible <= dispensed <= claimed <= capacity`, and every ID below `visible` is published;
 *  - an adder past the gate has its whole batch claimed before it dispenses, and takes the
 *    watermark turn in dispensed order, so the prefix is hole-free;
 *  - the unique side finds no adder inside and a settled prefix, `visible == dispensed`;
 *  - a load succeeds only on an index nothing was ever dispensed into;
 *  - a certified search saw no acquisition of the gate and the same watermark twice;
 *  - a row is wired from nothing, and a key is registered once;
 *  - the certified search's second refusal at entry, an unsettled prefix, is subsumed: an unlocked
 *    gate has no adder inside, so `visible == dispensed`, which the entry asserts.
 *
 *  @verify pass sc
 *  @verify fail sc row_clear=false: `insert` clears its row before wiring it, so a failed load's
 *      leftover wiring never reaches a published node; without the clear, an adder wires a row a
 *      dead load already wired
 *  @verify fail sc map_clear=false: `load_plan_t::load` registers keys as it decodes, and a plan
 *      that dies before `finish` wipes them under its hold, whether its steps fell short or nobody
 *      finished it; without the wipe, an `insert` dispenses an ID the failed load's keys still name
 *  @verify fail sc spends_plan=false: a successful `finish` spends the plan, so a second call
 *      answers a status; without it, the second call drops a hold the spent plan does not own, and
 *      the gate's assertion names it or an adder walks in on a loader's hold
 */
// The sizes: two adders bringing one or two values each overflow the capacity.
#define capacity 3
#define loaded_rows 2 // what a loader brings into an empty index

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef row_clear
#define row_clear true
#endif
#ifndef map_clear
#define map_clear true
#endif
#ifndef spends_plan
#define spends_plan true
#endif

byte claimed, dispensed, visible;
bool published[capacity];
bool wired[capacity]; // the row holds a wiring, published or a failed load's leftover
bool keyed[capacity]; // the key map holds the ID
byte holders;
bool unique;
byte generation;
byte adders_inside; // past the gate, with a claim or a batch in flight
bool walking;       // a certified search is between its probes
bool disturbed;     // the gate was acquired while it was

inline lock_shared() {
    atomic {
        !unique;
        holders++;
        generation++;
        if :: walking -> disturbed = true :: else fi
    }
}
inline unlock_shared() { holders-- }
inline try_lock(succeeded) {
    atomic {
        if
        :: holders == 0 && !unique ->
            unique = true;
            generation++;
            if :: walking -> disturbed = true :: else fi;
            succeeded = true
        :: else -> succeeded = false
        fi
    }
}
// the unique side is dropped by its holder alone: a plan spent by `finish` owns no hold to drop
inline unlock() { assert(unique); unique = false }

// what every unique holder sees: nobody inside, a settled prefix, every visible row published
inline settled_prefix() {
    atomic {
        assert(adders_inside == 0);
        assert(visible == dispensed);
        for (id : 0 .. capacity - 1) { assert(id >= visible || published[id]) }
    }
}

proctype monitor() {
    assert(visible <= dispensed && dispensed <= claimed && claimed <= capacity)
}

proctype adder() {
    byte batch, id, value, first_invisible;
    bool claimed_batch;
    select(batch : 1 .. 2);
    // the gate opens before the claim, so a batch parked behind a unique holder holds no capacity
    lock_shared();
    adders_inside++;
    atomic {
        claimed_batch = claimed + batch <= capacity;
        if :: claimed_batch -> claimed = claimed + batch :: else fi
    };
    if
    :: !claimed_batch -> adders_inside--; unlock_shared(); goto refused
    :: else
    fi;
    for (value : 1 .. batch) {
        // the walk, against the visible prefix
        atomic {
            first_invisible = visible;
            for (id : 0 .. capacity - 1) { assert(id >= first_invisible || published[id]) }
        };
        // identity after the walk, the row from nothing, then the watermark turn in dispensed order
        atomic { id = dispensed; dispensed++ };
        if
        :: row_clear -> wired[id] = false
        :: else
        fi;
        assert(!wired[id]);
        assert(!keyed[id]); // no failed load's key names this ID
        atomic { wired[id] = true; published[id] = true };
        (visible == id);
        visible = id + 1
    };
    adders_inside--;
    unlock_shared();
refused:
    skip
}

proctype searcher() {
    byte entry, first_invisible, id;
    atomic {
        if
        :: holders == 0 && !unique -> assert(visible == dispensed); entry = generation; walking = true; disturbed = false
        :: else -> goto contended
        fi
    };
    atomic {
        first_invisible = visible;
        for (id : 0 .. capacity - 1) { assert(id >= first_invisible || published[id]) }
    };
    atomic {
        walking = false;
        if
        :: holders == 0 && !unique && generation == entry && visible == first_invisible -> assert(!disturbed)
        :: else
        fi
    };
contended:
    skip
}

proctype saver() {
    byte id;
    bool succeeded;
    try_lock(succeeded);
    if
    :: succeeded ->
        settled_prefix();
        unlock();
        if
        :: !spends_plan -> unlock()
        :: else
        fi
    :: else
    fi
}

proctype loader() {
    byte id, decoded;
    bool succeeded;
    try_lock(succeeded);
    if
    :: succeeded ->
        if
        :: visible == 0 ->
            settled_prefix();
            assert(claimed == 0);
            // load: every row wired and keyed before anything publishes, then the decode's verdict
            for (id : 0 .. loaded_rows - 1) { wired[id] = true; assert(!keyed[id]); keyed[id] = true };
            select(decoded : 0 .. 1);
            if
            :: decoded ->
                for (id : 0 .. loaded_rows - 1) { published[id] = true };
                atomic { claimed = loaded_rows; dispensed = loaded_rows; visible = loaded_rows }
            :: else ->
                // the dying plan wipes the keys its steps registered, still under its hold
                if
                :: map_clear -> for (id : 0 .. loaded_rows - 1) { keyed[id] = false }
                :: else
                fi
            fi
        :: else
        fi;
        unlock()
    :: else
    fi
}

init { atomic { run monitor(); run adder(); run adder(); run searcher(); run saver(); run loader(); run loader() } }
