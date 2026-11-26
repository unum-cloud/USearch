/**
 *  @file verification/arena_lifetime.pml
 *  @author Ash Vardanian
 *  @date November 26, 2026
 *  @brief Spin model of an arena's lifetime in every engine: calls taking it, a binding reading its
 *      outputs under an export, an exporter, and the frees of the arena and of its executor.
 *
 *  A call takes the arena's claim, a holder word that names the call, re-reads the export count,
 *  takes the executor's claim, advances the arena's generation once, and writes outputs tagged with
 *  that generation. An export raises the count, then reads the holder. The binding stamps a result
 *  with the generation it read before submitting plus one, and copies it under an export only while
 *  the generation reads that stamp. The arena counts itself on its executor at init and off last.
 *
 *  - No call writes while an export holds, and no export holds while a call may write.
 *  - A copy whose check passed reads outputs of its stamp: never falsely current.
 *  - A call refused at its take or at its claims never advances; a served one advances once.
 *  - The executor is never freed while an arena lives, and nothing touches it after.
 *  - Under @c sc, the arena frees nothing an export holds, even one opened during the wait.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 handshake_fence=false: the take and the export count reorder past the reads
 *      after them, so a call rewinds outputs an export holds open
 *  @verify fail sc export_check=before_wait: an export opened while the free waits out a deferred
 *      call is open when the free deletes the arena under it
 *  @verify fail sc copy_bracket=false: a copy checks the generation, then a call takes the arena
 *      and rewinds it before the copy, which reads the next call's outputs as current
 *  @verify fail sc bump_site=at_take: advanced at the take, a call the export count refuses has
 *      advanced the generation anyway
 *  @verify fail sc bump_on_retake=true: the call's body re-holding its own arena advances it twice
 *  @verify fail sc count_lowered_last=false: lowered first, the executor is freed while the arena
 *      still gives its memory back through it
 */
#define thread_count 5
#define location_count 8
#define history_depth 6
#include <weak_memory.pml>
#include <monitor_wait.pml>

/** The words: the arena's holder, export count, generation and outputs, the count of calls naming
 *  it, the executor's arena count and claim, and the word standing in for every sequentially
 *  consistent pair. A holder of zero is free, and any other value names the call holding it. */
#define holder 0
#define exports 1
#define generation 2
#define outputs 3
#define named_calls 4
#define arenas 5
#define holding 6
#define fences 7

/** The roles' threads: the pool thread serving the owner's deferred call, the binding, the
 *  exporter, the owner of the arena, and the thread freeing the executor. */
#define pool_thread 0
#define binding_thread 1
#define exporter_thread 2
#define owner_thread 3
#define executor_thread 4

/** The calls: the owner's, deferred to the pool, and the binding's, run on its own thread. */
#define owner_call 0
#define binding_call 1

/** How a call ended: refused at its take or at one of its claims, refused by the pool after it took
 *  everything, or served. */
#define refused 0
#define pool_refused 1
#define served 2

/** The values of @c export_check: the free reads the export count after it waits out the calls
 *  naming the arena, or before. */
#define after_wait 0
#define before_wait 1

/** The values of @c bump_site: a call advances the generation once it took both claims and read no
 *  export, or right at its take. */
#define after_claims 0
#define at_take 1

/** The knobs: the engines' choices, each overridden by a <tt>@verify</tt> line. */
#ifndef handshake_fence
#define handshake_fence true
#endif
#ifndef export_check
#define export_check after_wait
#endif
#if export_check != after_wait && export_check != before_wait
#error "export_check is after_wait or before_wait"
#endif
#ifndef copy_bracket
#define copy_bracket true
#endif
#ifndef bump_site
#define bump_site after_claims
#endif
#if bump_site != after_claims && bump_site != at_take
#error "bump_site is after_claims or at_take"
#endif
#ifndef bump_on_retake
#define bump_on_retake false
#endif
#ifndef count_lowered_last
#define count_lowered_last true
#endif

byte advances[2];      // generations each call advanced
byte tag[2];           // the generation each call advanced to, which its outputs carry
bool passed[2];        // the call holds the arena and read no export
byte rewinding;        // calls that passed and have not let the arena go
byte exporting;        // exports that read the arena free and are not closed
byte dispatched;       // the owner's call: 0 until submitted, 1 dispatched, 2 not
bool binding_done;     // the binding dropped its result, as its reference to the arena
bool free_waited;      // the free read no call naming the arena
bool arena_freed;      // the arena was deleted
bool executor_freed;   // the executor was deleted
hidden int fenced;     // scratch: the value a fence's stand-in read

/** A sequentially consistent pair's fence: an acquire-release read-modify-write of @c fences, so of
 *  two such pairs the later one reads everything before the earlier one's fence. */
inline fence_seq_cst(t) {
    if
    :: handshake_fence -> read_modify_write(t, fences, order_acq_rel, fenced, 0)
    :: else
    fi
}

/** Advances the generation for @p call, under the arena it holds. */
inline advance(t, call, before) {
    atomic {
        read_modify_write(t, generation, order_relaxed, before, before + 1);
        advances[call]++;
        tag[call] = before + 1
    }
}

/** Lets the arena go, as the call holding it. */
inline release_arena(t, call) {
    atomic {
        store(t, holder, order_release, 0);
        if
        :: passed[call] -> passed[call] = false; rewinding--
        :: else
        fi
    }
}

/** The count raised for an export. */
inline raise_exports(t, seen) { read_modify_write(t, exports, order_acq_rel, seen, seen + 1) }

/** The holder read after the count was raised, closing the count again on a held arena; @p opened
 *  says whether the export holds. */
inline read_holder(t, opened, seen) {
    fence_seq_cst(t);
    atomic {
        load(t, holder, order_acquire, seen);
        opened = (seen == 0);
        if
        :: opened -> exporting++; assert(rewinding == 0)
        :: else
        fi
    };
    if
    :: !opened -> read_modify_write(t, exports, order_release, seen, seen - 1)
    :: else
    fi
}

/** Closes an export that holds. */
inline export_close(t, seen) {
    atomic { exporting--; read_modify_write(t, exports, order_release, seen, seen - 1) }
}

/** Submits @p call: the arena's holder taken, the export count read, the executor's claim taken,
 *  the generation advanced and the call counted, or a refusal that let go of what it took. */
inline submit(t, call, outcome, seen, taken) {
    seen = 0;
    compare_exchange(t, holder, order_acq_rel, seen, call + 1, taken);
    if
    :: !taken -> assert(advances[call] == 0); outcome = refused
    :: else ->
        if
        :: bump_site == at_take -> advance(t, call, seen)
        :: else
        fi;
        fence_seq_cst(t);
        atomic {
            load(t, exports, order_acquire, seen);
            if
            :: seen == 0 -> passed[call] = true; rewinding++
            :: else
            fi
        };
        if
        :: seen != 0 -> release_arena(t, call); assert(advances[call] == 0); outcome = refused
        :: else ->
            seen = 0;
            atomic { compare_exchange(t, holding, order_acquire, seen, 1, taken); assert(!executor_freed) };
            if
            :: !taken -> release_arena(t, call); assert(advances[call] == 0); outcome = refused
            :: else ->
                if
                :: bump_site == after_claims -> advance(t, call, seen)
                :: else
                fi;
                read_modify_write(t, named_calls, order_relaxed, seen, seen + 1);
                outcome = served
            fi
        fi
    fi
}

/** Ends a submitted call: the arena let go, the call counted off, and the executor's claim given
 *  back last, as the executor may be freed past it. */
inline end_call(t, call, seen) {
    release_arena(t, call);
    atomic { read_modify_write(t, named_calls, order_release, seen, seen - 1); wake(named_calls) };
    atomic { assert(!executor_freed); store(t, holding, order_release, 0); wake(holding) }
}

/** A submitted call's body: re-holds its own arena, which advances nothing, writes its outputs, and
 *  ends. */
inline body(t, call, seen, taken) {
    seen = 0;
    compare_exchange(t, holder, order_acq_rel, seen, call + 1, taken);
    assert(!taken && seen == call + 1);
    if
    :: bump_on_retake -> advance(t, call, seen)
    :: else
    fi;
    atomic {
        store(t, outputs, order_relaxed, tag[call]);
        assert(exporting == 0 && !arena_freed)
    };
    end_call(t, call, seen);
    assert(advances[call] == 1)
}

/** The arena's free: the calls naming it waited out and the export count read, in the order
 *  @c export_check names, then the arena deleted, its memory given back through the executor, and
 *  the executor's count lowered, last unless @c count_lowered_last says otherwise. */
inline arena_free(t, seen) {
    if
    :: export_check == before_wait -> load(t, exports, order_acquire, seen)
    :: else -> seen = 0
    fi;
    if
    :: seen != 0 -> goto contended
    :: else
    fi;
    do
    :: atomic {
           load(t, named_calls, order_acquire, seen);
           if
           :: seen == 0 -> free_waited = true
           :: else
           fi
       };
       if
       :: seen == 0 -> break
       :: else -> wait_on(t, named_calls, seen, false)
       fi
    od;
    if
    :: export_check == after_wait -> load(t, exports, order_acquire, seen)
    :: else
    fi;
    if
    :: seen != 0 -> goto contended
    :: else
    fi;
    atomic { assert(memory != sc || newest_value(exports) == 0); arena_freed = true };
    if
    :: count_lowered_last ->
        assert(!executor_freed);
        read_modify_write(t, arenas, order_release, seen, seen - 1)
    :: else ->
        read_modify_write(t, arenas, order_release, seen, seen - 1);
        assert(!executor_freed)
    fi;
contended:
    skip
}

/** Serves the owner's call once dispatched, holding the pool's view of the submission. */
proctype pool(byte t) {
    int seen;
    bool taken;
    (dispatched != 0);
    if
    :: dispatched == 1 -> body(t, owner_call, seen, taken)
    :: else
    fi
}

/** Reads the generation, submits its call and runs it, stamps its result, and copies it under an
 *  export while the generation reads the stamp. */
proctype binding(byte t) {
    int seen, before, stamped, copied;
    bool taken, opened;
    byte outcome;
    load(t, generation, order_relaxed, before);
    submit(t, binding_call, outcome, seen, taken);
    if
    :: outcome == served ->
        body(t, binding_call, seen, taken);
        stamped = before + 1;
        if
        :: copy_bracket ->
            raise_exports(t, seen);
            read_holder(t, opened, seen);
            if
            :: opened ->
                load(t, generation, order_relaxed, seen);
                if
                :: seen == stamped -> load(t, outputs, order_relaxed, copied); assert(copied == stamped)
                :: else
                fi;
                export_close(t, seen)
            :: else
            fi
        :: else ->
            load(t, generation, order_relaxed, seen);
            if
            :: seen == stamped -> load(t, outputs, order_relaxed, copied); assert(copied == stamped)
            :: else
            fi
        fi
    :: else
    fi;
    binding_done = true
}

/** Opens one export, under @c sc only before the free read no call left, holds it, and closes
 *  it. */
proctype exporter(byte t) {
    int seen;
    bool tried, opened;
    atomic {
        tried = (memory != sc || !free_waited);
        if
        :: tried -> raise_exports(t, seen)
        :: else
        fi
    };
    if
    :: tried ->
        read_holder(t, opened, seen);
        if
        :: opened -> export_close(t, seen)
        :: else
        fi
    :: else
    fi
}

/** Submits its deferred call, refused by the pool or dispatched to it, then frees the arena once
 *  the binding dropped its result. */
proctype owner(byte t) {
    int seen;
    bool taken;
    byte outcome;
    submit(t, owner_call, outcome, seen, taken);
    if
    :: outcome == served ->
        if
        :: atomic { merge_view(t, pool_thread); dispatched = 1 }
        :: end_call(t, owner_call, seen); assert(advances[owner_call] == 1); dispatched = 2
        fi
    :: else -> dispatched = 2
    fi;
    binding_done;
    arena_free(t, seen)
}

/** Frees the executor once: refused while it counts an arena, else it waits for the claim and
 *  deletes it. */
proctype executor(byte t) {
    int seen;
    bool taken;
    load(t, arenas, order_acquire, seen);
    if
    :: seen == 0 ->
        do
        :: seen = 0;
           compare_exchange(t, holding, order_acquire, seen, 1, taken);
           if
           :: taken -> break
           :: else -> wait_on(t, holding, 1, false)
           fi
        od;
        atomic { assert(arena_freed); executor_freed = true }
    :: else
    fi
}

/** The owner's @c arena_init counts the arena on its executor, and every role starts seeing it. */
init {
    int seen;
    atomic {
        read_modify_write(owner_thread, arenas, order_acq_rel, seen, seen + 1);
        merge_view(owner_thread, pool_thread);
        merge_view(owner_thread, binding_thread);
        merge_view(owner_thread, exporter_thread);
        merge_view(owner_thread, executor_thread);
        run pool(pool_thread);
        run binding(binding_thread);
        run exporter(exporter_thread);
        run owner(owner_thread);
        run executor(executor_thread)
    }
}
