/**
 *  @file verification/nvme_ring.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The io_uring words of `linux_uring_ring_t` behind the ring engines: the engine and the
 *      kernel over both rings, with every completion posted as soon as its entry is consumed.
 *
 *  From `include/usearch/utilities/nvme_device.hpp`: the submission ring's head and tail, the
 *  completion ring's head and tail, and the entries between them, shared with the kernel through
 *  the orders the ring documentation asks for. An engine and a kernel over two rings of depth two,
 *  and three requests, so each ring wraps once. The kernel refuses at most two calls.
 *
 *  - Every entry the kernel reads carries the request claimed at that index.
 *  - Every request ends reaped, withdrawn or dropped, each once.
 *  - The outstanding count equals the requests not yet ended.
 *  - An abandon the kernel waited out leaves nothing in flight, and a head never passes its tail.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 sq_tail_order=order_relaxed: relaxed, the kernel consumes an entry before its
 *      payload landed
 *  @verify fail rc11 cq_tail_order=order_relaxed: relaxed, the engine reaps a completion stale
 *  @verify fail sc abandon_waits=false: dropping at once, a read of the abandoned batch is still in
 *      flight when `submit` returns, and the engine polls it later
 */
#define thread_count 2
#define location_count 8
#define history_depth 6
#include <weak_memory.pml>

// The words: the four ring words, then the entries of each ring.
#define sq_head 0
#define sq_tail 1
#define cq_head 2
#define cq_tail 3
#define sqe(i) (4 + (i))
#define cqe(i) (6 + (i))
#define heads_behind_tails \
    (newest_value(sq_head) <= newest_value(sq_tail) && newest_value(cq_head) <= newest_value(cq_tail))

// The sizes: the rings' depth, the requests, one more than the depth so each ring wraps, and the
// calls the kernel refuses, so it may refuse an abandon's wait as well as the call that abandons.
#define depth 2
#define requests 3
#define refusals 2

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef sq_tail_order
#define sq_tail_order order_release
#endif
#ifndef cq_tail_order
#define cq_tail_order order_acquire
#endif
#ifndef abandon_waits
#define abandon_waits true
#endif

// The ghosts: each request's fate, numbered from one so an unwritten entry names none, the request
// claimed at each submission index, and the requests whose fate has not ended.
#define fate_claimed 1
#define fate_consumed 2
#define fate_posted 3
#define fate_reaped 4
#define fate_withdrawn 5
#define fate_dropped 6
byte fate[requests + 1];
byte request_at[requests];
byte unsettled;

// The ring's own members, which only the engine touches: `pending_tail_` and `outstanding_`.
byte pending_tail, outstanding;

// `io_uring_enter`: the engine waits while the kernel serves or refuses the call.
bool entering, refusal;
byte refused_calls;

// A load across `io_uring_enter`, which reads the newest write and merges the stamp it carries
inline load_newest(t, l, order, out) {
    do
    :: load(t, l, order, out);
        if
        :: seen_write(t, l) == newest_write(l) -> break
        :: else
        fi
    od
}

// `enter_`: the tail published when the claims moved past it, then the call
inline enter(t, refused) {
    if
    :: pending_tail != newest_value(sq_tail) -> store(t, sq_tail, sq_tail_order, pending_tail)
    :: else
    fi;
    entering = true;
    atomic { !entering -> refused = refusal; refusal = false }
}

// `poll_completions`: the tail with acquire, every completion up to it, then the head with a
// release, stored only when it moved, since the same head tells the kernel nothing new
inline poll(t, head, tail, seen) {
    head = newest_value(cq_head);
    load(t, cq_tail, cq_tail_order, tail);
    do
    :: head < tail ->
        load(t, cqe(head % depth), order_relaxed, seen);
        assert(seen == request_at[head] && fate[seen] == fate_posted);
        atomic { fate[seen] = fate_reaped; unsettled-- };
        head++;
        outstanding--
    :: else -> break
    od;
    if
    :: head != newest_value(cq_head) -> store(t, cq_head, order_release, head)
    :: else
    fi
}

// `abandon_`: the entries past the head withdrawn and the tail stored back on it, then rounds that
// drop the completions at hand and wait for the rest, stored only when the head moved, as in `poll`
inline abandon(t, head, tail, index, refused) {
    load_newest(t, sq_head, order_acquire, head);
    atomic {
        for (index : head .. pending_tail - 1) {
            assert(fate[request_at[index]] == fate_claimed);
            fate[request_at[index]] = fate_withdrawn;
            unsettled--
        }
    };
    outstanding = outstanding - (pending_tail - head);
    pending_tail = head;
    store(t, sq_tail, order_release, head);
    refused = false;
    do
    :: load(t, cq_tail, cq_tail_order, tail);
        atomic {
            for (index : newest_value(cq_head) .. tail - 1) {
                assert(fate[request_at[index]] == fate_posted);
                fate[request_at[index]] = fate_dropped;
                unsettled--
            }
        };
        outstanding = outstanding - (tail - newest_value(cq_head));
        if
        :: tail != newest_value(cq_head) -> store(t, cq_head, order_release, tail)
        :: else
        fi;
        if
        :: outstanding == 0 || !abandon_waits -> break
        :: else ->
            enter(t, refused);
            if
            :: refused -> break
            :: else -> (newest_value(cq_tail) - newest_value(cq_head) >= outstanding)
            fi
        fi
    od;
    assert(outstanding == unsettled && (refused || unsettled == 0) && heads_behind_tails)
}

// `submit`: `enter_`, then under a blocking wait a completion in the ring, as `enter_(1)` or the
// capped monitor before it waits; a refused call abandons
inline submit(t, blocking, head, tail, index, refused) {
    if
    :: outstanding > 0 ->
        enter(t, refused);
        if
        :: refused -> abandon(t, head, tail, index, refused)
        :: else ->
            if
            :: blocking -> (newest_value(cq_tail) != newest_value(cq_head))
            :: else
            fi
        fi
    :: else
    fi
}

// the engine: `next_submission` takes the head with acquire and the entry is written in place, a
// nonblocking `submit` may follow any claim, and a full ring or the last request polls
proctype engine(byte t) {
    int head, tail, seen;
    byte request = 1, index;
    bool refused;
    do
    :: request <= requests ->
        load(t, sq_head, order_acquire, head);
        if
        :: outstanding < depth && pending_tail - head < depth ->
            atomic { request_at[pending_tail] = request; fate[request] = fate_claimed; unsettled++ };
            store(t, sqe(pending_tail % depth), order_relaxed, request);
            pending_tail++;
            outstanding++;
            request++
        :: else ->
            submit(t, true, head, tail, index, refused);
            poll(t, head, tail, seen)
        fi
    :: pending_tail != newest_value(sq_tail) -> submit(t, false, head, tail, index, refused)
    :: request > requests && outstanding > 0 ->
        submit(t, true, head, tail, index, refused);
        poll(t, head, tail, seen)
    :: request > requests && outstanding == 0 -> break
    od;
    assert(unsettled == 0 && heads_behind_tails)
}

// the kernel: the submission ring only inside a call, the completion ring at any step
proctype kernel(byte t) {
    int tail, head, seen;
    byte consumed, completed;
    byte served[requests];
end:
    do
    :: entering ->
        if
        :: true ->
            // the newest tail with acquire, the entries, the head with release
            load_newest(t, sq_tail, order_acquire, tail);
            do
            :: consumed < tail ->
                load(t, sqe(consumed % depth), order_relaxed, seen);
                assert(seen == request_at[consumed] && fate[seen] == fate_claimed);
                fate[seen] = fate_consumed;
                served[consumed] = seen;
                consumed++;
                store(t, sq_head, order_release, consumed)
            :: else -> break
            od
        :: refused_calls < refusals -> refused_calls++; refusal = true
        fi;
        entering = false
    :: completed < consumed && completed - newest_value(cq_head) < depth ->
        // room from the head with acquire, the entry, the tail with release
        load(t, cq_head, order_acquire, head);
        if
        :: completed - head < depth ->
            store(t, cqe(completed % depth), order_relaxed, served[completed]);
            fate[served[completed]] = fate_posted;
            completed++;
            store(t, cq_tail, order_release, completed)
        :: else
        fi
    od
}

init { atomic { run engine(0); run kernel(1) } }
