/**
 *  The io_uring words of `dense_uring_ring_t` behind `dense_nvme_index`, from
 *  `include/usearch/dense/nvme_index.hpp`: the submission ring's head and tail, the completion
 *  ring's head and tail, and the entries between them, shared with the kernel through the orders
 *  the ring documentation asks for.
 *
 *  An engine and a kernel over two rings of depth two, and three requests, so each ring wraps
 *  once. The engine claims a submission slot after an acquire load of the submission head, writes
 *  the entry relaxed, and publishes the tail with a release; it reaps by waiting on the completion
 *  tail, loading it with acquire, reading each entry relaxed, and publishing the completion head
 *  with a release. The kernel is the mirror image: an acquire load of the submission tail, relaxed
 *  reads of the entries, a release store of the submission head; an acquire load of the completion
 *  head for room, relaxed writes of the entries, a release store of the completion tail. Every
 *  entry carries its request's sequence number, and every entry read carries the one expected at
 *  that index, never the initial value and never a stale one from the previous lap. Every request
 *  is consumed and reaped exactly once, and a head never passes its tail.
 *
 *  The acquire on the submission head and the release on the completion head guard the shape
 *  where a slot is reused before its reader is done, which RC11 forbids by construction, so no knob
 *  can fail on them here: with no eventual visibility in the model either, a relaxed head read can
 *  stay stale forever, and the run ends with a stuck engine rather than a failed assertion. They
 *  stay in the header, since the kernel is not a C++ thread.
 *
 *  Every scenario beside this file defines `defer_taskrun`, whether the kernel posts completions
 *  only while the engine is inside its blocking enter.
 */
#include "../weak_memory.pml"

// The words: the four ring words, then the entries of each ring.
#define sq_head 0
#define sq_tail 1
#define cq_head 2
#define cq_tail 3
#define sqe(i) (4 + (i))
#define cqe(i) (6 + (i))

// The sizes: the rings' depth, and the requests, one more than the depth so each ring wraps.
#define depth 2
#define requests 3

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef sq_tail_order
#define sq_tail_order order_release
#endif
#ifndef cq_tail_order
#define cq_tail_order order_acquire
#endif

bool entering; // the engine is inside a blocking enter, which is when a deferred ring posts

// `pump` and `pop`: the wait on the completion tail, then every completion up to the tail
inline reap(t, reaped, tail, seen) {
    entering = true;
    (newest_value(cq_tail) != reaped);
    entering = false;
    load(t, cq_tail, cq_tail_order, tail);
    do
    :: reaped != tail ->
        load(t, cqe(reaped % depth), order_relaxed, seen);
        assert(seen == reaped + 1);
        reaped++;
        store(t, cq_head, order_release, reaped)
    :: else -> break
    od
}

// the engine: `next_submission` takes the head with acquire and writes the entry in place, `enter`
// publishes the tail, and a full ring or the last request reaps
inline submit_and_reap(t) {
    int head, tail, seen;
    byte posted, reaped;
    do
    :: posted < requests ->
        load(t, sq_head, order_acquire, head);
        if
        :: posted - head < depth ->
            store(t, sqe(posted % depth), order_relaxed, posted + 1);
            posted++;
            store(t, sq_tail, sq_tail_order, posted)
        :: else -> reap(t, reaped, tail, seen)
        fi
    :: posted == requests && reaped < requests -> reap(t, reaped, tail, seen)
    :: reaped == requests -> break
    od
}

// the kernel: the waits sit in the guards, so it never commits to one ring while it owes the other
inline consume_and_complete(t) {
    int tail, head, seen;
    byte consumed, completed;
    do
    :: consumed < requests && newest_value(sq_tail) != consumed ->
        // the submission ring: the tail with acquire, the entries, the head with release
        load(t, sq_tail, order_acquire, tail);
        do
        :: consumed != tail ->
            load(t, sqe(consumed % depth), order_relaxed, seen);
            assert(seen == consumed + 1);
            consumed++;
            store(t, sq_head, order_release, consumed)
        :: else -> break
        od
    :: completed < consumed && (!defer_taskrun || entering) && completed - newest_value(cq_head) < depth ->
        // the completion ring: room from the head with acquire, the entry, the tail with release
        load(t, cq_head, order_acquire, head);
        if
        :: completed - head < depth ->
            store(t, cqe(completed % depth), order_relaxed, completed + 1);
            completed++;
            store(t, cq_tail, order_release, completed)
        :: else
        fi
    :: consumed == requests && completed == requests -> break
    od;
    assert(newest_value(sq_head) <= newest_value(sq_tail) && newest_value(cq_head) <= newest_value(cq_tail))
}
