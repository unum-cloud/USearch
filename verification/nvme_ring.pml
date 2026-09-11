/**
 *  The io_uring words of `dense_nvme_index` from `include/usearch/dense/nvme_index.hpp`: the
 *  submission ring's head and tail, the completion ring's head and tail, and the entries
 *  between them, shared with the kernel through the orders the ring documentation asks for.
 *
 *  An engine and a kernel over two rings of depth two, and three requests, so each ring wraps
 *  once. The engine claims a submission slot after an acquire load of the submission head,
 *  writes the entry relaxed, and publishes the tail with a release; it reaps by waiting on the
 *  completion tail, loading it with acquire, reading each entry relaxed, and publishing the
 *  completion head with a release. The kernel is the mirror image: an acquire load of the
 *  submission tail, relaxed reads of the entries, a release store of the submission head; an
 *  acquire load of the completion head for room, relaxed writes of the entries, a release
 *  store of the completion tail. Every entry carries its request's sequence number.
 *
 *  Invariants:
 *  - payload: every entry read carries the sequence number expected at that index, never the
 *    initial value and never a stale one from the previous lap. `-Dwithout_sq_tail_release`
 *    lets the kernel read an entry the engine has not written, `-Dwithout_cq_tail_acquire` lets
 *    the engine read a completion the kernel has not written; both fail under views. The
 *    acquire on the submission head and the release on the completion head guard the shape
 *    where a slot is reused before its reader is done, which RC11 forbids by construction, so
 *    no variant can fail on them here; they stay in the header because the kernel is not a C++
 *    thread and the ring documentation asks for them;
 *  - accounting: every request is consumed and reaped exactly once, and a head never passes
 *    its tail;
 *  - `-Dscenario=deferred` is `IORING_SETUP_DEFER_TASKRUN`: the kernel posts completions only
 *    while the engine is inside its blocking enter, which changes when things happen and
 *    nothing about what is seen.
 */
#include "weak_memory.pml"

// The knob's values are integers, so a typo fails the range check below.
#define immediate 1
#define deferred 2
#ifndef scenario
#define scenario immediate
#endif
#if scenario < immediate || scenario > deferred
#error "scenario is immediate or deferred"
#endif

// The threads, by role.
#define engine_thread 0
#define kernel_thread 1

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

#ifdef without_sq_tail_release
#define sq_tail_order order_relaxed
#else
#define sq_tail_order order_release
#endif

#ifdef without_cq_tail_acquire
#define cq_tail_order order_relaxed
#else
#define cq_tail_order order_acquire
#endif

bool entering; // the engine is inside a blocking enter, which is when a deferred ring posts

// pump and pop: the wait on the completion tail, then every completion up to the tail: nvme_index.hpp:872-879, 894-898
inline reap() {
    entering = true;
    (newest_value(cq_tail) != reaped);
    entering = false;
    load(engine_thread, cq_tail, cq_tail_order, tail);
    do
    :: reaped != tail ->
        load(engine_thread, cqe(reaped % depth), order_relaxed, seen);
        assert(seen == reaped + 1);
        reaped++;
        store(engine_thread, cq_head, order_release, reaped)
    :: else -> break
    od
}

active proctype engine() {
    int head, tail, seen;
    byte posted, reaped;
    do
    :: posted < requests ->
        // next_submission: the head with acquire, then the entry in place; enter publishes the tail: nvme_index.hpp:838-843, 849-851
        load(engine_thread, sq_head, order_acquire, head);
        if
        :: posted - head < depth ->
            store(engine_thread, sqe(posted % depth), order_relaxed, posted + 1);
            posted++;
            store(engine_thread, sq_tail, sq_tail_order, posted)
        :: else -> reap()
        fi
    :: posted == requests && reaped < requests -> reap()
    :: reaped == requests -> break
    od
}

active proctype kernel() {
    int tail, head, seen;
    byte consumed, completed;
    // the waits sit in the guards, so the kernel never commits to one ring while it owes the other
    do
    :: consumed < requests && newest_value(sq_tail) != consumed ->
        // the kernel's side of the submission ring: the tail with acquire, the entries, the head with release
        load(kernel_thread, sq_tail, order_acquire, tail);
        do
        :: consumed != tail ->
            load(kernel_thread, sqe(consumed % depth), order_relaxed, seen);
            assert(seen == consumed + 1);
            consumed++;
            store(kernel_thread, sq_head, order_release, consumed)
        :: else -> break
        od
#if scenario == deferred
    :: completed < consumed && entering && completed - newest_value(cq_head) < depth ->
#else
    :: completed < consumed && completed - newest_value(cq_head) < depth ->
#endif
        // the kernel's side of the completion ring: room from the head with acquire, the entry, the tail with release
        load(kernel_thread, cq_head, order_acquire, head);
        if
        :: completed - head < depth ->
            store(kernel_thread, cqe(completed % depth), order_relaxed, completed + 1);
            completed++;
            store(kernel_thread, cq_tail, order_release, completed)
        :: else
        fi
    :: consumed == requests && completed == requests -> break
    od;
    assert(newest_value(sq_head) <= newest_value(sq_tail) && newest_value(cq_head) <= newest_value(cq_tail))
}
