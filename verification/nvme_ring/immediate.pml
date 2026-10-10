/**
 *  @brief The engine and the kernel over both rings, with every completion posted as soon as its
 *      entry is consumed.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 sq_tail_order=order_relaxed: `enter` publishes the submission tail with a
 *      release that carries the entries; relaxed, the kernel consumes an entry before its payload
 *      landed
 *  @verify fail rc11 cq_tail_order=order_relaxed: `pump` acquires the completion tail, which carries
 *      the kernel's entries; relaxed, the engine reaps a completion stale
 */
#define thread_count 2
#define location_count 8
#define history_depth 4
#define defer_taskrun false
#include "protocol.pml"

proctype engine(byte t) { submit_and_reap(t) }

proctype kernel(byte t) { consume_and_complete(t) }

init { atomic { run engine(0); run kernel(1) } }
