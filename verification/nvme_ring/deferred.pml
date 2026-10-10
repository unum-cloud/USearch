/**
 *  @brief `IORING_SETUP_DEFER_TASKRUN`: the kernel posts completions only while the engine is
 *      inside its blocking enter, which changes when things happen and nothing about what is seen.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 cq_tail_order=order_relaxed: `pump` acquires the completion tail, which carries
 *      the kernel's entries; relaxed, the engine reaps a completion stale even when the kernel posts
 *      only inside the engine's enter
 */
#define thread_count 2
#define location_count 8
#define history_depth 4
#define defer_taskrun true
#include "protocol.pml"

proctype engine(byte t) { submit_and_reap(t) }

proctype kernel(byte t) { consume_and_complete(t) }

init { atomic { run engine(0); run kernel(1) } }
