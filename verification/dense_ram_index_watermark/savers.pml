/**
 *  @brief Two concurrent saves beside two inserters, each with its own plan and its own tallies.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 back_link_fence=false: the release fence heading `back_link_neighbors` carries
 *      the relaxed tail stores into either save's re-check; without it, a settled copy holds a new
 *      tail beside an old slot zero
 */
#define thread_count 4
#define location_count 7
#define history_depth 4
#define named_rows false
#define appended_rows false
#include "protocol.pml"

proctype inserter(byte t) { insert(t) }

proctype saver(byte t) { save(t) }

init { atomic { run inserter(0); run inserter(1); run saver(2); run saver(3) } }
