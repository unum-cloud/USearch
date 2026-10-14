/**
 *  @brief A basic index batch publishes whole: one add dispenses its range, one turn publishes it.
 *      Other batches' walks admit its rows earlier through the `wired_k` bit of each row's state.
 *
 *  `basic::ram_index::insert` from `include/usearch/basic/ram_index.hpp`. Two inserters bring two
 *  rows each. One add dispenses an inserter's range, then each row walks, writes its value and
 *  neighbors plain, and sets its `wired_k` bit with a release; one turn from the range's first ID
 *  to its end publishes both rows, and the keys land after it. A row's walk draws below its own ID.
 *  Its own batch's earlier row it wired itself, so that one needs no acquire; the other batch's
 *  rows, when that batch dispensed first, `dispensed_admission_t` admits below the walk's snapshot
 *  outright, and past it only once an acquire of `wired_k` shows them wired. The searcher acquires
 *  the watermark, which never lies strictly inside a range, and reads every row below it wired and
 *  whole; under `sc` it also finds no key naming a row at or above the watermark.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 range_turn=false: one turn per range keeps a batch whole; a turn per row
 *      lets the searcher see half a batch
 *  @verify fail sc,rc11 wired_check=false: a walk admits a row past its snapshot only once an
 *      acquire of its `wired_k` bit shows it wired; unchecked, a later batch's walk reads an earlier
 *      batch's row before its inserter wrote it
 *  @verify fail rc11 wired_order=order_relaxed: `wired_k` is set with a release, which carries the
 *      row's plain value and neighbors to the walk that admits it; relaxed, the walk reads the bit
 *      set and the row unwritten
 *  @verify fail rc11 admit_order=order_relaxed: `admits` loads `wired_k` with an acquire, the other
 *      half of the pair; relaxed, the walk reads the bit set and the row unwritten
 *  @verify fail sc keys_after_turn=false: the keys land after the turn, so a key `find` returns
 *      names a published row; before it, a lookup names a row the watermark does not cover
 */
#define thread_count 3
#define location_count 10
#define history_depth 5
#include "protocol.pml"

proctype inserter(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

init { atomic { run inserter(0); run inserter(1); run searcher(2) } }
