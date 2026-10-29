/**
 *  @file verification/basic_ram_index_batches/serial.pml
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief A basic index batch publishes whole: one add dispenses its range, one turn publishes it.
 *      Other batches' walks admit its rows earlier through the `wired_k` bit of each row's state.
 *
 *  `basic::ram_index::insert` from `include/usearch/basic/ram_index.hpp`. Two inserters bring two
 *  rows each. One add dispenses an inserter's range, then each row walks, writes its value and
 *  neighbors plain, and sets its `wired_k` bit with a release; one turn from the range's first ID
 *  to its end files both keys and then publishes both rows.
 *
 *  - The searcher acquires the watermark and reads every row below it wired and whole.
 *  - The finder counts exactly the rows below the watermark it acquired, each one whole.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 range_turn=false: a turn per row lets the searcher see half a batch
 *  @verify fail sc,rc11 wired_check=false: unchecked, a later batch's walk reads an earlier batch's
 *      row before its inserter wrote it
 *  @verify fail rc11 wired_order=order_relaxed: relaxed, the walk reads the bit set and the row
 *      unwritten
 *  @verify fail rc11 admit_order=order_relaxed: relaxed, the walk reads the bit set and the row
 *      unwritten
 *  @verify fail sc,rc11 keys_in_turn=false: filed past the release, the finder acquires the watermark
 *      past a row whose key is not filed yet, and counts one short
 */
#define thread_count 4
#define location_count 15
#define history_depth 5
#include <basic_ram_index_batches/protocol.pml>

proctype inserter(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

proctype finder(byte t) { find_each(t) }

init { atomic { run inserter(0); run inserter(1); run searcher(2); run finder(3) } }
