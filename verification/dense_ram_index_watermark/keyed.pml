/**
 *  @brief The sharded index's key column beside the publication: `on_assigned` writes a row's key
 *      right after the dispense, so every row a reader locates carries its key.
 *
 *  Two inserters and a saver as in `inserters.pml`. The searcher reads the rows below the watermark
 *  it acquired with their keys, then reads row 0's tail once, relaxed, and locates the ID it names
 *  only below its snapshot, the traversal's `neighbor_id >= first_invisible_id` filter.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc key_before_turn=false: `on_assigned` stores the key before the walk and the
 *      publication; after the turn, a reader locates a row that carries no key yet
 *  @verify fail sc watermark_filter=false: the walk locates only IDs below its snapshot; without
 *      the filter it follows a fresher ID, whose turn has not come and whose row the snapshot did
 *      not carry
 */
#define thread_count 4
#define location_count 9
#define history_depth 7
#define named_rows true
#define appended_rows false
#define refined_rows false
#include "protocol.pml"

proctype inserter(byte t) { insert(t) }

// the walk: one relaxed read of row 0's tail, and the ID it holds located only below the snapshot,
// since a fresher one, read from a re-linked slot, has no happens-before edge to its row yet
proctype searcher(byte t) {
    int watermark, found, walked;
    search(t, watermark);
    if
    :: watermark > 0 ->
        load(t, row_tail(0), order_relaxed, found);
        assert(id_of(found) < newest_value(next_dispensed_id)); // a row names dispensed IDs alone
        if
        :: !watermark_filter || id_of(found) < watermark ->
            assert(memory != sc || newest_value(last_visible_id) > id_of(found)); // a located ID took its turn
            read_published(t, id_of(found), walked)
        :: else
        fi
    :: else
    fi
}

proctype saver(byte t) { save(t) }

init { atomic { run inserter(0); run inserter(1); run searcher(2); run saver(3) } }
