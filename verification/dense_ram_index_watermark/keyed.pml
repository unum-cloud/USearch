/**
 *  @file verification/dense_ram_index_watermark/keyed.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The index's keys beside the publication: `keyed_nodes::assign` writes a row's key right
 *      after the dispense, and the turn holder publishes it into the map before the watermark.
 *
 *  Two inserters and a saver as in `inserters.pml`, both rows carrying one key. The searcher reads
 *  the rows below the watermark it acquired with their keys, then reads row 0's tail once, relaxed,
 *  and locates the ID it names only below its snapshot, the traversal's `neighbor_id >=
 *  first_invisible_id` filter. The finder acquires the watermark, probes the map and drops the nodes
 *  at or above it, as `keyed_nodes::find_each` does.
 *
 *  - A slot the finder matches carries its key and node.
 *  - The finder counts every node below its watermark, exactly.
 *
 *  @budget 2048
 *  @verify pass sc,rc11
 *  @verify fail sc key_before_turn=false: after the turn, a reader locates a row that carries no key
 *      yet
 *  @verify fail sc watermark_filter=false: without the filter it follows a fresher ID, whose turn
 *      has not come and whose row the snapshot did not carry
 *  @verify fail sc publish=after_turn: past the release, the finder acquires a watermark above a node
 *      whose key is not in the map yet, and counts one node short
 */
#define thread_count 5
#define location_count 15
#define history_depth 7
#define named_rows true
#define appended_rows false
#define refined_rows false
#include <dense_ram_index_watermark/protocol.pml>

proctype inserter(byte t) { insert_keyed(t, true) }

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

// `count` of the key every row carries: one per node below the watermark
proctype finder(byte t) {
    byte chain;
    int watermark, seen, found;
    find_keys(t, watermark, chain, seen, found);
    assert(found == watermark)
}

init { atomic { run inserter(0); run inserter(1); run searcher(2); run saver(3); run finder(4) } }
