/**
 *  @file verification/basic_vram_index_reach.pml
 *  @author Ash Vardanian
 *  @date November 17, 2026
 *  @brief The basic VRAM index's reach on every vendor: re-selections over listed IDs, the
 *      back-link launch behind them, and the heal after the check that follows.
 *
 *  @c queue_reach_ in @c vram_nvidia_index.hpp, @c vram_amd_index.hpp and @c vram_apple_index.hpp
 *  launches @c refine_k over the closing check's island nodes, then @c back_link_k on the same
 *  stream. In @c cuda_insert, @c hip_insert and @c metal_build each walker assigns itself an
 *  offset and walks the ID @c listed holds there, every ID once; @c vram_reselect_one and
 *  @c metal_reselect_one rewrite that row, the tail slot relaxed and slot zero last by a release,
 *  and @c vram_thread_back_links and @c metal_thread_back_links thread its picks by that offset.
 *  The back-link launch starts once every walker ended, so it sees every pick, and
 *  @c vram_back_link_one and @c metal_back_link_one recover each walker from @c listed. A search
 *  on another stream acquires the row's slot zero and then reads its tail. The check after the
 *  reach finds the island again when the back-link's re-selection of the root's full row dropped
 *  it, and @c queue_heal_ links it then without zeroing the round's counters, which on Apple the
 *  check itself zeroes, the round's tally having read them first.
 *
 *  Two walkers re-select the island and one more node, each picking the root; a row is its slot
 *  zero and one tail slot, and the connectivity is two.
 *
 *  - A search that acquired a re-selected slot zero reads the tail the same walker wrote.
 *  - The back-link recovers exactly the listed walkers.
 *  - The island ends linked from the root.
 *  - The heal keeps the rows the restitch counted as changed.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc listed_twice=true: two walkers of one launch write one row, and a search reads
 *      one walker's slot zero beside the other's tail
 *  @verify fail rc11 publish_order=order_relaxed: relaxed, slot zero carries no tail, and a search
 *      that read the new slot zero reads the old tail
 *  @verify fail sc recover_listed=false: read as the first ID plus the offset, the picks name the
 *      wave's offsets, and the root links nodes no walker re-selected
 *  @verify fail sc heal_after_reach=false: a full root row whose re-selection drops the island
 *      leaves it unlinked
 *  @verify fail sc heal_zeroes=true: zeroing again, the heal forgets the rows the restitch changed
 */
#define thread_count 3
#define location_count 4
#define history_depth 4
#include <weak_memory.pml>

/** The threads: the two walkers of one launch, then a search on another stream. */
#define walker_count 2
#define reader_thread 2

/** The words: rows 1 and 2, each its slot zero and its tail slot. */
#define row_slot0(id) (((id) - 1) * 2)
#define row_tail(id) (((id) - 1) * 2 + 1)

/** The nodes, the unknown pick of an empty chain, and the slots of a row. */
#define island 1
#define other_node 2
#define unknown 255
#define connectivity 2

/** The values: what a row held before the reach, then what walker w selected. */
#define old_slot0 1
#define old_tail 2
#define picked_slot0(w) (10 + (w))
#define picked_tail(w) (20 + (w))

/** The knobs: the header's choices, each overridden by a verify line replaying its failure. */
#ifndef listed_twice
#define listed_twice false
#endif
#ifndef publish_order
#define publish_order order_release
#endif
#ifndef recover_listed
#define recover_listed true
#endif
#ifndef heal_after_reach
#define heal_after_reach true
#endif
#ifndef heal_zeroes
#define heal_zeroes false
#endif

byte listed[walker_count];                   // the island nodes, by offset
byte link_head = unknown;                    // the root's chain: the last pick of it
byte link_next[walker_count * connectivity]; // per pick, the pick threaded before it
byte walked;                                 // walkers whose launch slice ended
byte recovered;                              // the walkers the back-link took, a bit each
bool island_linked;                          // the root's row holds the island
byte changed_rows = 1;                       // the round's counter, which the restitch already raised

/** @c vram_reselect_one, then @c vram_thread_back_links: one row rewritten, then its pick. */
proctype walker(byte w) {
    byte row = listed[w];
    byte previous;
    store(w, row_tail(row), order_relaxed, picked_tail(w));
    store(w, row_slot0(row), publish_order, picked_slot0(w));
    atomic {
        previous = link_head;
        link_head = w * connectivity;
        link_next[w * connectivity] = previous;
        walked++
    }
}

/** A walk on another stream: slot zero by an acquire, then the tail slot. */
proctype reader(byte t) {
    int first, tail;
    load(t, row_slot0(island), order_acquire, first);
    load(t, row_tail(island), order_acquire, tail);
    if
    :: first >= picked_slot0(0) -> assert(tail == picked_tail(first - picked_slot0(0)))
    :: else
    fi
}

/** The back-link launch behind the walkers, then the check and the heal on the host. */
proctype back_link_and_heal() {
    byte pick, walker_id;
    (walked == walker_count);
    pick = link_head;
    do
    :: pick == unknown -> break
    :: else ->
        walker_id = (recover_listed -> listed[pick / connectivity] : pick / connectivity);
        recovered = recovered | (1 << walker_id);
        pick = link_next[pick]
    od;
    assert(recovered == ((1 << listed[0]) | (1 << listed[1])));
    // an append, or a re-selection of the full row that keeps the island or drops it
    if
    :: island_linked = true
    :: island_linked = false
    fi;
    if
    :: !island_linked && heal_after_reach ->
        if
        :: heal_zeroes -> changed_rows = 0
        :: else
        fi;
        island_linked = true;
        changed_rows++;
        assert(changed_rows == 2)
    :: else
    fi;
    assert(island_linked)
}

init {
    atomic {
        listed[0] = island;
        listed[1] = (listed_twice -> island : other_node);
        seed(row_slot0(island), old_slot0);
        seed(row_tail(island), old_tail);
        seed(row_slot0(other_node), old_slot0);
        seed(row_tail(other_node), old_tail);
        run walker(0);
        run walker(1);
        run reader(reader_thread);
        run back_link_and_heal()
    }
}
