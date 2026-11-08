/**
 *  @file verification/dense_vram_index_sweep.pml
 *  @author Ash Vardanian
 *  @date November 6, 2026
 *  @brief The NVIDIA and AMD refines' Reverse Sweep: each walk's first write, then tiles of target
 *      rows through the count, offset, scatter and merge launches a stream orders.
 *
 *  From @c include/usearch/dense/vram_index.cuh: @c vram_count_back_links adds per link into a
 *  tile, @c vram_offset_back_links turns the counts into run starts and may shorten the tile, and
 *  the host reads what it kept behind the launch's event, then @c vram_scatter_back_links takes a
 *  tape position per link and @c vram_merge_back_links reads each run. Two warps run every launch,
 *  and the boundary between launches is the stream's: every later launch and copy sees every
 *  earlier write. Rows 0 and 2 are walked, row 1 is not, and row 2 links to rows 0 and 1, row 0 to
 *  row 1. @c tape_capacity of 2 ends the first tile at row 0, so the links into row 1 cross into a
 *  second tile, reusing the first tile's offsets from their first word.
 *
 *  - Every link in a tile lands in exactly one tape slot, and a link past a tile lands in the next.
 *  - A run holds exactly its target's links, less those past the tape.
 *  - A walk reads each slot of a row another warp writes old or merged.
 *  - A row counts as changed once, and only a walked one, so @c changed never passes @c reselected.
 *
 *  @verify pass sc,rc11,far
 *  @verify pass sc,rc11,far tape_capacity=2
 *  @verify pass sc,rc11 tape_capacity=1
 *  @verify fail far offsets_add=add: a relaxed add whose result nothing reads may lower to a posted
 *      one, in flight past the launch's end, so the offset kernel reads a short count, and a run
 *      comes up short
 *  @verify fail sc,rc11 offsets_add=plain: two warps lose a count, so two links share a slot
 *  @verify fail rc11 launch_barrier=false: a launch reading its predecessor's words without the
 *      stream's edge reads a count it was never told is final
 *  @verify fail sc,rc11 clear_offsets=false: the count starts from the previous pass's run ends,
 *      and the tile ends a row early
 *  @verify fail sc,rc11 changed_bit=false: row 0 gains on both writes and counts twice, passing the
 *      rows the pass re-selected
 *  @verify fail sc,rc11 walked_only=false: row 1 gains a merge no walk asked for, passing the rows
 *      the pass re-selected
 */
#define thread_count 3
#define location_count 10
#define history_depth 12
#include <weak_memory.pml>

/** The words: row 0's slot zero and tail slot, the tile's offsets by row in the tile, the tape, the
 *  rows and links the offset kernel kept, and the changed bits. */
#define row_first 0
#define row_tail 1
#define incoming_offsets(position) (2 + (position))
#define back_links(position) (4 + (position))
#define swept_rows 7
#define swept_links 8
#define changed_bits 9

/** The kernels a launch runs. */
#define walk_k 1
#define count_k 2
#define offset_k 3
#define scatter_k 4
#define merge_k 5
#define done_k 6

/** The links, by source row and target row, as bits. */
#define target_rows 2
#define link_count 3
#define links_into(row) ((row) == 0 -> 4 : 5)
#define links_count(row) ((row) == 0 -> 1 : 2)
#define run_length(row) (links_count(row) < tape_capacity -> links_count(row) : tape_capacity)
#define bit_of(source) (1 << (source))
#define merged_bit(row, source) (1 << ((row) * 3 + (source)))
#define every_link (merged_bit(0, 2) | merged_bit(1, 0) | merged_bit(1, 2))

/** A tape slot names its source or nothing yet, and row 0 holds its old members until its first
 *  write merges them. */
#define hole 0
#define link_of(source) ((source) + 1)
#define source_of(value) ((value) - 1)
#define old_first 1
#define old_tail 2
#define merged_first 3
#define merged_tail 4

/** The count's forms: the kernel's returning add, a posted one, or a plain read then write. */
#define fetch_add 0
#define add 1
#define plain 2

/** The knobs: the header's choices, each overridden by a <tt>@verify</tt> line replaying them. */
#ifndef offsets_add
#define offsets_add fetch_add
#endif
#if offsets_add != add && offsets_add != fetch_add && offsets_add != plain
#error "offsets_add is add, fetch_add or plain"
#endif
#ifndef launch_barrier
#define launch_barrier true
#endif
#ifndef clear_offsets
#define clear_offsets true
#endif
#ifndef changed_bit
#define changed_bit true
#endif
#ifndef walked_only
#define walked_only true
#endif
#ifndef tape_capacity
#define tape_capacity link_count
#endif

byte kernel;             // the launch running
byte launched;           // the launches so far
byte arrivals;           // the warps that finished the running launch
byte tile_first;         // `sweep_.first`, which the launch carries
byte tile_bound;         // `sweep_.bound`, the window before the offset kernel and its end after
byte tape_size;          // the tape's links, which the scatter's launch carries
byte scattered[link_count]; // the links written to each tape slot, a ghost over plain writes
byte changed_nodes;      // the telemetry block, a tally the last copy reads
int merged;              // the links some merge read, a ghost
#define reselected 2

/** The stream's edge: whatever ran before it is visible to whatever runs after it. */
inline stream_edge() {
    if
    :: launch_barrier -> atomic { see_newest(0); see_newest(1); see_newest(2) }
    :: else -> skip
    fi
}

/** Queues kernel @p k and drains it, as the host's next step behind the launch's event. */
inline launch(k) {
    atomic {
        kernel = k;
        arrivals = 0;
        launched = launched + 1
    };
    (arrivals == 2);
    stream_edge()
}

/** A row that gained counts as changed once, by the bit its walk or its merge set first. */
inline count_changed(t, mask, seen) {
    read_modify_write(t, changed_bits, order_relaxed, seen, seen | mask);
    if
    :: !changed_bit || (seen & mask) == 0 -> atomic { changed_nodes = changed_nodes + 1 }
    :: else -> skip
    fi
}

/** @c vram_refine_one 's first write: warp 0 merges row 0's picks slot by slot in either order, and
 *  warp 1 walks over the row meanwhile, then each counts its row. */
inline first_write(t, mask, first, tail, seen) {
    if
    :: t == 0 ->
        if
        :: true -> store(t, row_tail, order_relaxed, merged_tail); store(t, row_first, order_relaxed, merged_first)
        :: true -> store(t, row_first, order_relaxed, merged_first); store(t, row_tail, order_relaxed, merged_tail)
        fi
    :: else ->
        load(t, row_first, order_relaxed, first);
        load(t, row_tail, order_relaxed, tail);
        assert(first == old_first || first == merged_first);
        assert(tail == old_tail || tail == merged_tail)
    fi;
    count_changed(t, mask, seen)
}

#define in_tile(row) ((row) >= tile_first && (row) < tile_bound)

/** The count of one link into the row at @p position , in the form @c offsets_add names. */
inline count_link(t, position, counted) {
    if
    :: offsets_add == add ->
        landed(t);
        add_no_return(t, incoming_offsets(position), order_relaxed, 1)
    :: offsets_add == fetch_add ->
        read_modify_write(t, incoming_offsets(position), order_relaxed, counted, counted + 1)
    :: offsets_add == plain ->
        load(t, incoming_offsets(position), order_relaxed, counted);
        store(t, incoming_offsets(position), order_relaxed, counted + 1)
    fi
}

/** @c vram_count_back_links: warp 0 reads row 0's link into row 1, and warp 1 reads row 2's links
 *  into rows 0 and 1. */
inline count_back_links(t, counted) {
    if
    :: t == 1 && in_tile(0) -> count_link(t, 0 - tile_first, counted)
    :: else -> skip
    fi;
    if
    :: in_tile(1) -> count_link(t, 1 - tile_first, counted)
    :: else -> skip
    fi
}

/** @c vram_offset_back_links on one lane: the tile ends before the first row passing the tape, and
 *  a lone row past it keeps the tape's worth. */
inline offset_back_links(t, rows, offset, count) {
    rows = 0;
    offset = 0;
    do
    :: tile_first + rows == tile_bound -> break
    :: else ->
        load(t, incoming_offsets(rows), order_relaxed, count);
        if
        :: offset + count > tape_capacity && rows > 0 -> break
        :: offset + count > tape_capacity && rows == 0 -> count = tape_capacity
        :: else -> skip
        fi;
        store(t, incoming_offsets(rows), order_relaxed, offset);
        offset = offset + count;
        rows = rows + 1
    od;
    store(t, swept_rows, order_relaxed, rows);
    store(t, swept_links, order_relaxed, offset)
}

/** The scatter of @p source 's link into @p row : a slot takes one link, and a position past the
 *  tape is left out. */
inline scatter_link(t, source, row, position, link) {
    if
    :: in_tile(row) ->
        read_modify_write(t, incoming_offsets((row) - tile_first), order_relaxed, position, position + 1);
        if
        :: position < tape_size ->
            atomic {
                scattered[position] = scattered[position] + 1;
                assert(scattered[position] == 1)
            };
            link = link_of(source);
            store(t, back_links(position), order_relaxed, link)
        :: else -> skip
        fi
    :: else -> skip
    fi
}

#define capped(position) ((position) < tape_size -> (position) : tape_size)

/** @c vram_merge_back_links over the row at position @p t of the tile: its run holds its own links,
 *  each once, cut only by the tape, and a walked row that gained counts as changed. */
inline merge_back_links(t, row, first, bound, slot, value, sources, seen) {
    if
    :: tile_first + t < tile_bound ->
        row = tile_first + t;
        first = 0;
        if
        :: t > 0 ->
            load(t, incoming_offsets(t - 1), order_relaxed, first);
            first = capped(first)
        :: else -> skip
        fi;
        load(t, incoming_offsets(t), order_relaxed, bound);
        bound = capped(bound);
        assert(bound - first == run_length(row));
        sources = 0;
        slot = first;
        do
        :: slot < bound ->
            load(t, back_links(slot), order_relaxed, value);
            assert(value != hole);
            assert((links_into(row) & bit_of(source_of(value))) != 0);
            assert((sources & bit_of(source_of(value))) == 0);
            sources = sources | bit_of(source_of(value));
            atomic { merged = merged | merged_bit(row, source_of(value)) };
            slot = slot + 1
        :: else -> break
        od;
        if
        :: row == 0 || !walked_only -> count_changed(t, bit_of(row), seen)
        :: else -> skip
        fi
    :: else -> skip
    fi
}

/** A warp of every launch: warp 0 owns source row 0, warp 1 source row 2. */
proctype warp(byte t) {
    int seen, counted, first, bound, value, rows, offset, count, position;
    byte row, slot, sources, launches;
    do
    :: (launched != launches) ->
        launches = launched;
        if
        :: kernel == walk_k -> first_write(t, bit_of(2 * t), first, bound, seen)
        :: kernel == count_k -> count_back_links(t, counted)
        :: kernel == offset_k ->
            if
            :: t == 0 -> offset_back_links(t, rows, offset, count)
            :: else -> skip
            fi
        :: kernel == scatter_k ->
            if
            :: t == 0 -> scatter_link(t, 0, 1, position, value)
            :: else ->
                scatter_link(t, 2, 0, position, value);
                scatter_link(t, 2, 1, position, value)
            fi
        :: kernel == merge_k -> merge_back_links(t, row, first, bound, slot, value, sources, seen)
        :: kernel == done_k -> break
        fi;
        atomic { arrivals = arrivals + 1 }
    od;
    landed(t)
}

/** The host: the walks, then the tiles of the plan's @c advance, reading what each kept. */
proctype host(byte t) {
    int rows, links;
    byte first_row;
    launch(walk_k);
    do
    :: first_row == target_rows -> break
    :: else ->
        tile_first = first_row;
        tile_bound = target_rows;
        if
        :: clear_offsets ->
            store(t, incoming_offsets(0), order_relaxed, 0);
            if
            :: tile_bound - tile_first > 1 -> store(t, incoming_offsets(1), order_relaxed, 0)
            :: else -> skip
            fi
        :: else -> skip
        fi;
        stream_edge();
        launch(count_k);
        launch(offset_k);
        load(t, swept_rows, order_relaxed, rows);
        load(t, swept_links, order_relaxed, links);
        tile_bound = tile_first + rows;
        tape_size = links;
        assert(tile_first > 0 || rows == (tape_capacity < link_count -> 1 : target_rows));
        d_step {
            scattered[0] = 0;
            scattered[1] = 0;
            scattered[2] = 0
        };
        launch(scatter_k);
        launch(merge_k);
        first_row = tile_bound
    od;
    assert(changed_nodes <= reselected);
    assert(tape_capacity < 2 || merged == every_link);
    atomic {
        kernel = done_k;
        launched = launched + 1
    }
}

/** the offsets as an earlier pass left them, and row 0 before its first write */
init {
    atomic {
        seed(row_first, old_first);
        seed(row_tail, old_tail);
        seed(incoming_offsets(0), 1);
        seed(incoming_offsets(1), 1);
        run warp(0); run warp(1); run host(2)
    }
}
