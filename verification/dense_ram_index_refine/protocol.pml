/**
 *  @file verification/dense_ram_index_refine/protocol.pml
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief One tile of a refinement's sweep in the dense RAM index.
 *
 *  @c refine_plan_t collects the links of the walked rows into a tile of target rows by two scans.
 *  @c open_sweep_ zeroes @c incoming_offsets_, @c count_back_links_ adds to them,
 *  @c offset_back_links_ turns the counts into run starts and may shorten the tile,
 *  @c scatter_back_links_ takes a tape position per link, and @c merge_back_links_ reads each run
 *  and rewrites its row under the row's lock. The executor's fork and await part the phases, and
 *  @c advance runs on the caller alone.
 */
#include <weak_memory.pml>

/** The words: row 0's slot zero and its one tail slot, the executor's fork and await, the tile's
 *  offsets, and the tape. */
#define row_lock 0
#define row_tail 1
#define pool_forks 2
#define pool_arrivals 3
#define incoming_offsets(row) (4 + (row))
#define back_links(position) (6 + (position))

/** The tile: source row 0 links to both target rows, source row 1 to the first, and each thread
 *  runs the steps of the source row and the target row of its own index. */
#define tile_rows 2
#define link_count 3
#define links_into(row) ((row) == 0 -> 3 : 1)
#define links_count(row) ((row) == 0 -> 2 : 1)
#define run_length(row) (links_count(row) < tape_capacity -> links_count(row) : tape_capacity)
#define bit_of(source) (1 << (source))

/** The values: a tape slot names its source or nothing yet, and row 0 holds its old members until
 *  the merge rewrites them, each distinct, with the sentinel the all-ones ID in silicon. */
#define hole 0
#define link_of(source) ((source) + 1)
#define source_of(value) ((value) - 1)
#define sentinel 0
#define old_first 1
#define old_tail 2
#define merged_first 3
#define merged_tail 4

/** The count's forms: the dense @c add, the basic @c fetch_add, or a plain read then write. */
#define add 0
#define fetch_add 1
#define plain 2

/** The knobs: the header's choices, each overridden by a verify line that replays them. */
#ifndef offsets_add
#define offsets_add fetch_add
#endif
#if offsets_add != add && offsets_add != fetch_add && offsets_add != plain
#error "offsets_add is add, fetch_add or plain"
#endif
#ifndef await_order
#define await_order order_acq_rel
#endif
#ifndef clear_offsets
#define clear_offsets true
#endif
#ifndef rows_change_between_scans
#define rows_change_between_scans false
#endif
#ifndef tape_capacity
#define tape_capacity link_count
#endif
#ifndef peek_order
#define peek_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif
#ifndef whole_rows
#define whole_rows false
#endif

byte scattered[link_count]; // the links written to each tape slot, a ghost over plain writes
byte sweep_bound;           // `sweep_.bound` past the tile's first row, which the fork carries
byte tape_size;             // `back_links_.size()`, which the fork carries

/** The caller's fork of phase @p phase . */
inline fork_phase(t, phase) { store(t, pool_forks, order_release, phase) }

/** A thread starts phase @p phase once it acquires the fork. */
inline await_fork(t, phase, seen) {
    (newest_value(pool_forks) >= phase);
    do
    :: load(t, pool_forks, order_acquire, seen);
        if
        :: seen >= phase -> break
        :: else
        fi
    od
}

/** A thread's arrival at the await, once its steps of the phase ran. */
inline arrive(t, seen) { read_modify_write(t, pool_arrivals, await_order, seen, seen + 1) }

/** The caller waits out @p arrivals arrivals before its @c advance. */
inline await_join(t, arrivals, seen) {
    (newest_value(pool_arrivals) == arrivals);
    do
    :: load(t, pool_arrivals, await_order & order_acquire, seen);
        if
        :: seen == arrivals -> break
        :: else
        fi
    od
}

/** @c open_sweep_: the previous tile's run ends are gone before the fork. */
inline open_sweep(t) {
    if
    :: clear_offsets ->
        store(t, incoming_offsets(0), order_relaxed, 0);
        store(t, incoming_offsets(1), order_relaxed, 0)
    :: else
    fi
}

/** The count of one link into @p row , in the form @c offsets_add names. */
inline count_link(t, row, counted) {
    if
    :: offsets_add == add ->
        landed(t);
        add_no_return(t, incoming_offsets(row), order_relaxed, 1)
    :: offsets_add == fetch_add ->
        read_modify_write(t, incoming_offsets(row), order_relaxed, counted, counted + 1)
    :: offsets_add == plain ->
        load(t, incoming_offsets(row), order_relaxed, counted);
        store(t, incoming_offsets(row), order_relaxed, counted + 1)
    fi
}

/** @c count_back_links_ over source row @p t . */
inline count_back_links(t, counted) {
    count_link(t, 0, counted);
    if
    :: t == 0 -> count_link(t, 1, counted)
    :: else
    fi
}

/** @c offset_back_links_: reads the counts plain, the tile ends before the first row passing the
 *  tape, and a lone row past it keeps the tape's worth. */
inline offset_back_links(t, rows, offset, count) {
    do
    :: rows == tile_rows -> break
    :: else ->
        load(t, incoming_offsets(rows), order_relaxed, count);
        if
        :: offset + count > tape_capacity && rows > 0 -> break
        :: offset + count > tape_capacity && rows == 0 -> count = tape_capacity
        :: else
        fi;
        store(t, incoming_offsets(rows), order_relaxed, offset);
        offset = offset + count;
        rows = rows + 1
    od;
    sweep_bound = rows;
    tape_size = offset;
    assert(sweep_bound == (tape_capacity < link_count -> 1 : tile_rows))
}

/** The scatter of @p source 's link into @p target : a slot takes one link, and a position past
 *  the tape is left out. */
inline scatter_link(t, source, target, position, link) {
    if
    :: target < sweep_bound ->
        read_modify_write(t, incoming_offsets(target), order_relaxed, position, position + 1);
        if
        :: position < tape_size ->
            atomic {
                scattered[position] = scattered[position] + 1;
                assert(scattered[position] == 1)
            };
            link = link_of(source);
            store(t, back_links(position), order_relaxed, link)
        :: else
        fi
    :: else
    fi
}

/** @c scatter_back_links_ over source row @p source , and over a third row that gained a link since
 *  the count under @c rows_change_between_scans . */
inline scatter_back_links(t, source, position, link) {
    scatter_link(t, source, 0, position, link);
    if
    :: source == 0 -> scatter_link(t, source, 1, position, link)
    :: else
    fi;
    if
    :: rows_change_between_scans && source == 1 -> scatter_link(t, 2, 0, position, link)
    :: else
    fi
}

/** @c rewrite_row_: the lock, the tail, and the unlock that republishes the row. */
inline rewrite_row(t, displaced) {
    read_modify_write(t, row_lock, order_acq_rel, displaced, sentinel);
    store(t, row_tail, order_relaxed, merged_tail);
    store(t, row_lock, unlock_order, merged_first)
}

#define capped(position) ((position) < tape_size -> (position) : tape_size)

/** @c merge_back_links_ over target row @p row : its run holds its own links, each once, cut only
 *  by the tape. */
inline merge_back_links(t, row, first, bound, position, value, sources) {
    if
    :: row < sweep_bound ->
        if
        :: row > 0 ->
            load(t, incoming_offsets(0), order_relaxed, first);
            first = capped(first)
        :: else
        fi;
        load(t, incoming_offsets(row), order_relaxed, bound);
        bound = capped(bound);
        assert(bound - first == run_length(row));
        position = first;
        do
        :: position < bound ->
            load(t, back_links(position), order_relaxed, value);
            assert(value != hole);
            assert((links_into(row) & bit_of(source_of(value))) != 0);
            assert((sources & bit_of(source_of(value))) == 0);
            sources = sources | bit_of(source_of(value));
            position = position + 1
        :: else -> break
        od;
        if
        :: row == 0 -> rewrite_row(t, value)
        :: else
        fi
    :: else
    fi
}

/** The acquire of slot zero a walk makes, waiting out the lock holder. */
inline peek_first(t, first) {
    do
    :: load(t, row_lock, peek_order, first);
        if
        :: first != sentinel -> break
        :: else -> (newest_value(row_lock) != sentinel)
        fi
    od
}

/** Row 0 as a walk reads it: the peek, then the tail for the prefetch and again for the gather. */
inline walk_row(t, first, prefetched, gathered) {
    peek_first(t, first);
    load(t, row_tail, order_relaxed, prefetched);
    load(t, row_tail, order_relaxed, gathered);
    assert(prefetched == old_tail || prefetched == merged_tail);
    assert(gathered == merged_tail || prefetched == old_tail);
    assert(first != merged_first || gathered == merged_tail);
    assert(!whole_rows || first != old_first || gathered == old_tail)
}
