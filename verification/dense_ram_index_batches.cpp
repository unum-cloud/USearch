/**
 *  @file verification/dense_ram_index_batches.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for a batch's commit on the dense index in
 *      `include/usearch/dense/ram_index.hpp`.
 *
 *  Two commits dispense two IDs each with one add, each spawns a helper for its second row as the
 *  executor's fan-out would, joins it, and takes one watermark turn for the whole range. A row's walk
 *  reads row 0 below its snapshot, and past it the other commit's first row once the dispenser shows
 *  it dispensed and its slot zero shows it published; a leader that reached row 0 back-links into
 *  it, and a walk admits its sibling row only once slot zero shows it published. A searcher acquires
 *  the watermark, which never lies strictly inside a range, and reads every row's payload below it.
 *  The same scenario is `dense_ram_index_batches.pml`.
 *
 *  @verify pass sc,rc11,imm dense_ram_index_batches
 *  @verify fail sc,rc11,imm dense_ram_index_batches_range_turn_false: one turn per range keeps a
 *      batch whole; a turn per row lets the searcher see half a commit
 *  @verify fail sc,rc11,imm dense_ram_index_batches_own_range_check_false: a walk admits a row of its
 *      own range only once slot zero shows it published; unchecked, it reads the sibling's payload
 *      before its unlock
 */
#include <atomic> // `std::memory_order`, `std::atomic_thread_fence`

#include <genmc.hpp> // `spawn`, `join`, `verify` - the threads and the assertion GenMC intercepts

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the reference the index binds

#include "usearch/utilities/types.hpp" // `u32_t` - node IDs and row words

namespace fu = ashvardanian::forkunion;
namespace us = unum::usearch;
using us::u32_t;
template <typename value_type_>
using atomic_ref = fu::standard_atomic_ref<value_type_>;

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    bool range_turn = true;
    bool own_range_check = true;
};

constexpr knobs_t header_k {};
constexpr knobs_t range_turn_false_k {.range_turn = false};
constexpr knobs_t own_range_check_false_k {.own_range_check = false};

/** The all-ones ID a locked slot zero holds. */
constexpr u32_t sentinel_k = ~u32_t {0};

constexpr u32_t range_rows_k = 2;

/** The index's two counters: IDs dispensed so far, and rows visible below the watermark. */
u32_t next_dispensed_id = 0;
u32_t last_visible_id = 0;

/** Four rows: slot zero, which locks and publishes, and a tail; and each row's payload. */
u32_t batch_rows[4][2] = {{sentinel_k, 0}, {sentinel_k, 0}, {sentinel_k, 0}, {sentinel_k, 0}};
u32_t payloads[4] = {0, 0, 0, 0};

/** A walk's read of a row it admitted: the payload its wirer stored before the unlock. */
void read_wired(u32_t id) noexcept {
    verify(atomic_ref<u32_t>(payloads[id]).load(std::memory_order_relaxed) == 10 + id);
}

void take_turn(u32_t from, u32_t to) noexcept {
    while (atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire) != from) {}
    atomic_ref<u32_t>(last_visible_id).store(to, std::memory_order_release);
}

template <knobs_t const& knobs_>
void wire(u32_t id, u32_t sibling) noexcept {
    u32_t const first_invisible = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire);
    u32_t const first_undispensed = atomic_ref<u32_t>(next_dispensed_id).load(std::memory_order_acquire);
    if (first_invisible) read_wired(0);
    // Past the snapshot, the other commit's first row once dispensed and its slot zero shows it published.
    u32_t const other = id < range_rows_k ? range_rows_k : 0;
    bool const reached = !first_invisible && other < first_undispensed &&
                         atomic_ref<u32_t>(batch_rows[other][0]).load(std::memory_order_acquire) != sentinel_k;
    if (reached) read_wired(other);
    bool const linked = first_invisible || (reached && !other);
    if constexpr (knobs_.own_range_check) {
        if (atomic_ref<u32_t>(batch_rows[sibling][0]).load(std::memory_order_acquire) != sentinel_k)
            read_wired(sibling);
    }
    else read_wired(sibling);

    atomic_ref<u32_t>(payloads[id]).store(10 + id, std::memory_order_relaxed);
    atomic_ref<u32_t>(batch_rows[id][1]).store(10 + id, std::memory_order_relaxed);
    atomic_ref<u32_t>(batch_rows[id][0]).store(10 + id, std::memory_order_release);

    // The leader alone back-links: a contender's failed exchange and the holder's unlock are writes GenMC cannot order.
    if (linked && id % range_rows_k == 0) {
        std::atomic_thread_fence(std::memory_order_release);
        while (atomic_ref<u32_t>(batch_rows[0][0]).exchange(sentinel_k, std::memory_order_acq_rel) == sentinel_k) {}
        atomic_ref<u32_t>(batch_rows[0][1]).store(20 + id, std::memory_order_relaxed);
        atomic_ref<u32_t>(batch_rows[0][0]).store(20 + id, std::memory_order_release);
    }
}

template <knobs_t const& knobs_>
void* help(void* range) noexcept {
    u32_t const first = *static_cast<u32_t*>(range);
    wire<knobs_>(first + 1, first);
    if constexpr (!knobs_.range_turn) take_turn(first + 1, first + 2);
    return nullptr;
}

template <knobs_t const& knobs_>
void* commit(void*) noexcept {
    u32_t first = atomic_ref<u32_t>(next_dispensed_id).fetch_add(range_rows_k, std::memory_order_acq_rel);
    thread_t const helper = spawn(help<knobs_>, &first);
    wire<knobs_>(first, first + 1);
    if constexpr (!knobs_.range_turn) take_turn(first, first + 1);
    join(helper);
    if constexpr (knobs_.range_turn) take_turn(first, first + range_rows_k);
    return nullptr;
}

void* search(void*) noexcept {
    u32_t const watermark = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire);
    verify(watermark % range_rows_k == 0);
    for (u32_t id = 0; id != watermark; ++id) read_wired(id);
    return nullptr;
}

template <knobs_t const& knobs_>
int commit_and_search() noexcept {
    thread_t const first = spawn(commit<knobs_>);
    thread_t const second = spawn(commit<knobs_>);
    thread_t const searcher = spawn(search);
    join(first);
    join(second);
    join(searcher);
    return 0;
}

extern "C" int dense_ram_index_batches() { return commit_and_search<header_k>(); }
extern "C" int dense_ram_index_batches_range_turn_false() { return commit_and_search<range_turn_false_k>(); }
extern "C" int dense_ram_index_batches_own_range_check_false() { return commit_and_search<own_range_check_false_k>(); }
