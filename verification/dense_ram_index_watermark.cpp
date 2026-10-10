/**
 *  @file verification/dense_ram_index_watermark.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for the dense index's publication protocol in
 *      `include/usearch/dense/ram_index.hpp`.
 *
 *  The dispenser, the rows, the back-link's release fence, the watermark turn, and the save under
 *  `concurrent_k` with its settlement probe, spelled over the ForkUnion reference the index binds,
 *  without the claim counter the header's save also reads, since nothing grows here; the header's
 *  chunk fills are beyond GenMC's graph, see `dense_ram_index_capacity.cpp`.
 *
 *  Two inserters dispense IDs 0 and 1, publish their rows, and the second back-links into row
 *  0. A saver plans, copies the rows relaxed, and re-checks the counters after an acquire fence.
 *  A settled copy holds whole rows and no locked one. `-Dwithout_release_fence` drops the fence
 *  heading `back_link_neighbors`, and a settled copy pairs a new tail with an old slot zero.
 *  The same scenario is `dense_ram_index_watermark.pml`.
 *
 *  `-Dscenario=batched` is that model's batched commit: two commits dispense two IDs each with
 *  one add, each spawns a helper for its second row as the executor's fan-out would, joins it,
 *  and takes one watermark turn for the whole range. A row's walk reads row 0 below its snapshot,
 *  and past it the other commit's first row once the dispenser shows it dispensed and its slot
 *  zero shows it published; a leader that reached row 0 back-links into it, and a walk admits its
 *  sibling row only once slot zero shows it published. A searcher acquires the watermark, which
 *  never lies strictly inside a range, and reads every row's payload below it.
 *  `-Dwithout_range_turn` takes a turn per row and the searcher sees half a commit;
 *  `-Dwithout_own_range_check` reads the sibling's payload before its unlock.
 */
#include <atomic> // `std::memory_order`, `std::atomic_thread_fence`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the reference the index binds

#include "usearch/utilities/types.hpp" // `u32_t` - node IDs and row words

#include "genmc.hpp"

// The scenarios `-Dscenario=` names: two inserters beside a saver, or two batched commits.
#define inserters 1
#define batched   2
#ifndef scenario
#define scenario inserters
#endif

namespace fu = ashvardanian::forkunion;
namespace us = unum::usearch;
using us::u32_t;
template <typename value_type_>
using atomic_ref = fu::standard_atomic_ref<value_type_>;

/** The all-ones ID a locked slot zero holds. */
constexpr u32_t sentinel_k = ~u32_t {0};

/** The index's two counters: IDs dispensed so far, and rows visible below the watermark. */
u32_t next_dispensed_id = 0;
u32_t last_visible_id = 0;

/** Two rows of two words: slot zero locks and publishes, slot one is the tail. */
u32_t rows[2][2] = {{sentinel_k, 0}, {sentinel_k, 0}};

#if scenario == batched

constexpr u32_t range_rows_k = 2;

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
#ifdef without_own_range_check
    read_wired(sibling);
#else
    if (atomic_ref<u32_t>(batch_rows[sibling][0]).load(std::memory_order_acquire) != sentinel_k) read_wired(sibling);
#endif

    atomic_ref<u32_t>(payloads[id]).store(10 + id, std::memory_order_relaxed);
    atomic_ref<u32_t>(batch_rows[id][1]).store(10 + id, std::memory_order_relaxed);
    atomic_ref<u32_t>(batch_rows[id][0]).store(10 + id, std::memory_order_release);

    // The leader alone back-links: a contender's failed exchange and the holder's unlock are writes GenMC cannot order.
    if (linked && id % range_rows_k == 0) {
#ifndef without_release_fence
        std::atomic_thread_fence(std::memory_order_release);
#endif
        while (atomic_ref<u32_t>(batch_rows[0][0]).exchange(sentinel_k, std::memory_order_acq_rel) == sentinel_k) {}
        atomic_ref<u32_t>(batch_rows[0][1]).store(20 + id, std::memory_order_relaxed);
        atomic_ref<u32_t>(batch_rows[0][0]).store(20 + id, std::memory_order_release);
    }
}

void* help(void* range) noexcept {
    u32_t const first = *static_cast<u32_t*>(range);
    wire(first + 1, first);
#ifdef without_range_turn
    take_turn(first + 1, first + 2);
#endif
    return nullptr;
}

void* commit(void*) noexcept {
    u32_t first = atomic_ref<u32_t>(next_dispensed_id).fetch_add(range_rows_k, std::memory_order_acq_rel);
    thread_t const helper = spawn(help, &first);
    wire(first, first + 1);
#ifdef without_range_turn
    take_turn(first, first + 1);
#endif
    join(helper);
#ifndef without_range_turn
    take_turn(first, first + range_rows_k);
#endif
    return nullptr;
}

void* search(void*) noexcept {
    u32_t const watermark = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire);
    verify(watermark % range_rows_k == 0);
    for (u32_t id = 0; id != watermark; ++id) read_wired(id);
    return nullptr;
}

int main() {
    thread_t const first = spawn(commit);
    thread_t const second = spawn(commit);
    thread_t const searcher = spawn(search);
    join(first);
    join(second);
    join(searcher);
    return 0;
}

#else

void* insert(void*) noexcept {
    u32_t const id = atomic_ref<u32_t>(next_dispensed_id).fetch_add(1, std::memory_order_acq_rel);
    u32_t const first_invisible = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire);

    atomic_ref<u32_t>(rows[id][1]).store(10 + id, std::memory_order_relaxed);
    atomic_ref<u32_t>(rows[id][0]).store(10 + id, std::memory_order_release);

    if (first_invisible) {
#ifndef without_release_fence
        std::atomic_thread_fence(std::memory_order_release);
#endif
        while (atomic_ref<u32_t>(rows[0][0]).exchange(sentinel_k, std::memory_order_acq_rel) == sentinel_k) {}
        atomic_ref<u32_t>(rows[0][1]).store(20 + id, std::memory_order_relaxed);
        atomic_ref<u32_t>(rows[0][0]).store(20 + id, std::memory_order_release);
    }

    while (atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire) != id) {}
    atomic_ref<u32_t>(last_visible_id).store(id + 1, std::memory_order_release);
    return nullptr;
}

void* save(void*) noexcept {
    u32_t const visible = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire);
    u32_t const dispensed = atomic_ref<u32_t>(next_dispensed_id).load(std::memory_order_acquire);
    if (visible != dispensed) return nullptr;
    u32_t copies[2][2];
    for (u32_t id = 0; id != visible; ++id) {
        copies[id][0] = atomic_ref<u32_t>(rows[id][0]).load(std::memory_order_relaxed);
        copies[id][1] = atomic_ref<u32_t>(rows[id][1]).load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    bool const settled = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_relaxed) == visible &&
                         atomic_ref<u32_t>(next_dispensed_id).load(std::memory_order_relaxed) == dispensed;
    if (!settled) return nullptr;
    for (u32_t id = 0; id != visible; ++id) {
        verify(copies[id][0] != sentinel_k);
        verify(copies[id][0] == copies[id][1]);
    }
    return nullptr;
}

int main() {
    thread_t const first = spawn(insert);
    thread_t const second = spawn(insert);
    thread_t const saver = spawn(save);
    join(first);
    join(second);
    join(saver);
    return 0;
}

#endif
