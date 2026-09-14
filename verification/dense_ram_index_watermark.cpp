/**
 *  @file verification/dense_ram_index_watermark.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for the dense index's publication protocol in
 *      `include/usearch/dense/ram_index.hpp`.
 *
 *  The dispenser, the rows, the back-link's release fence, the watermark turn, and the save under
 *  `concurrent_k` with its settlement probe, spelled line for line over the ForkUnion reference the
 *  index binds; the header's chunk fills are beyond GenMC's graph, see
 *  `dense_ram_index_capacity.cpp`.
 *
 *  Two inserters dispense IDs 0 and 1, publish their rows, and the second back-links into row
 *  0. A saver plans, copies the rows relaxed, and re-checks the counters after an acquire fence.
 *  A settled copy holds whole rows and no locked one. `-Dwithout_release_fence` drops the fence
 *  heading `back_link_neighbors`, and a settled copy pairs a new tail with an old slot zero.
 *  The same scenario is `dense_ram_index_watermark.pml`.
 */
#include <atomic> // `std::memory_order`, `std::atomic_thread_fence`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the reference the index binds

#include "usearch/utilities/types.hpp" // `u32_t` - node ids and row words

#include "genmc.hpp"

namespace fu = ashvardanian::forkunion;
namespace us = unum::usearch;
using us::u32_t;
template <typename value_type_>
using atomic_ref = fu::standard_atomic_ref<value_type_>;

/** @brief The all-ones id a locked slot zero holds. */
constexpr u32_t sentinel_k = ~u32_t {0};
/** @brief The index's two counters: ids dispensed so far, and rows visible below the watermark. */
u32_t next_dispensed_id = 0;
u32_t last_visible_id = 0;
/** @brief Two rows of two words: slot zero locks and publishes, slot one is the tail. */
u32_t rows[2][2] = {{sentinel_k, 0}, {sentinel_k, 0}};

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
