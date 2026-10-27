/**
 *  @file verification/dense_ram_index_watermark/client.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for the dense index's publication protocol in
 *      `include/usearch/dense/ram_index.hpp`.
 *
 *  The dispenser, the rows, the back-link's release fence, the watermark turn, and the save under
 *  @c concurrent_k with its settlement probe, spelled over the ForkUnion reference the index binds.
 *  Two inserters dispense IDs 0 and 1, publish their rows, and the second back-links into row 0.
 *  A saver plans, copies each row under its lock, and re-checks the counters after an acquire
 *  fence. The same scenario is @c inserters.pml.
 *
 *  - A settled copy holds whole rows and no locked one.
 *
 *  @verify pass sc,rc11,imm inserters
 *  @verify pass sc,rc11,imm inserters_back_link_fence_false
 */
#include <atomic> // `std::memory_order`, `std::atomic_thread_fence`

#include <genmc.hpp> // `spawn`, `join`, `verify` - the threads and the assertion GenMC intercepts

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the reference the index binds

#include "usearch/utilities/types.hpp" // `u32_t` - node IDs and row words

namespace fu = ashvardanian::forkunion;
namespace us = unum::usearch;
using us::utilities::u32_t;
template <typename value_type_>
using atomic_ref = fu::standard_atomic_ref<value_type_>;

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    bool back_link_fence = true;
};

constexpr knobs_t header_k {};
constexpr knobs_t back_link_fence_false_k {.back_link_fence = false};

/** The all-ones ID a locked slot zero holds. */
constexpr u32_t sentinel_k = ~u32_t {0};

/** The index's two counters: IDs dispensed so far, and rows visible below the watermark. */
u32_t next_dispensed_id = 0;
u32_t last_visible_id = 0;

/** Two rows of two words: slot zero locks and publishes, slot one is the tail. */
u32_t rows[2][2] = {{sentinel_k, 0}, {sentinel_k, 0}};

/** `neighbors_ref::lock`: the peek waits out a holder, then the claim. A contender's failed exchange
 *  and the holder's unlock are writes GenMC cannot order, so the claim is a strong compare-exchange
 *  from the value the peek read, which writes nothing when it fails. */
u32_t lock_row(u32_t& slot) noexcept {
    u32_t displaced;
    do {
        while ((displaced = atomic_ref<u32_t>(slot).load(std::memory_order_acquire)) == sentinel_k) {}
    } while (!atomic_ref<u32_t>(slot).compare_exchange_strong(displaced, sentinel_k, std::memory_order_acq_rel,
                                                              std::memory_order_relaxed));
    return displaced;
}

template <knobs_t const& knobs_>
void* insert(void*) noexcept {
    u32_t const id = atomic_ref<u32_t>(next_dispensed_id).fetch_add(1, std::memory_order_acq_rel);
    u32_t const first_invisible = atomic_ref<u32_t>(last_visible_id).load(std::memory_order_acquire);

    atomic_ref<u32_t>(rows[id][1]).store(10 + id, std::memory_order_relaxed);
    atomic_ref<u32_t>(rows[id][0]).store(10 + id, std::memory_order_release);

    if (first_invisible) {
        if constexpr (knobs_.back_link_fence) std::atomic_thread_fence(std::memory_order_release);
        lock_row(rows[0][0]);
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
        copies[id][0] = lock_row(rows[id][0]);
        copies[id][1] = atomic_ref<u32_t>(rows[id][1]).load(std::memory_order_relaxed);
        atomic_ref<u32_t>(rows[id][0]).store(copies[id][0], std::memory_order_release);
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

template <knobs_t const& knobs_>
int insert_and_save() noexcept {
    thread_t const first = spawn(insert<knobs_>);
    thread_t const second = spawn(insert<knobs_>);
    thread_t const saver = spawn(save);
    join(first);
    join(second);
    join(saver);
    return 0;
}

extern "C" int inserters() { return insert_and_save<header_k>(); }
extern "C" int inserters_back_link_fence_false() { return insert_and_save<back_link_fence_false_k>(); }
