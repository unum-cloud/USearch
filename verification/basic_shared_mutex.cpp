/**
 *  @file verification/basic_shared_mutex.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for `basic_shared_mutex` from `include/usearch/basic/ram_index.hpp`, the
 *      real header: one mutator takes the shared side and writes a pair of rows inside it, one
 *      prober brackets its reads with the wait-free snapshots.
 *
 *  A certified prober must not see the pair torn. The mutant spells `lock_shared`'s acquisition
 *  over the gate's own word, without the release fence the header ends it in. The same scenario
 *  with a second mutator and a unique taker is `basic_shared_mutex.pml`.
 *
 *  @verify pass sc,rc11,imm basic_shared_mutex
 *  @verify fail rc11,imm basic_shared_mutex_lock_shared_fence_false: `lock_shared` is an acquire
 *      read-modify-write that releases nothing before the row stores, so its closing release fence
 *      is what pairs with the probe's acquire fence; without it, the exit snapshot reads the
 *      untouched word while a row read returned a store made under the lock, in compiled code on
 *      Arm and POWER too, since the acquire orders the add's read and not its write
 *  @verify pass sc,tso basic_shared_mutex_lock_shared_fence_false: x86's locked add orders the row
 *      stores after its write on its own
 */
#include <atomic> // `std::atomic_ref`, `std::atomic_thread_fence`

#include <genmc.hpp> // `spawn`, `join`, `verify` - the threads and the assertion GenMC intercepts

#include "usearch/basic/ram_index.hpp" // `basic_shared_mutex` - the gate under test

namespace us = unum::usearch;
using us::u64_t;
using gate_t = us::basic_shared_mutex<us::dummy_waiter_t>;

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    bool lock_shared_fence = true;
};

constexpr knobs_t header_k {};
constexpr knobs_t lock_shared_fence_false_k {.lock_shared_fence = false};

gate_t gate;
int rows[2] = {0, 0};

template <knobs_t const& knobs_>
void* mutate(void*) noexcept {
    // The mutant is `lock_shared` without its closing fence; no unique holder runs here, so the add never backs off
    if constexpr (knobs_.lock_shared_fence) gate.lock_shared();
    else gate_t::atomic_ref<u64_t>(gate.word).fetch_add(gate_t::generation_unit_k + 1, std::memory_order_acquire);
    std::atomic_ref<int>(rows[0]).store(1, std::memory_order_relaxed);
    std::atomic_ref<int>(rows[1]).store(1, std::memory_order_relaxed);
    gate.unlock_shared();
    return nullptr;
}

void* probe(void*) noexcept {
    gate_t::snapshot_t const entry = gate.snapshot();
    if (!entry.unlocked()) return nullptr;
    int const first = std::atomic_ref<int>(rows[0]).load(std::memory_order_relaxed);
    int const second = std::atomic_ref<int>(rows[1]).load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    gate_t::snapshot_t const exit = gate.snapshot();
    if (exit.unlocked() && exit.generation == entry.generation) verify(first == second);
    return nullptr;
}

template <knobs_t const& knobs_>
int mutate_and_probe() noexcept {
    thread_t const mutator = spawn(mutate<knobs_>);
    thread_t const prober = spawn(probe);
    join(mutator);
    join(prober);
    return 0;
}

extern "C" int basic_shared_mutex() { return mutate_and_probe<header_k>(); }
extern "C" int basic_shared_mutex_lock_shared_fence_false() { return mutate_and_probe<lock_shared_fence_false_k>(); }
