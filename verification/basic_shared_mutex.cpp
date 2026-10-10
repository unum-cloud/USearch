/**
 *  @file verification/basic_shared_mutex.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for `basic_shared_mutex` from `include/usearch/basic/ram_index.hpp`, the
 *      real header: one mutator takes the shared side and writes a pair of rows inside it, one
 *      prober brackets its reads with the wait-free snapshots.
 *
 *  A certified prober must not see the pair torn, and RC11 would let it: `lock_shared` is an
 *  acquire read-modify-write, and without the release fence it ends in the exit snapshot may
 *  read the untouched word while a row read returned a store made under the lock. The fence
 *  is in the header; `basic_shared_mutex.pml` keeps the variant without it.
 */
#include <atomic> // `std::atomic_ref`, `std::atomic_thread_fence`

#include "usearch/basic/ram_index.hpp" // `basic_shared_mutex` - the gate under test

#include "genmc.hpp"

namespace us = unum::usearch;
using gate_t = us::basic_shared_mutex<us::dummy_waiter_t>;

gate_t gate;
int rows[2] = {0, 0};

void* mutate(void*) noexcept {
    gate.lock_shared();
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

int main() {
    thread_t const mutator = spawn(mutate);
    thread_t const prober = spawn(probe);
    join(mutator);
    join(prober);
    return 0;
}
