/**
 *  @brief Vector Search index baseline for arbitrary values, metrics & concurrency models.
 *  @author Ash Vardanian
 *  @file include/usearch/basic_index.hpp
 *  @date July 17, 2026
 */
#pragma once

#include <atomic> // `std::atomic`, `std::atomic_ref`
#include <memory> // `std::allocator`
#include <thread> // `std::this_thread::yield`

#include "usearch/types.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief Production concurrency backend: real hardware atomics & real thread yields.
 *
 *  The concurrency policy is the seam that lets one index template run on real threads in
 *  production, or on virtual threads with software-emulated weak memory ordering inside a
 *  single-threaded model-checking harness - fuzzing schedules, detecting livelocks, and
 *  validating acquire/release placement even on strongly-ordered hardware like x86.
 */
struct concurrency_model_t {
    template <typename value_type_>
    using atomic = std::atomic<value_type_>;
    template <typename value_type_>
    using atomic_ref = std::atomic_ref<value_type_>;
    static void yield() noexcept { std::this_thread::yield(); }
};

/**
 *  @brief Vector Search index baseline for arbitrary values & similarity measures.
 *
 *  Unlike the specialized @c dense_* family - hardwired to inner-product-class metrics,
 *  NumKong kernels, and hardware atomics - this baseline stays deliberately thin and generic,
 *  offloading every specialization decision to a policy:
 *
 *  @tparam value_type_ is any copyable entry payload; no dense-vector assumptions.
 *  @tparam metric_type_ is any callable measuring similarity between two values.
 *  @tparam neighbor_selector_type_ is the heuristic picking the final neighbors from the
 *      candidate pool of a traversal - diversity-pruning, distance-only, or custom.
 *  @tparam concurrency_type_ supplies the atomic types & yield primitive, swapping real
 *      hardware synchronization for a deterministic single-threaded simulation in tests.
 *  @tparam allocator_type_ owns all memory placement, including NUMA-aware arenas.
 *
 *  All thread-local scratch state lives outside the index object and is passed in by the
 *  caller, letting applications co-locate it with their own allocators and NUMA arenas.
 */
template <                                               //
    typename value_type_,                                //
    typename metric_type_,                               //
    typename neighbor_selector_type_,                    //
    typename concurrency_type_ = concurrency_model_t,    //
    typename allocator_type_ = std::allocator<std::byte> //
    >
struct basic_volatile_index;

} // namespace usearch
} // namespace unum
