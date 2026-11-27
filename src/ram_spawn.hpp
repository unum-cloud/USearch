/**
 *  @file src/ram_spawn.hpp
 *  @author Ash Vardanian
 *  @date August 30, 2026
 *  @brief The library's policy over @c ram_executor::spawn: default thread counts, caller
 *      exclusivity and placement for the RAM and NVMe executors.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <utility> // `std::move`

#include <forkunion/forkunion.hpp> // `fu::pool_t`, `fu::topology_t`, `fu::default_threads_count`

#include "usearch/utilities/nvme_device.hpp" // `nvme_executor`
#include "usearch/utilities/ram_device.hpp"  // `ram_executor`, `no_memory_domain_k`

namespace util = unum::usearch::utilities;
namespace fu = ashvardanian::forkunion;

/** The thread count a zero ask means: ForkUnion's @c fu::default_threads_count over the cores
 *  of @p topology, or the caller alone in a build without OS threads. */
inline util::expected<std::size_t> default_threads(fu_topology_t topology) noexcept {
    if (!(fu_comptime_capabilities() & fu_capability_os_threads_k)) return {1, util::status_code_t::success_k};
    std::size_t cores = 0;
    if (fu_logical_cores_count(topology, &cores) != fu_success_k) return {0, util::status_code_t::uninitialized_k};
    return {fu::default_threads_count(cores), util::status_code_t::success_k};
}

/** Spawns @p threads over @p topology with the caller inside the pool or outside it, as
 *  @p exclusivity says, using only the ForkUnion facilities @p allowed names, each thread's
 *  scratch holding at most @p scratch_limit bytes; zero threads means @ref default_threads. */
inline util::expected<util::ram_executor<fu::pool_t>> spawn(fu_topology_t topology, std::size_t threads,
                                                            fu::caller_exclusivity_t exclusivity,
                                                            fu_capabilities_t allowed,
                                                            std::size_t scratch_limit = 0) noexcept {
    if (!threads) {
        auto const [count, status] = default_threads(topology);
        if (util::failed(status)) return {{}, status};
        threads = count;
    }
    // A flat pool, unlike a distributed one, never pins the caller's own thread.
    if (exclusivity == fu::caller_inclusive_k)
        allowed = static_cast<fu_capabilities_t>(allowed & ~fu_capability_place_memory_on_domain_k);
    return util::ram_executor<fu::pool_t>::spawn(topology, threads, exclusivity, allowed, util::no_memory_domain_k,
                                                 scratch_limit);
}

/** Spawns @p threads over @p topology and takes the pool, each thread's scratch holding at most
 *  @p scratch_limit bytes; zero threads means @ref default_threads. One thread is the caller
 *  itself, and a build without OS threads refuses more. */
inline util::expected<util::ram_executor<fu::pool_t>> spawn(fu_topology_t topology, std::size_t threads,
                                                            fu_capabilities_t allowed = fu_capabilities_all_k,
                                                            std::size_t scratch_limit = 0) noexcept {
    if (!threads) {
        auto const [count, status] = default_threads(topology);
        if (util::failed(status)) return {{}, status};
        threads = count;
    }
    return spawn(topology, threads, threads != 1 ? fu::caller_exclusive_k : fu::caller_inclusive_k, allowed,
                 scratch_limit);
}

/**
 *  @brief Spawns @p threads over the compute domains local to @p memory_domain, the caller
 *      outside the pool, using only the ForkUnion facilities @p allowed names, each thread's
 *      scratch holding at most @p scratch_limit bytes; zero threads means @ref default_threads.
 *
 *  @par Statuses:
 *  - @c unsupported_k More than one thread in a build without OS threads, or placement masked
 *    off on a machine of several memory domains.
 *  - @c invalid_argument_k No compute domain is local to @p memory_domain.
 */
inline util::expected<util::ram_executor<fu::pool_t>> spawn_near(fu_topology_t topology,
                                                                 fu_memory_domain_id_t memory_domain,
                                                                 std::size_t threads, fu_capabilities_t allowed,
                                                                 std::size_t scratch_limit = 0) noexcept {
    if (!threads) {
        auto const [count, status] = default_threads(topology);
        if (util::failed(status)) return {{}, status};
        threads = count;
    }
    return util::ram_executor<fu::pool_t>::spawn(topology, threads, fu::caller_exclusive_k, allowed, memory_domain,
                                                 scratch_limit);
}

/**
 *  @brief Spawns @p threads over @p topology under the ForkUnion facilities @p allowed names,
 *      zero threads meaning @ref default_threads.
 *
 *  The threads run on the compute domains local to @p near_domain, or across the machine without
 *  one, each thread's scratch holding at most @p scratch_limit bytes, zero for
 *  @c scratch_limit_default_k.
 *
 *  Each queue opens with @p config on its own thread, every one on the transport thread zero's
 *  queue picked. A nonzero @p config queue depth pins every call's depth, while zero leaves
 *  each call to derive its own from the media it reads, the queues @p opening_depth reads
 *  deep, at least one, until a call needs more.
 *
 *  @par Statuses:
 *  - @c uninitialized_k SPDK is named without a driver.
 *  - @c unsupported_k The build has no queue for @p config's transport, or none set up.
 *  - @c io_k The named queue's setup was refused.
 */
template <typename queue_type_>
util::expected<util::nvme_executor<queue_type_, fu::pool_t>> spawn(
    fu_topology_t topology, std::size_t threads, typename queue_type_::config_t const& config,
    fu_capabilities_t allowed = fu_capabilities_all_k, fu_memory_domain_id_t near_domain = util::no_memory_domain_k,
    std::size_t scratch_limit = 0, std::size_t opening_depth = 0) noexcept {
    using executor_t = util::nvme_executor<queue_type_, fu::pool_t>;
    // The caller never runs a slice, so each thread index stays one OS thread, the single
    // issuer a ring wants; a build without OS threads runs its one thread on the caller.
    auto [threads_executor, spawn_status] = near_domain != util::no_memory_domain_k
                                                ? spawn_near(topology, near_domain, threads, allowed, scratch_limit)
                                                : spawn(topology, threads, fu::caller_exclusive_k, allowed,
                                                        scratch_limit);
    if (util::failed(spawn_status)) return {executor_t {}, spawn_status};
    return executor_t::acquire(std::move(threads_executor), config, opening_depth);
}
