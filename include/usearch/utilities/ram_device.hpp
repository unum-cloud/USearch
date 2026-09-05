/**
 *  @brief Host-side vocabulary for driving one CPU machine - domain placement & executors.
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/ram_device.hpp
 *  @date August 30, 2026
 *
 *  Everything a RAM-resident index needs from the machine and nothing it stores: the
 *  ForkUnion-backed allocator binding storage to one memory domain, and the executor holding
 *  the threads with one byte arena apiece. A thread's scratch is a run of bytes, so one
 *  executor drives indexes of any quantization, metric or capability tier.
 */
#pragma once

#include <cstdlib> // `std::aligned_alloc`, `std::free` - the heap fallback of a domainless allocator

#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <type_traits> // `std::remove_reference_t` - the trampolines' callback recovery
#include <utility>     // `std::swap`, `std::move`, `std::exchange`

#include <forkunion.h>                // `fu_allocate_on_domain_id`, `fu_pool_spawn`, `fu_local_memory_of`
#include <forkunion/capabilities.hpp> // `fu::preferred_yield_t` - the hold's spin policy
#include <forkunion/types.hpp>        // `fu::spin_mutex`

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the arenas' storage
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`

namespace unum {
namespace usearch {

#pragma region Placement

/** @brief The domain ID of a machine without NUMA; ForkUnion asserts on it, so the heap serves. */
inline constexpr fu_memory_domain_id_t no_memory_domain_k = -1;

/** @brief Allocates on one ForkUnion memory domain, so storage lands beside the threads that walk
 *      it; @c no_memory_domain_k means the plain heap. */
template <typename value_type_ = byte_t>
struct forkunion_allocator {
    using value_type = value_type_;

    /** @brief The OS memory-domain ID storage is claimed on. */
    fu_memory_domain_id_t memory_domain = no_memory_domain_k;

    constexpr forkunion_allocator() noexcept = default;
    explicit constexpr forkunion_allocator(fu_memory_domain_id_t memory_domain) noexcept
        : memory_domain(memory_domain) {}

    /** @brief The rebinding conversion @c std::allocator_traits performs; the domain travels. */
    template <typename other_type_>
    constexpr forkunion_allocator(forkunion_allocator<other_type_> const& other) noexcept
        : memory_domain(other.memory_domain) {}

    /** @brief Every run starts on @c default_alignment_k: arenas slice on that boundary and size
     *      themselves assuming an aligned base, so the heap fallback aligns like the domains do. */
    value_type_* allocate(std::size_t count) noexcept {
        std::size_t const bytes = aligned_bytes_(count);
        if (memory_domain == no_memory_domain_k)
            return static_cast<value_type_*>(std::aligned_alloc(default_alignment_k, bytes));
        void* memory = nullptr;
        if (fu_allocate_on_domain_id(memory_domain, bytes, &memory) != fu_success_k) return nullptr;
        return static_cast<value_type_*>(memory);
    }

    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (memory_domain == no_memory_domain_k) return std::free(pointer);
        fu_free_on_domain_id(memory_domain, pointer, aligned_bytes_(count));
    }

    bool operator==(forkunion_allocator const& other) const noexcept { return memory_domain == other.memory_domain; }

  private:
    /** @brief The byte count a run of @p count elements is claimed and released with. */
    static std::size_t aligned_bytes_(std::size_t count) noexcept {
        return divide_round_up<std::size_t>(count * sizeof(value_type_), default_alignment_k) * default_alignment_k;
    }
};

/** @brief The byte specialization every container and index places its storage through. */
using forkunion_allocator_t = forkunion_allocator<byte_t>;

static_assert(buffer_allocator<forkunion_allocator_t>);

#pragma endregion Placement

#pragma region Executor

/** @brief The threads a bulk verb advances on and the scratch they walk with - one arena per
 *      thread, so a fan-out never leases or waits. Admits one caller at a time. */
struct forkunion_executor_t {
    /** @brief One thread's scratch on its own memory domain: chunks that double and never move,
     *      rewound between calls. */
    using arena_t = growing_arena<forkunion_allocator_t>;
    /** @brief One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex<fu::preferred_yield_t>>;

    /** @brief The label the pool's threads carry in the OS. */
    inline static constexpr char const pool_name_k[] = "usearch";
    /** @brief One core in this many stays idle on a zero ask, for the ordered publication path. */
    inline static constexpr std::size_t idle_cores_denominator_k = 8;

    forkunion_executor_t() = default;
    forkunion_executor_t(forkunion_executor_t&& other) noexcept { swap(other); }
    forkunion_executor_t& operator=(forkunion_executor_t&& other) noexcept { return swap(other), *this; }
    forkunion_executor_t(forkunion_executor_t const&) = delete;
    forkunion_executor_t& operator=(forkunion_executor_t const&) = delete;
    /** @brief Deletes the pool and every arena, unless @ref release reclaimed the pool first. */
    ~forkunion_executor_t() noexcept {
        if (pool_) fu_pool_delete(pool_);
    }

    void swap(forkunion_executor_t& other) noexcept {
        std::swap(arenas_per_thread_, other.arenas_per_thread_);
        std::swap(pool_, other.pool_);
        std::swap(topology_, other.topology_);
    }

    /** @brief Takes over @p pool, owned and deleted here from now on - even on failure. Colocates
     *      one arena with each thread's memory domain. */
    static expected<forkunion_executor_t> acquire(fu_topology_t topology, fu_pool_t pool) noexcept {
        forkunion_executor_t executor;
        executor.pool_ = pool;
        executor.topology_ = topology;
        std::size_t threads = 0;
        if (fu_pool_threads_count(pool, &threads) != fu_success_k)
            return {std::move(executor), status_t::uninitialized_k};
        // Each thread names its own memory domain; the arenas are then placed on those in order.
        growing_buffer<fu_memory_domain_id_t> domains {global_allocator};
        if (failed(domains.resize(threads))) return {std::move(executor), status_t::bad_alloc_k};
        for (fu_memory_domain_id_t& domain : domains) domain = no_memory_domain_k;
        status_t const status = executor.for_threads([&](std::size_t thread, std::size_t compute_domain) noexcept {
            std::size_t memory_index = 0;
            fu_memory_domain_id_t memory_domain = no_memory_domain_k;
            if (fu_local_memory_of(topology, compute_domain, &memory_index) != fu_success_k ||
                fu_memory_domain_id_at_index(topology, memory_index, &memory_domain) != fu_success_k)
                return;
            domains[thread] = memory_domain;
        });
        if (failed(status)) return {std::move(executor), status};
        if (failed(executor.arenas_per_thread_.reserve_exactly(threads)))
            return {std::move(executor), status_t::bad_alloc_k};
        for (fu_memory_domain_id_t const domain : domains)
            executor.arenas_per_thread_.push_back_reserved(arena_t {forkunion_allocator_t(domain)});
        return {std::move(executor), status_t::success_k};
    }

    /** @brief Relinquishes the pool without deleting it - the inverse of @ref acquire, so the
     *      same pool can back a different executor next; the arenas drop and this empties. */
    fu_pool_t release() noexcept {
        arenas_per_thread_.clear();
        topology_ = nullptr;
        return std::exchange(pool_, nullptr);
    }

    /** @brief The thread count a zero ask means: every core but one in @ref idle_cores_denominator_k. */
    static expected<std::size_t> default_threads(fu_topology_t topology) noexcept {
        std::size_t cores = 0;
        if (fu_logical_cores_count(topology, &cores) != fu_success_k) return {0, status_t::uninitialized_k};
        return {cores - cores / idle_cores_denominator_k, status_t::success_k};
    }

    /** @brief Spawns @p threads over @p topology and takes the pool; zero means @ref default_threads. */
    static expected<forkunion_executor_t> spawn(fu_topology_t topology, std::size_t threads) noexcept {
        if (!threads) {
            expected<std::size_t> const defaulted = default_threads(topology);
            if (failed(defaulted.status)) return {{}, defaulted.status};
            threads = defaulted.value;
        }
        fu_pool_t pool = nullptr;
        if (fu_pool_new(pool_name_k, fu_capabilities_all_k, &pool) != fu_success_k) return {{}, status_t::bad_alloc_k};
        if (fu_pool_spawn(topology, pool, threads, fu_caller_exclusive_k) != fu_success_k)
            return fu_pool_delete(pool), expected<forkunion_executor_t> {{}, status_t::uninitialized_k};
        return acquire(topology, pool);
    }

    /** @brief Claims the executor for one call; a second concurrent caller gets @c contended_k. */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /** @brief Rewinds every thread's arena and seeds it at @p bytes_per_thread, the first chunk's
     *      width; the widest chunk an earlier call opened stays, so the next call starts at its
     *      high-water mark. */
    status_t reserve(std::size_t bytes_per_thread) noexcept {
        if (arenas_per_thread_.empty()) return status_t::uninitialized_k;
        for (arena_t& arena : arenas_per_thread_) {
            arena.clear();
            if (failed(arena.reserve_exactly(bytes_per_thread))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** @brief Thread @p thread 's own arena; no two threads share one. */
    arena_t& arena(std::size_t thread) noexcept { return arenas_per_thread_[thread]; }

    /** @brief The memory domain thread @p thread 's arena sits on - the thread's home, which a
     *      fan-out serves first. */
    fu_memory_domain_id_t memory_domain(std::size_t thread) const noexcept {
        return arenas_per_thread_[thread].upstream().memory_domain;
    }

    /** @brief Threads this executor drives. */
    std::size_t size() const noexcept { return arenas_per_thread_.size(); }
    /** @brief The topology the pool was spawned over; borrowed, never owned. */
    fu_topology_t topology() const noexcept { return topology_; }
    /** @brief Nothing is ever pending: every fan-out blocks inside, so a drain is a no-op. */
    status_t drain() noexcept { return status_t::success_k; }

    /** @brief Runs @p callback on every thread with the thread index and its compute domain; must
     *      be @c noexcept. */
    template <typename callback_type_>
    status_t for_threads(callback_type_&& callback) noexcept {
        if (!pool_) return status_t::uninitialized_k;
        fu_for_threads_t const trampoline = [](fu_lambda_context_t context, std::size_t thread,
                                               std::size_t compute_domain) {
            (*static_cast<std::remove_reference_t<callback_type_>*>(context))(thread, compute_domain);
        };
        if (fu_pool_for_threads(pool_, trampoline, &callback) != fu_success_k) return status_t::uninitialized_k;
        return status_t::success_k;
    }

    /** @brief Runs @p callback over @p tasks items, stolen dynamically, with the task index, the
     *      thread index and its compute domain; must be @c noexcept. */
    template <typename callback_type_>
    status_t for_tasks(std::size_t tasks, callback_type_&& callback) noexcept {
        if (!pool_) return status_t::uninitialized_k;
        if (!tasks) return status_t::success_k;
        fu_for_task_t const trampoline = [](fu_lambda_context_t context, std::size_t task, std::size_t thread,
                                            std::size_t compute_domain) {
            (*static_cast<std::remove_reference_t<callback_type_>*>(context))(task, thread, compute_domain);
        };
        if (fu_pool_for_n_dynamic(pool_, tasks, trampoline, &callback) != fu_success_k)
            return status_t::uninitialized_k;
        return status_t::success_k;
    }

  private:
    /** @brief One arena per thread, colocated with it once at @ref acquire; contexts point into it,
     *      so its chunks never move. */
    growing_buffer<arena_t> arenas_per_thread_ {global_allocator};
    /** @brief Held by the one caller driving the threads; never swapped. */
    fu::spin_mutex<fu::preferred_yield_t> held_;
    /** @brief The pool every fan-out dispatches on, owned outright and deleted here. */
    fu_pool_t pool_ = nullptr;
    /** @brief The topology the pool was spawned over; borrowed, never owned. */
    fu_topology_t topology_ = nullptr;
};

#pragma endregion Executor

} // namespace usearch
} // namespace unum
