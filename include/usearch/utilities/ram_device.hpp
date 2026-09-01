/**
 *  @brief Host-side vocabulary for driving one CPU machine - domain placement & executors.
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/ram_device.hpp
 *  @date August 30, 2026
 *
 *  Everything a RAM-resident index needs from the machine and nothing it stores: the
 *  ForkUnion-backed allocator binding storage to one memory domain, and the executor holding
 *  the worker threads with one byte arena apiece. The executor knows nothing of what walks -
 *  a thread's scratch is a run of bytes, so one executor drives indexes of any quantization,
 *  metric or capability tier, where a typed context could only ever serve the one
 *  instantiation it was declared for.
 */
#pragma once

#include <cstdlib> // `std::malloc`, `std::free` - the heap fallback of a domainless allocator

#include <atomic>      // `std::atomic_flag` - the executor's one-caller gate
#include <memory>      // `std::allocator_traits` - the rebinding conversions
#include <type_traits> // `std::true_type` - the propagation traits
#include <utility>     // `std::swap`, `std::move`, `std::exchange`

#include <forkunion.h> // `fu_allocate_on_domain_id`, `fu_pool_spawn`, `fu_local_memory_of`

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the arenas' storage
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`

namespace unum {
namespace usearch {

#pragma region Placement

/**
 *  @brief Allocates on one ForkUnion memory domain, so storage lands where the threads that walk
 *      it run. A negative domain names no node and falls back to the plain heap, which is what a
 *      machine without NUMA reports for every domain.
 */
template <typename value_type_ = std::byte>
struct forkunion_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    /** @brief The OS memory-domain id storage is claimed on; @c -1 is the heap. */
    fu_memory_domain_id_t memory_domain = -1;

    constexpr forkunion_allocator() noexcept = default;
    explicit constexpr forkunion_allocator(fu_memory_domain_id_t memory_domain) noexcept
        : memory_domain(memory_domain) {}

    /** @brief The rebinding conversion @c std::allocator_traits performs; the domain travels. */
    template <typename other_type_>
    constexpr forkunion_allocator(forkunion_allocator<other_type_> const& other) noexcept
        : memory_domain(other.memory_domain) {}

    value_type_* allocate(std::size_t count) noexcept {
        std::size_t const bytes = count * sizeof(value_type_);
        if (memory_domain < 0) return static_cast<value_type_*>(std::malloc(bytes));
        void* memory = nullptr;
        if (fu_allocate_on_domain_id(memory_domain, bytes, &memory) != fu_success_k) return nullptr;
        return static_cast<value_type_*>(memory);
    }

    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (memory_domain < 0) return std::free(pointer);
        fu_free_on_domain_id(memory_domain, pointer, count * sizeof(value_type_));
    }

    bool operator==(forkunion_allocator const& other) const noexcept { return memory_domain == other.memory_domain; }
};

static_assert(buffer_allocator<forkunion_allocator<std::byte>>);

#pragma endregion Placement

#pragma region Executor

/**
 *  @brief The worker threads a bulk verb advances on and the per-thread scratch they walk with -
 *      everything a call needs that an index does not own.
 *
 *  One thread holds one run, so the thread index @b is the scratch index and a fan-out never
 *  leases, never locks, and never waits for scratch. Each run is claimed on the memory domain
 *  local to the thread that carves it. The executor admits one caller at a time - @ref claim
 *  reports @c contended_k to a second - while the indexes it drives stay freely concurrent.
 */
struct forkunion_executor_t {
    /** @brief One thread's run of scratch, claimed on the memory domain that thread runs closest
     *      to; its @c size is what has been carved and its @c capacity is the run. */
    using thread_arena_t = growing_buffer<std::byte, forkunion_allocator<std::byte>>;

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

    /** @brief Takes over @p pool, which this executor owns and deletes from here on - even on
     *      failure - so the caller must not delete or re-spawn it. Colocates one arena with each
     *      worker's own memory domain, so an executor is fully usable the moment it exists and no
     *      fan-out ever waits on a first reservation. */
    static expected<forkunion_executor_t> acquire(fu_topology_t topology, fu_pool_t pool) noexcept {
        forkunion_executor_t executor;
        executor.pool_ = pool;
        executor.topology_ = topology;
        std::size_t threads = 0;
        if (fu_pool_threads_count(pool, &threads) != fu_success_k)
            return {std::move(executor), status_t::uninitialized_k};
        if (failed(executor.arenas_per_thread_.resize(threads))) return {std::move(executor), status_t::bad_alloc_k};
        status_t const status = executor.for_threads([&](std::size_t thread, std::size_t compute_domain) noexcept {
            std::size_t memory_index = 0;
            fu_memory_domain_id_t memory_domain = -1;
            if (fu_local_memory_of(topology, compute_domain, &memory_index) != fu_success_k ||
                fu_memory_domain_id_at_index(topology, memory_index, &memory_domain) != fu_success_k)
                return;
            executor.arenas_per_thread_[thread] = thread_arena_t {forkunion_allocator<std::byte>(memory_domain)};
        });
        return {std::move(executor), status};
    }

    /** @brief Relinquishes the pool without deleting it - the inverse of @ref acquire, so the
     *      same pool can back a different executor next; the arenas drop and this empties. */
    fu_pool_t release() noexcept {
        arenas_per_thread_.clear();
        topology_ = nullptr;
        return std::exchange(pool_, nullptr);
    }

    /** @brief Spawns @p threads workers over @p topology and takes the pool; zero takes seven
     *      eighths of the cores, since full occupancy starves the ordered publication path. */
    static expected<forkunion_executor_t> spawn(fu_topology_t topology, std::size_t threads) noexcept {
        fu_pool_t pool = nullptr;
        if (fu_pool_new("usearch", fu_capabilities_all_k, &pool) != fu_success_k)
            return {forkunion_executor_t {}, status_t::bad_alloc_k};
        std::size_t cores = 0;
        if (!threads && fu_logical_cores_count(topology, &cores) != fu_success_k)
            return fu_pool_delete(pool), expected<forkunion_executor_t> {{}, status_t::uninitialized_k};
        std::size_t const wanted = threads ? threads : cores - cores / 8;
        if (fu_pool_spawn(topology, pool, wanted, fu_caller_exclusive_k) != fu_success_k)
            return fu_pool_delete(pool), expected<forkunion_executor_t> {{}, status_t::uninitialized_k};
        return acquire(topology, pool);
    }

    /** @brief One call's exclusive hold on the executor, released on destruction - like a capacity
     *      claim, so an early exit from a bulk verb can never leave the executor held. */
    struct executor_claim_t {
        executor_claim_t() noexcept = default;
        /** @brief The granted hold on @p executor; only @ref claim ever calls it. */
        explicit executor_claim_t(forkunion_executor_t& executor) noexcept : executor_(&executor) {}
        executor_claim_t(executor_claim_t&& other) noexcept { swap(other); }
        executor_claim_t& operator=(executor_claim_t&& other) noexcept { return swap(other), *this; }
        executor_claim_t(executor_claim_t const&) = delete;
        executor_claim_t& operator=(executor_claim_t const&) = delete;
        ~executor_claim_t() noexcept {
            if (executor_) executor_->held_.clear(std::memory_order_release);
        }

        void swap(executor_claim_t& other) noexcept { std::swap(executor_, other.executor_); }
        explicit operator bool() const noexcept { return executor_ != nullptr; }

      private:
        /** @brief The held executor, reached only to release the gate; null marks the empty state
         *      a default construction and a move-out both leave behind. */
        forkunion_executor_t* executor_ = nullptr;
    };

    /** @brief Claims the executor for one call, reporting @c contended_k to a second concurrent
     *      caller - the executor serializes, never the indexes it drives, so genuine concurrency
     *      is one executor per calling thread. */
    expected<executor_claim_t> claim() noexcept {
        if (held_.test_and_set(std::memory_order_acquire)) return {executor_claim_t {}, status_t::contended_k};
        return {executor_claim_t {*this}, status_t::success_k};
    }

    /**
     *  @brief Claims at least @p bytes_per_thread of scratch for every worker, each run on the
     *      memory domain local to it, so a run and the thread using it share a node.
     *
     *  Monotone: the runs grow to the widest ask and are kept between calls, so a steady workload
     *  allocates once and a narrower later call costs one compare per thread.
     */
    status_t reserve(std::size_t bytes_per_thread) noexcept {
        if (arenas_per_thread_.empty()) return status_t::uninitialized_k;
        // Rewind before growing: a relocation moves only dead carves, never live ones.
        for (thread_arena_t& arena : arenas_per_thread_) {
            arena.clear();
            if (failed(arena.reserve_exactly(bytes_per_thread))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** @brief Worker @p thread 's own arena; no two ever share one, so a traversal context built
     *      over it needs no synchronization at all. */
    thread_arena_t& arena(std::size_t thread) noexcept { return arenas_per_thread_[thread]; }

    /** @brief Threads this executor drives - one worker, one arena, one scratch index. */
    std::size_t size() const noexcept { return arenas_per_thread_.size(); }
    fu_topology_t topology() const noexcept { return topology_; }

    /** @brief Runs @p callback once on every worker, with the thread index and its compute
     *      domain - for per-thread setup the OS demands, like the AMX tile-permission
     *      handshake; must be @c noexcept. */
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

    /**
     *  @brief Runs @p callback over @p tasks items, stolen dynamically.
     *  @param[in] tasks Work items to distribute; zero is a legal no-op.
     *  @param[in] callback Invoked on a worker with the task index, the thread index and that
     *      thread's compute domain, in that order; must be @c noexcept.
     */
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
    /** @brief One arena per worker thread, colocated with it once at @ref acquire - a context's
     *      rebound allocators point at the cursor inside its entry, so relocating this dangles. */
    growing_buffer<thread_arena_t> arenas_per_thread_;
    /** @brief The one-caller gate @ref claim spins on; never swapped, since a move with a call in
     *      flight is already misuse. */
    std::atomic_flag held_;
    /** @brief The pool every fan-out dispatches on, owned outright and deleted here. */
    fu_pool_t pool_ = nullptr;
    fu_topology_t topology_ = nullptr;
};

#pragma endregion Executor

} // namespace usearch
} // namespace unum
