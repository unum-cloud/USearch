/**
 *  @file include/usearch/utilities/ram_device.hpp
 *  @author Ash Vardanian
 *  @date August 30, 2026
 *  @brief Host-side vocabulary for driving one CPU machine - domain placement & executors.
 *
 *  Everything a RAM-resident index needs from the machine and nothing it stores: the
 *  ForkUnion-backed allocator binding storage to one memory domain, and the executor holding
 *  the threads with one byte arena apiece. A thread's scratch is a run of bytes, so one
 *  executor drives indexes of any quantization, metric or capability tier.
 */
#pragma once

#include <cstdlib> // `std::aligned_alloc`, `std::free` - the heap fallback of a domainless allocator

#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <type_traits> // `std::remove_reference_t`, `std::true_type`, `std::is_same_v` - callbacks, moves, hints
#include <utility>     // `std::swap`, `std::move`, `std::exchange`

#include <forkunion/forkunion.h>    // `fu_pool_spawn`, `fu_pool_spawn_near_memory_domain`, `fu_local_memory_of`
#include <forkunion/allocators.hpp> // `fu::domain_allocator_t`, `fu::page_size_1g_k` - the payload pages
#include <forkunion/types.hpp>      // `fu::spin_mutex_t` - the hold, `fu::is_cache_hints_functor` - the line tags

#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the arenas' storage
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`

namespace unum::usearch::utilities {

/*  The two projects derive their padding width from the same target facts in two headers that
 *  cannot see each other - `utilities/types.hpp` forward-declares the sister namespaces rather
 *  than including them. This is the first point both are visible, so it is where drift surfaces. */
static_assert(default_alignment_k == fu::default_alignment_k,
              "USEARCH_DEFAULT_ALIGNMENT and FORKUNION_DEFAULT_ALIGNMENT disagree");

/** ForkUnion's hints take ForkUnion's line tags; its standard hints alone emit nothing. */
template <typename cache_hints_type_>
    requires fu::is_cache_hints_functor<cache_hints_type_>::value
struct cache_line_tags<cache_hints_type_> {
    inline static constexpr bool prefetches_k = !std::is_same_v<cache_hints_type_, fu::standard_cache_hints_t>;
    inline static constexpr fu::prefetch_line_t prefetch_k {};
    inline static constexpr fu::demote_line_t demote_k {};
    inline static constexpr fu::promote_line_t promote_k {};
};

#pragma region Placement

/** The domain ID of a machine without NUMA; ForkUnion asserts on it, so the heap serves. */
inline constexpr fu_memory_domain_id_t no_memory_domain_k = -1;

/** Allocates on one ForkUnion memory domain, so storage is placed beside the threads that walk
 *  it; @c no_memory_domain_k means the plain heap, which also serves every block under a page, as
 *  per-entry packs and neighbor lists are. It claims 1 GiB pages for blocks of a GiB or more while
 *  the pool covers them, and base pages otherwise. */
template <typename value_type_ = byte_t>
struct forkunion_allocator {
    using value_type = value_type_;

    /** A moved container keeps freeing on the domain its storage came from, rather than the target's default heap. */
    using propagate_on_container_move_assignment = std::true_type;

    /** The OS memory-domain ID storage is claimed on. */
    fu_memory_domain_id_t memory_domain = no_memory_domain_k;

    constexpr forkunion_allocator() noexcept = default;
    explicit constexpr forkunion_allocator(fu_memory_domain_id_t memory_domain) noexcept
        : memory_domain(memory_domain) {}

    /** The rebinding conversion @c std::allocator_traits performs; the domain travels. */
    template <typename other_type_>
    constexpr forkunion_allocator(forkunion_allocator<other_type_> const& other) noexcept
        : memory_domain(other.memory_domain) {}

    /** Every run starts on @c default_alignment_k: arenas slice on that boundary and size
     *  themselves assuming an aligned base, so the heap fallback aligns like the domains do. */
    value_type_* allocate(std::size_t count) noexcept {
        std::size_t const bytes = aligned_bytes_(count);
        if (heap_serves_(bytes)) return static_cast<value_type_*>(std::aligned_alloc(default_alignment_k, bytes));
        if constexpr (places_by_mmap_k) {
            fu::domain_allocator_t pages {static_cast<fu::memory_domain_id_t>(memory_domain)};
            char* memory = nullptr;
            if (bytes >= fu::page_size_1g_k) memory = pages.allocate_at_least(bytes, fu::page_size_1g_k).ptr;
            if (!memory) memory = pages.allocate(mapped_bytes_(bytes), fu::ram_page_size());
            return reinterpret_cast<value_type_*>(memory);
        }
        void* memory = nullptr;
        if (fu_allocate_on_domain_id(memory_domain, bytes, &memory) != fu_success_k) return nullptr;
        return static_cast<value_type_*>(memory);
    }

    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (heap_serves_(aligned_bytes_(count))) return std::free(pointer);
        if constexpr (places_by_mmap_k)
            return fu::domain_allocator_t {static_cast<fu::memory_domain_id_t>(memory_domain)}.deallocate(
                reinterpret_cast<char*>(pointer), mapped_bytes_(aligned_bytes_(count)));
        fu_free_on_domain_id(memory_domain, pointer, aligned_bytes_(count));
    }

    bool operator==(forkunion_allocator const& other) const noexcept { return memory_domain == other.memory_domain; }

  private:

    /** Whether ForkUnion places domains with Linux or FreeBSD @c mmap, where the page size is ours
     *  to pick. */
    inline static constexpr bool places_by_mmap_k =
        std::is_same_v<fu::domain_allocator_t, fu::linux_numa_allocator_t> ||
        std::is_same_v<fu::domain_allocator_t, fu::freebsd_numa_allocator_t>;

    /** Whether the heap places a block of @p bytes: a domain maps a page and a syscall apiece, so
     *  a small block taking it costs a whole page and a page-splitting @c munmap to free. */
    bool heap_serves_(std::size_t bytes) const noexcept {
        return memory_domain == no_memory_domain_k || bytes < fu::ram_page_size();
    }

    /** The byte count a run of @p count elements is claimed and released with. */
    static std::size_t aligned_bytes_(std::size_t count) noexcept {
        return divide_round_up<std::size_t>(count * sizeof(value_type_), default_alignment_k) * default_alignment_k;
    }

    /** The length a block of @p bytes maps. The base-page fallback maps the same rounded length,
     *  as @c munmap refuses part of a 1 GiB page and release cannot tell which it got. */
    static std::size_t mapped_bytes_(std::size_t bytes) noexcept {
        return bytes < fu::page_size_1g_k ? bytes : round_up_to_multiple<std::size_t>(bytes, fu::page_size_1g_k);
    }
};

/** The byte specialization every container and index places its storage through. */
using forkunion_allocator_t = forkunion_allocator<byte_t>;

static_assert(buffer_allocator<forkunion_allocator_t>);

#pragma endregion Placement

#pragma region Executor

/** The threads a bulk verb advances on and the scratch they walk with - one arena per
 *  thread, so a fan-out never leases or waits. Admits one caller at a time. */
struct forkunion_executor_t {

    /** One thread's scratch on its own memory domain: chunks that double and never move,
     *  rewound between calls. */
    using arena_t = growing_arena<forkunion_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** The label the pool's threads carry in the OS. */
    inline static constexpr char const executor_name_k[] = "usearch";

    /** One core in this many stays idle on a zero ask, for the ordered publication path. */
    inline static constexpr std::size_t idle_cores_denominator_k = 8;

    forkunion_executor_t() = default;
    forkunion_executor_t(forkunion_executor_t&& other) noexcept { swap(other); }
    forkunion_executor_t& operator=(forkunion_executor_t&& other) noexcept { return swap(other), *this; }
    forkunion_executor_t(forkunion_executor_t const&) = delete;
    forkunion_executor_t& operator=(forkunion_executor_t const&) = delete;

    /** Deletes the pool and every arena, unless @ref release reclaimed the pool first. */
    ~forkunion_executor_t() noexcept {
        if (pool_) fu_pool_delete(pool_);
    }

    void swap(forkunion_executor_t& other) noexcept {
        std::swap(arenas_per_thread_, other.arenas_per_thread_);
        std::swap(pool_, other.pool_);
        std::swap(topology_, other.topology_);
    }

    /**
     *  Takes over @p pool, owned and deleted here from now on - even on failure. Colocates
     *  one arena with each thread's memory domain.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
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

    /** Relinquishes the pool without deleting it - the inverse of @ref acquire, so the
     *  same pool can back a different executor next; the arenas drop and this empties. */
    fu_pool_t release() noexcept {
        arenas_per_thread_.clear();
        topology_ = nullptr;
        return std::exchange(pool_, nullptr);
    }

    /** The thread count a zero ask means: every core but one in @ref idle_cores_denominator_k, or
     *  the caller alone in a build without OS threads. */
    static expected<std::size_t> default_threads(fu_topology_t topology) noexcept {
        if (!(fu_comptime_capabilities() & fu_capability_os_threads_k)) return {1, status_t::success_k};
        std::size_t cores = 0;
        if (fu_logical_cores_count(topology, &cores) != fu_success_k) return {0, status_t::uninitialized_k};
        return {cores - cores / idle_cores_denominator_k, status_t::success_k};
    }

    /**
     *  Spawns @p threads over @p topology and takes the pool; zero means @ref default_threads. One
     *  thread, or a build without OS threads, is the caller itself.
     *
     *  @par Statuses:
     *  - @c contended_k The fresh pool is running another dispatch.
     */
    static expected<forkunion_executor_t> spawn(fu_topology_t topology, std::size_t threads,
                                                fu_capabilities_t allowed = fu_capabilities_all_k) noexcept {
        auto const [count, status] = threads ? expected<std::size_t> {threads, status_t::success_k}
                                             : default_threads(topology);
        if (failed(status)) return {{}, status};
        bool const os_threads = fu_comptime_capabilities() & fu_capability_os_threads_k;
        return spawn(topology, count, count != 1 && os_threads ? fu_caller_exclusive_k : fu_caller_inclusive_k,
                     allowed);
    }

    /**
     *  Spawns @p threads over @p topology with the caller inside the pool or outside it, as
     *  @p exclusivity says, using only the ForkUnion facilities @p allowed names; zero threads
     *  means @ref default_threads.
     *
     *  @par Statuses:
     *  - @c contended_k The fresh pool is running another dispatch.
     */
    static expected<forkunion_executor_t> spawn(fu_topology_t topology, std::size_t threads,
                                                fu_caller_exclusivity_t exclusivity,
                                                fu_capabilities_t allowed) noexcept {
        if (!threads) {
            auto const [count, status] = default_threads(topology);
            if (failed(status)) return {{}, status};
            threads = count;
        }
        // A flat pool, unlike a distributed one, never pins the caller's own thread.
        if (exclusivity == fu_caller_inclusive_k)
            allowed = static_cast<fu_capabilities_t>(allowed & ~fu_capability_place_memory_on_domain_k);
        fu_pool_t created = nullptr;
        if (fu_pool_new(executor_name_k, allowed, &created) != fu_success_k) return {{}, status_t::bad_alloc_k};
        if (fu_pool_spawn(topology, created, threads, exclusivity) != fu_success_k)
            return fu_pool_delete(created), expected<forkunion_executor_t> {{}, status_t::uninitialized_k};
        return acquire(topology, created);
    }

    /**
     *  @brief Spawns @p threads over the compute domains local to @p memory_domain, the caller
     *      outside the pool, using only the ForkUnion facilities @p allowed names; zero means
     *      @ref default_threads.
     *
     *  @par Statuses:
     *  - @c unsupported_k A build without OS threads, or placement masked off on a machine of
     *    several memory domains.
     *  - @c invalid_argument_k No compute domain is local to @p memory_domain.
     *  - @c contended_k The fresh pool is running another dispatch.
     */
    static expected<forkunion_executor_t> spawn_near(fu_topology_t topology, fu_memory_domain_id_t memory_domain,
                                                     std::size_t threads, fu_capabilities_t allowed) noexcept {
        auto const [count, count_status] = threads ? expected<std::size_t> {threads, status_t::success_k}
                                                   : default_threads(topology);
        if (failed(count_status)) return {{}, count_status};
        fu_pool_t created = nullptr;
        if (fu_pool_new(executor_name_k, allowed, &created) != fu_success_k) return {{}, status_t::bad_alloc_k};
        status_t const spawned = dispatch_status_(
            fu_pool_spawn_near_memory_domain(topology, created, memory_domain, count, fu_caller_exclusive_k));
        if (failed(spawned)) return fu_pool_delete(created), expected<forkunion_executor_t> {{}, spawned};
        return acquire(topology, created);
    }

    /**
     *  Claims the executor for one call; a second concurrent caller gets @c contended_k.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, status_t::contended_k};
        return {claim_t {held_, std::adopt_lock}, status_t::success_k};
    }

    /** Rewinds every thread's arena and seeds it at @p bytes_per_thread, the first chunk's
     *  width; the widest chunk an earlier call opened stays, so the next call starts at its
     *  high-water mark.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no arenas.
     */
    status_t reserve(std::size_t bytes_per_thread) noexcept {
        if (arenas_per_thread_.empty()) return status_t::uninitialized_k;
        for (arena_t& arena : arenas_per_thread_) {
            arena.clear();
            if (failed(arena.reserve_exactly(bytes_per_thread))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Thread @p thread 's own arena; no two threads share one. */
    arena_t& arena(std::size_t thread) noexcept { return arenas_per_thread_[thread]; }

    /** The memory domain thread @p thread 's arena sits on - the thread's home, which a
     *  fan-out serves first. */
    fu_memory_domain_id_t memory_domain(std::size_t thread) const noexcept {
        return arenas_per_thread_[thread].upstream().memory_domain;
    }

    /** Threads this executor drives. */
    std::size_t size() const noexcept { return arenas_per_thread_.size(); }

    /** The topology the pool was spawned over; borrowed, never owned. */
    fu_topology_t topology() const noexcept { return topology_; }

    /** Nothing is ever pending: every fan-out blocks inside, so a drain is a no-op. */
    status_t wait() noexcept { return status_t::success_k; }

    /**
     *  Runs @p callback on every thread with the thread index and its compute domain; must be
     *  @c noexcept. A pool running another dispatch answers @c contended_k, and nothing ran.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no pool, or the pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
    template <typename callback_type_>
    status_t for_threads(callback_type_&& callback) noexcept {
        if (!pool_) return status_t::uninitialized_k;
        fu_for_threads_t const trampoline = [](fu_lambda_context_t context, std::size_t thread,
                                               std::size_t compute_domain) {
            (*static_cast<std::remove_reference_t<callback_type_>*>(context))(thread, compute_domain);
        };
        return dispatch_status_(fu_pool_for_threads(pool_, trampoline, &callback));
    }

    /** Runs @p callback over @p tasks items, stolen dynamically, with the task index, the thread
     *  index and its compute domain; must be @c noexcept. A pool running another dispatch answers
     *  @c contended_k, and nothing ran.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no pool, or the pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
    template <typename callback_type_>
    status_t for_tasks(std::size_t tasks, callback_type_&& callback) noexcept {
        if (!pool_) return status_t::uninitialized_k;
        if (!tasks) return status_t::success_k;
        fu_for_task_t const trampoline = [](fu_lambda_context_t context, std::size_t task, std::size_t thread,
                                            std::size_t compute_domain) {
            (*static_cast<std::remove_reference_t<callback_type_>*>(context))(task, thread, compute_domain);
        };
        return dispatch_status_(fu_pool_for_n_dynamic(pool_, tasks, trampoline, &callback));
    }

  private:

    /** What a fan-out answers for ForkUnion's @p status: a nested or concurrent dispatch is
     *  @c contended_k, a pool without workers @c uninitialized_k. */
    static status_t dispatch_status_(fu_status_t status) noexcept {
        switch (status) {
        case fu_success_k: return status_t::success_k;
        case fu_contended_k: return status_t::contended_k;
        case fu_not_spawned_k: return status_t::uninitialized_k;
        case fu_invalid_argument_k: return status_t::invalid_argument_k;
        case fu_unsupported_k: return status_t::unsupported_k;
        default: return status_t::internal_k;
        }
    }

    /** One arena per thread, colocated with it once at @ref acquire; contexts point into it,
     *  so its chunks never move. */
    growing_buffer<arena_t> arenas_per_thread_ {global_allocator};

    /** Held by the one caller driving the threads; never swapped. */
    fu::spin_mutex_t held_;

    /** The pool every fan-out dispatches on, owned outright and deleted here. */
    fu_pool_t pool_ = nullptr;

    /** The topology the pool was spawned over; borrowed, never owned. */
    fu_topology_t topology_ = nullptr;
};

#pragma endregion Executor

} // namespace unum::usearch::utilities
