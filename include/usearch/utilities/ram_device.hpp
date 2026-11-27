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

#include <cassert> // `assert` - only a caller-exclusive pool splits a dispatch

#include <memory>      // `std::allocator_traits` - the placing allocator's propagation
#include <mutex>       // `std::unique_lock`, `std::adopt_lock` - the executor's one-caller hold
#include <new>         // `std::align_val_t`, `std::nothrow` - the heap fallback of a domainless allocator
#include <type_traits> // `std::true_type`, `std::is_same_v` - moves, hints
#include <utility>     // `std::swap`, `std::move`

#include <forkunion/forkunion.h>    // `fu_local_memory_of`, `fu_memory_domain_id_at_index`
#include <forkunion/types.hpp>      // `fu::dummy_pool_t`, `fu::dispatched_status` - the threads
#include <forkunion/allocators.hpp> // `fu::domain_allocator_t`, `fu::page_size_1g_k` - the payload pages
#include <forkunion/locks.hpp>      // `fu::spin_mutex_t` - the hold

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

/** The memory domain nearest the PCI function at @p bus_id, or @c no_memory_domain_k where the
 *  topology names none. */
inline fu_memory_domain_id_t memory_domain_of_pci(char const* bus_id) noexcept {
    fu_memory_domain_id_t domain = no_memory_domain_k;
    return fu_memory_domain_id_of_pci(bus_id, &domain) == fu_success_k ? domain : no_memory_domain_k;
}

/** Allocates on one ForkUnion memory domain, so storage is placed beside the threads that walk
 *  it; @c no_memory_domain_k means the plain heap, which also serves every block under a page, as
 *  per-entry packs and neighbor lists are, and every block where ForkUnion places nothing. It
 *  claims 1 GiB pages for blocks of a GiB or more while the pool covers them, and base pages
 *  otherwise. Every block of a page or more starts on a page, from the heap too, so a direct I/O
 *  buffer is taken as placed. */
template <typename value_type_ = byte_t>
struct forkunion_allocator {
    using value_type = value_type_;

    /** A moved or swapped container keeps freeing on the domain its storage came from. */
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    /** A domain unmaps the pages the heap would delete, so only one domain's allocators match. */
    using is_always_equal = std::false_type;

    /** The OS memory-domain ID storage is claimed on. */
    fu_memory_domain_id_t memory_domain = no_memory_domain_k;

    constexpr forkunion_allocator() noexcept = default;
    explicit constexpr forkunion_allocator(fu_memory_domain_id_t memory_domain) noexcept
        : memory_domain(memory_domain) {}

    /** The rebinding conversion @c std::allocator_traits performs; the domain travels. */
    template <typename other_type_>
    constexpr forkunion_allocator(forkunion_allocator<other_type_> const& other) noexcept
        : memory_domain(other.memory_domain) {}

    /** Every run starts on @c default_alignment_k, and a run of a page or more on a page: arenas
     *  slice on that boundary and size themselves assuming an aligned base, so the heap fallback
     *  aligns like the domains do. */
    value_type_* allocate(std::size_t count) noexcept {
        std::size_t const bytes = aligned_bytes_(count);
        if (heap_serves_(bytes))
            return static_cast<value_type_*>(::operator new(bytes, heap_alignment_(bytes), std::nothrow));
        if constexpr (places_by_mmap_k) {
            pages_allocator_t pages {static_cast<fu::memory_domain_id_t>(memory_domain)};
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
        if (heap_serves_(aligned_bytes_(count)))
            return ::operator delete(pointer, heap_alignment_(aligned_bytes_(count)));
        if constexpr (places_by_mmap_k)
            return pages_allocator_t {static_cast<fu::memory_domain_id_t>(memory_domain)}.deallocate(
                reinterpret_cast<char*>(pointer), mapped_bytes_(aligned_bytes_(count)));
        fu_free_on_domain_id(memory_domain, pointer, aligned_bytes_(count));
    }

    template <typename other_type_>
    bool operator==(forkunion_allocator<other_type_> const& other) const noexcept {
        return memory_domain == other.memory_domain;
    }

  private:

    /** Whether ForkUnion places domains with Linux or FreeBSD @c mmap, where the page size is ours
     *  to pick. */
    inline static constexpr bool places_by_mmap_k =
        std::is_same_v<fu::domain_allocator_t, fu::linux_numa_allocator_t> ||
        std::is_same_v<fu::domain_allocator_t, fu::freebsd_numa_allocator_t>;

    /** ForkUnion's placing allocator, named through the element type, so a platform whose
     *  allocator takes no page size never compiles the mapped path. */
    using pages_allocator_t = std::conditional_t<sizeof(value_type_) != 0, fu::domain_allocator_t, void>;

    /** Whether ForkUnion places memory on domains here at all, rather than on its heap. */
    inline static constexpr bool places_on_domains_k =
        !std::is_same_v<fu::domain_allocator_t, fu::portable_aligned_allocator_t>;

    /** Whether the heap places a block of @p bytes: a domain maps a page and a syscall apiece, so
     *  a small block taking it costs a whole page and a page-splitting @c munmap to free. */
    bool heap_serves_(std::size_t bytes) const noexcept {
        return !places_on_domains_k || memory_domain == no_memory_domain_k || bytes < fu::ram_page_size();
    }

    /** Where the heap aligns a block of @p bytes: on a page from a page up, as domains do. */
    static std::align_val_t heap_alignment_(std::size_t bytes) noexcept {
        return std::align_val_t {bytes < fu::ram_page_size() ? default_alignment_k : fu::ram_page_size()};
    }

    /** The byte count a run of @p count elements is claimed and released with. */
    static std::size_t aligned_bytes_(std::size_t count) noexcept {
        return divide_round_up<std::size_t>(count * sizeof(value_type_), default_alignment_k) * default_alignment_k;
    }

    /** The length a block of @p bytes maps. The base-page fallback maps the same rounded length,
     *  as @c munmap refuses part of a 1 GiB page and a release, handed only the count, cannot tell
     *  which it got. */
    static std::size_t mapped_bytes_(std::size_t bytes) noexcept {
        return bytes < fu::page_size_1g_k ? bytes : round_up_to_multiple<std::size_t>(bytes, fu::page_size_1g_k);
    }
};

/** The byte specialization every container and index places its storage through. */
using forkunion_allocator_t = forkunion_allocator<byte_t>;

static_assert(a_buffer_allocator<forkunion_allocator_t>);
static_assert(std::allocator_traits<forkunion_allocator_t>::propagate_on_container_move_assignment::value &&
              std::allocator_traits<forkunion_allocator_t>::propagate_on_container_swap::value &&
              !std::allocator_traits<forkunion_allocator_t>::is_always_equal::value);

#pragma endregion Placement

#pragma region Executor

/** Bytes of cache each core over @p topology keeps to itself, as ForkUnion names it per compute
 *  domain, at its smallest across domains, or zero where no platform source names one. */
inline std::size_t smallest_cache_bytes_per_core(fu_topology_t topology) noexcept {
    std::size_t domains = 0, smallest = 0;
    if (fu_compute_domains_count(topology, &domains) != fu_success_k) return 0;
    for (std::size_t domain = 0; domain != domains; ++domain) {
        std::size_t bytes = 0;
        if (fu_compute_cache_bytes_per_core_in(topology, domain, &bytes) != fu_success_k || !bytes) continue;
        smallest = smallest ? min_of(smallest, bytes) : bytes;
    }
    return smallest;
}

/** Bytes of scratch each thread of an executor holds at most when its caller names no limit. */
inline constexpr std::size_t scratch_limit_default_k = std::size_t {1} << 30;

/**
 *  @brief The threads a bulk verb advances on and the scratch they walk with - one arena per
 *      thread, so a fan-out never references or waits. Admits one caller at a time.
 *  @tparam threads_type_ The pool the threads come from: ForkUnion's serial @c fu::dummy_pool_t, or
 *      ForkUnion's compiled @c fu::pool_t, which @ref spawn fills.
 */
template <typename threads_type_ = fu::dummy_pool_t>
struct ram_executor {

    /** The pool every fan-out dispatches on. */
    using threads_t = threads_type_;

    /** One thread's scratch on its own memory domain: chunks that double and never move,
     *  rewound between calls. */
    using arena_t = growing_arena<forkunion_allocator_t>;

    /** One call's hold on the executor, released when it dies. */
    using claim_t = std::unique_lock<fu::spin_mutex_t>;

    /** The label the pool's threads hold in the OS. */
    inline static constexpr char const executor_name_k[] = "usearch";

    ram_executor() = default;
    ram_executor(ram_executor&& other) noexcept { swap(other); }
    ram_executor& operator=(ram_executor&& other) noexcept { return swap(other), *this; }
    ram_executor(ram_executor const&) = delete;
    ram_executor& operator=(ram_executor const&) = delete;

    void swap(ram_executor& other) noexcept {
        std::swap(arenas_per_thread_, other.arenas_per_thread_);
        std::swap(threads_, other.threads_);
        std::swap(cache_bytes_per_core_, other.cache_bytes_per_core_);
    }

    /**
     *  @brief The calling thread alone: no pool, thread or topology probe; one arena for this call
     *      only, holding at most @p scratch_limit bytes, zero for @ref scratch_limit_default_k.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The arena's slot could not be allocated.
     */
    static expected<ram_executor> on_calling_thread(std::size_t scratch_limit = 0) noexcept {
        ram_executor executor;
        if (failed(executor.arenas_per_thread_.reserve_exactly(1)))
            return {std::move(executor), {status_code_t::out_of_memory_k, "No room for the call's arena"}};
        executor.arenas_per_thread_.push_back_reserved(
            arena_t {forkunion_allocator_t {no_memory_domain_k}, limit_of_(scratch_limit)});
        return {std::move(executor), status_code_t::success_k};
    }

    /**
     *  Claims the executor for one call; a second concurrent caller gets @c contended_k.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept {
        if (!held_.lock_if_free()) return {{}, {status_code_t::contended_k, "Another call holds the executor"}};
        return {claim_t {held_, std::adopt_lock}, status_code_t::success_k};
    }

    /**
     *  @brief Rewinds every thread's arena and seeds it at @p bytes_per_thread, the first chunk's
     *      width; the widest chunk an earlier call opened stays, so a call starts at the last
     *      one's high-water mark.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no arenas.
     *  - @c out_of_memory_k An arena could not grow, or @p bytes_per_thread pass the scratch limit.
     */
    status_t reserve(std::size_t bytes_per_thread) noexcept {
        if (arenas_per_thread_.empty()) return status_code_t::uninitialized_k;
        for (arena_t& arena : arenas_per_thread_) {
            arena.clear();
            if (failed(arena.reserve_exactly(bytes_per_thread))) return status_code_t::out_of_memory_k;
        }
        return status_code_t::success_k;
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

    /** Bytes of cache each core of the pool keeps to itself: the tile every bulk verb fills. */
    std::size_t cache_bytes_per_core() const noexcept { return cache_bytes_per_core_; }

    /** Whether the caller runs a slice of every fan-out or stays out. */
    fu::caller_exclusivity_t exclusivity() const noexcept { return threads_.caller_exclusivity(); }

    /**
     *  @brief Starts @p callback on every thread with @p context and returns at once; a fan-out
     *      from thread 0's slice forks the other threads, and @ref unsafe_await ends the dispatch.
     *
     *  @par Statuses:
     *  - @c contended_k The pool is running another dispatch, and nothing ran.
     */
    status_t unsafe_for_threads(fu_for_threads_t callback, void* context) noexcept {
        // Thread 0 may end the call before this returns, so no other caller may read the token.
        fu_generation_t generation = 0;
        return dispatch_status(threads_.unsafe_for_threads(callback, context, generation));
    }

    /** Waits until the dispatch in flight, if any, is over; only the claim holder waits. */
    void unsafe_await() noexcept {
        // An exclusive pool reads no token, while an inclusive one stays held on zero.
        assert(threads_.caller_exclusivity() == fu::caller_exclusive_k && "Only an exclusive pool splits a dispatch");
        if (fu_pool_t const spawned = threads_.get()) fu_pool_unsafe_join(spawned, 0);
    }

    /** Parks the idle threads in the kernel until the next fan-out wakes them. */
    void park() noexcept {
        if (fu_pool_t const spawned = threads_.get()) fu_pool_sleep(spawned, 0);
    }

    /** Nothing is ever pending: every fan-out blocks inside, so a drain is a no-op. */
    status_t wait() noexcept { return status_code_t::success_k; }

    /**
     *  @brief Runs @p callback on every thread with the thread index and its compute domain; must
     *      be @c noexcept. A pool running another dispatch answers @c contended_k, and nothing ran.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
    template <typename callback_type_>
    status_t for_threads(callback_type_&& callback) noexcept {
        return dispatch_status(fu::dispatched_status(
            threads_.for_threads([&](typename threads_t::thread_in_domain_t thread_in_domain) noexcept {
                callback(thread_in_domain.thread, thread_in_domain.compute_domain);
            })));
    }

    /**
     *  @brief Runs @p callback over @p tasks items, stolen dynamically, with the task index, the
     *      thread index and its compute domain; must be @c noexcept. A pool running another
     *      dispatch answers @c contended_k, and nothing ran.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
    template <typename callback_type_>
    status_t for_tasks(std::size_t tasks, callback_type_&& callback) noexcept {
        if (!tasks) return status_code_t::success_k;
        return dispatch_status(fu::dispatched_status(threads_.for_n_dynamic(
            tasks, [&](std::size_t task, typename threads_t::thread_in_domain_t thread_in_domain) noexcept {
                callback(task, thread_in_domain.thread, thread_in_domain.compute_domain);
            })));
    }

    /**
     *  @brief Takes over @p taken, owned and deleted here from now on - even on failure. Colocates
     *      one arena with each thread's memory domain over @p topology, holding at most
     *      @p scratch_limit bytes, zero for @ref scratch_limit_default_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The pool has no workers.
     */
    static expected<ram_executor> acquire(fu_topology_t topology, threads_t&& taken,
                                          std::size_t scratch_limit = 0) noexcept {
        ram_executor executor;
        executor.threads_ = std::move(taken);
        std::size_t const threads = executor.threads_.threads_count();
        if (!threads) return {std::move(executor), status_code_t::uninitialized_k};
        // Each thread names its own memory domain; the arenas are then placed on those in order.
        growing_buffer<fu_memory_domain_id_t> domains {global_allocator};
        if (failed(domains.resize(threads))) return {std::move(executor), status_code_t::out_of_memory_k};
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
            return {std::move(executor), status_code_t::out_of_memory_k};
        for (fu_memory_domain_id_t const domain : domains)
            executor.arenas_per_thread_.push_back_reserved(
                arena_t {forkunion_allocator_t(domain), limit_of_(scratch_limit)});
        executor.cache_bytes_per_core_ = smallest_cache_bytes_per_core(topology);
        return {std::move(executor), status_code_t::success_k};
    }

    /**
     *  @brief Spawns @p threads of a new pool over @p topology, a @c fu_topology_t or ForkUnion's
     *      @c fu::topology_t, and takes it as @ref acquire does; only on the compute domains local
     *      to @p memory_domain unless it is @ref no_memory_domain_k. Instantiates only where
     *      ForkUnion's SDK declares @c threads_type_.
     *
     *  @par Statuses:
     *  - @c unsupported_k More than one thread in a build without OS threads.
     *  - @c invalid_argument_k No compute domain is local to @p memory_domain.
     */
    template <typename topology_type_>
    static expected<ram_executor> spawn(topology_type_ const& topology, std::size_t threads,
                                        fu::caller_exclusivity_t exclusivity = fu::caller_exclusive_k,
                                        fu_capabilities_t allowed = fu_capabilities_all_k,
                                        fu_memory_domain_id_t memory_domain = no_memory_domain_k,
                                        std::size_t scratch_limit = 0) noexcept {
        static_assert(!std::is_same_v<threads_type_, fu::dummy_pool_t>,
                      "The serial pool spawns no threads: take `on_calling_thread` instead");
        fu_topology_t handle = nullptr;
        if constexpr (std::is_same_v<topology_type_, fu_topology_t>) handle = topology;
        else handle = topology.get();
        threads_t created;
        status_t const spawned = dispatch_status(
            memory_domain == no_memory_domain_k
                ? created.spawn(handle, threads, exclusivity, allowed, executor_name_k)
                : created.spawn_near(handle, memory_domain, threads, exclusivity, allowed, executor_name_k));
        if (failed(spawned)) return {{}, spawned};
        return acquire(handle, std::move(created), scratch_limit);
    }

    /** What a fan-out or spawn answers for ForkUnion's @p status: a nested or concurrent dispatch
     *  is @c contended_k, a pool without workers @c uninitialized_k. */
    static status_t dispatch_status(fu::status_t status) noexcept {
        switch (static_cast<fu_status_t>(status)) {
        case fu_success_k: return status_code_t::success_k;
        case fu_bad_alloc_k: return status_code_t::out_of_memory_k;
        case fu_contended_k: return status_code_t::contended_k;
        case fu_not_spawned_k: return status_code_t::uninitialized_k;
        case fu_invalid_argument_k: return status_code_t::invalid_argument_k;
        case fu_unsupported_k: return status_code_t::unsupported_k;
        case fu_config_mismatch_k: return status_code_t::config_mismatch_k;
        default: return status_code_t::internal_k;
        }
    }

  private:

    /** The bytes an arena holds at most under a @p scratch_limit, zero taking the default. */
    static std::size_t limit_of_(std::size_t scratch_limit) noexcept {
        return scratch_limit ? scratch_limit : scratch_limit_default_k;
    }

    /** One arena per thread, colocated with it once at @ref acquire; contexts point into it,
     *  so its chunks never move. */
    growing_buffer<arena_t> arenas_per_thread_ {global_allocator};

    /** Held by the one caller driving the threads; never swapped. */
    fu::spin_mutex_t held_;

    /** The pool every fan-out dispatches on; an empty or moved-from one runs every fan-out on the
     *  calling thread alone. */
    threads_t threads_;

    /** Bytes of cache each core of the pool keeps to itself, read once at @ref acquire. */
    std::size_t cache_bytes_per_core_ = 0;
};

/** The executor over ForkUnion's serial pool, which every call names unless it spawns threads. */
using ram_executor_t = ram_executor<>;

#pragma endregion Executor

} // namespace unum::usearch::utilities
