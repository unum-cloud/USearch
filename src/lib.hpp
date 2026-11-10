/**
 *  @file src/lib.hpp
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What every modality's C surface shares: the outcome and progress crossings, the CPU
 *      classes, the executor and arena handles, and the one entry every verb's call goes through.
 *
 *  Each modality's dispatch unit includes this, so one verb reads the same on every modality. The
 *  executor, arena and outcome verbs are defined once, in @c dense_index.cuh.
 */
#pragma once

#include <cassert> // `assert` - a per-thread family is opened before it is asked
#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - strided keys and coordinates packed into the arena

#include <array>            // `std::array` - the predicate's batch
#include <bit>              // `std::bit_cast` - the C key struct & `us::utilities::uuid128_t` share sixteen bytes
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <memory>           // `std::unique_ptr` - the executor's topology
#include <mutex>            // `std::lock_guard` over the ticks' lock
#include <span>             // `std::span` - key batches crossing the ABI
#include <string_view>      // `std::string_view` - checkpoint paths
#include <thread>           // `std::this_thread::get_id` - the thread running the holding call's body
#include <type_traits>      // `std::is_same_v` - the residency discrimination
#include <variant>          // `std::variant`, `std::visit`

#include <forkunion/forkunion.h>      // `fu_topology_new` - the executor factories' discovery
#include <forkunion/atomics.hpp>      // `fu::standard_atomic_ref` & kin - the classes' atomic references
#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfe_t` & waiter kin
#include <forkunion/types.hpp>        // `fu::spin_mutex_t` - the ticks' lock
#include <numkong/capabilities.h>     // `nk_capability_t`, `nk_cpu_configure_thread` - compute masks, AMX grant
#include <stringzilla/capabilities.h> // `sz_capability_t` - the basic modalities' kit masks
#include <numkong/vector.hpp>         // `nk::vector_view` - the rows a value lookup resolves

#include "usearch/usearch.h"                 // every modality's argument structures
#include "usearch/dense/types.hpp"           // `dense::search_config_t` & kin - the configs every verb lowers
#include "usearch/utilities/nvme_device.hpp" // `nvme_executor`, `default_io_engine`, `io_engine_config_t`
#include "usearch/utilities/ram_device.hpp"  // `forkunion_executor_t`, `forkunion_allocator_t`, `no_memory_domain_k`
#include "usearch/utilities/sample.hpp"      // `sample_inputs`, `sample_outputs` - what SAMPLE reads and writes
#include "usearch/utilities/safegraphs_format.hpp" // `checkpoint_source_t` - a checkpoint on a device
#include "usearch/utilities/vram_device.hpp"       // `gpu_executor_t`, `gpu_arena_t`

using namespace unum::usearch;
using namespace unum::usearch::utilities;
namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;
namespace fu = ashvardanian::forkunion;

#pragma region Scaffolding

/** The C scalar vocabulary beside NumKong's - independent enumerations, so both crossings read
 *  one table. */
struct scalar_dtype_pair_t {
    usearch_scalar_t scalar;
    nk_dtype_t dtype;
};
inline constexpr scalar_dtype_pair_t scalar_dtype_pairs_k[] = {
    {usearch_scalar_f64_k, nk_f64_k},   {usearch_scalar_f32_k, nk_f32_k},   {usearch_scalar_bf16_k, nk_bf16_k},
    {usearch_scalar_f16_k, nk_f16_k},   {usearch_scalar_e5m2_k, nk_e5m2_k}, {usearch_scalar_e4m3_k, nk_e4m3_k},
    {usearch_scalar_e3m2_k, nk_e3m2_k}, {usearch_scalar_e2m3_k, nk_e2m3_k}, {usearch_scalar_i8_k, nk_i8_k},
    {usearch_scalar_i4x2_k, nk_i4_k},   {usearch_scalar_i32_k, nk_i32_k},   {usearch_scalar_u8_k, nk_u8_k},
    {usearch_scalar_u4x2_k, nk_u4_k},   {usearch_scalar_u1x8_k, nk_u1_k},
};

inline nk_dtype_t nk_dtype_of(usearch_scalar_t scalar) noexcept {
    for (scalar_dtype_pair_t const& pair : scalar_dtype_pairs_k)
        if (pair.scalar == scalar) return pair.dtype;
    return nk_dtype_unknown_k;
}

/** The C spelling of how an index's last load from a path read its parts. */
inline usearch_load_path_t usearch_load_path_of(vram_load_path_t load_path) noexcept {
    switch (load_path) {
    case vram_load_path_t::host_copy_k: return usearch_load_path_host_copy_k;
    case vram_load_path_t::cufile_k: return usearch_load_path_cufile_k;
    default: return usearch_load_path_none_k;
    }
}

inline usearch_scalar_t usearch_scalar_of(nk_dtype_t dtype) noexcept {
    for (scalar_dtype_pair_t const& pair : scalar_dtype_pairs_k)
        if (pair.dtype == dtype) return pair.scalar;
    return usearch_scalar_unknown_k;
}

/** Reports a handler hears when the caller leaves @c usearch_progress_t::steps zero, a tenth of the
 *  call apart. */
inline constexpr std::size_t default_progress_steps_k = 10;

/** Whether a call holds an executor: taken by exchange, given back by a store. */
enum class call_claim_t : u32_t {
    free_k = 0,
    held_k = 1,
};

/** The verbs over the words a call's waiters watch - its outcome's status, its executor's claim
 *  and the named-call counts of its handles - lowered onto one CPU class's reference and waiter. A
 *  release store is the same instruction on every class; the wait is what differs, and it watches
 *  the address, so no store notifies. */
struct outcome_verbs_t {
    void (*publish)(usearch_outcome_t*, usearch_status_t, std::memory_order) noexcept;
    usearch_status_t (*load)(usearch_outcome_t const*) noexcept;

    /** Loops on the class waiter while the status reads deferred. */
    usearch_status_t (*wait)(usearch_outcome_t const*) noexcept;

    /** Exchanges the claim to held with an acquire, answering what it read. */
    call_claim_t (*take_claim)(call_claim_t*) noexcept;

    /** Stores the claim free with a release. */
    void (*release_claim)(call_claim_t*) noexcept;

    /** Loops on the class waiter while the claim reads held. */
    void (*wait_claim)(call_claim_t const*) noexcept;

    /** Counts one more call naming a handle; the claim's acquire orders it. */
    void (*raise_named)(u32_t*) noexcept;

    /** Counts one call naming a handle off, with a release. */
    void (*lower_named)(u32_t*) noexcept;

    /** Loops on the class waiter until the count reads zero. */
    void (*wait_named)(u32_t const*) noexcept;
};

/** The row of the newest class this machine admits, picked once by the CPU ladder walk. */
inline outcome_verbs_t const& host_outcome_verbs() noexcept;

/** Marks @p outcome as running; no waiter watches for this store, so it is relaxed. */
inline void propagate_pending(usearch_outcome_t* outcome) noexcept {
    outcome->message = nullptr;
    host_outcome_verbs().publish(outcome, usearch_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status every waiter acquires. The call
 *  touches nothing it was passed after this, and the message is null exactly on success. */
inline bool propagate_error(usearch_status_t status, usearch_outcome_t* outcome, char const* message) noexcept {
    bool const succeeded = status == usearch_status_success_k;
    outcome->message = succeeded ? nullptr : message;
    host_outcome_verbs().publish(outcome, status, std::memory_order_release);
    return succeeded;
}

/** The engine-side overload: the C++ mirror shares the ABI's values, so crossing is a cast, and an
 *  unnamed @p message falls back to the status's own static text. */
inline bool propagate_error(status_t status, usearch_outcome_t* outcome, char const* message = nullptr) noexcept {
    return propagate_error(static_cast<usearch_status_t>(status), outcome,
                           message ? message : status_to_string(status));
}

/** Adds one fan-out's telemetry into the caller's stats - accumulating, so chunked verbs report the
 *  whole call rather than the last chunk. */
template <typename result_type_>
inline void accumulate_stats(usearch_traversal_stats_t* stats, result_type_ const& result) noexcept {
    if constexpr (requires { result.telemetry.visited_members; }) accumulate_stats(stats, result.telemetry);
    else if constexpr (requires { result.traversal_stats.visited_members; })
        accumulate_stats(stats, result.traversal_stats);
    else if (stats) {
        stats->visited_members += result.visited_members;
        stats->computed_distances += result.computed_distances;
        stats->restarts += result.restarts;
        if constexpr (requires { result.missing_hints; }) stats->missing_hints += result.missing_hints;
    }
}

/** The traversal stats @p result carries: itself, the telemetry it holds, or its own counters. */
template <typename result_type_>
inline traversal_stats_t traversal_stats_of(result_type_ const& result) noexcept {
    if constexpr (std::is_same_v<result_type_, traversal_stats_t>) return result;
    else if constexpr (requires { result.telemetry.visited_members; }) return traversal_stats_of(result.telemetry);
    else if constexpr (requires { result.traversal_stats.visited_members; }) return result.traversal_stats;
    else {
        traversal_stats_t stats;
        stats.computed_distances = result.computed_distances;
        stats.visited_members = result.visited_members;
        stats.restarts = result.restarts;
        if constexpr (requires { result.missing_hints; }) stats.missing_hints = result.missing_hints;
        return stats;
    }
}

/** Places one shard per memory domain the process may allocate on through @p emplace, as
 *  <tt>emplace(domain) → status_t</tt>, over a transient topology; @c uninitialized_k where it
 *  names none. */
template <typename emplace_type_>
status_t emplace_on_every_domain(emplace_type_&& emplace) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k) return status_t::uninitialized_k;
    // The topology dies on every path, so the domains are walked before it is deleted.
    auto place_on_every_domain = [&]() noexcept {
        std::size_t domains_count = 0;
        if (fu_memory_domains_count(topology, &domains_count) != fu_success_k || !domains_count)
            return status_t::uninitialized_k;
        for (std::size_t position = 0; position != domains_count; ++position) {
            fu_memory_domain_id_t domain = -1;
            if (fu_memory_domain_id_at_index(topology, position, &domain) != fu_success_k)
                return status_t::uninitialized_k;
            if (status_t const placed = emplace(domain); failed(placed)) return placed;
        }
        return status_t::success_k;
    };
    status_t const placed = place_on_every_domain();
    fu_topology_delete(topology);
    return placed;
}

/** The domain host tables sit on: the first @p ram names, else the heap's. */
inline fu_memory_domain_id_t tables_domain(usearch_ram_options_t const& ram) noexcept {
    return ram.memory_domains && ram.memory_domains_count ? static_cast<fu_memory_domain_id_t>(ram.memory_domains[0])
                                                          : no_memory_domain_k;
}

/** The host allocator a variant keeps its tables on: @ref tables_domain on RAM, the heap's
 *  otherwise. */
template <typename residency_options_type_>
inline forkunion_allocator_t host_allocator_of(residency_options_type_ const& options) noexcept {
    if constexpr (std::is_same_v<residency_options_type_, usearch_ram_options_t>)
        return forkunion_allocator_t {tables_domain(options)};
    else return forkunion_allocator_t {no_memory_domain_k};
}

/** Whether a result carries traversal telemetry - one probe both crossings read. */
template <typename value_type_>
concept carries_telemetry = requires(value_type_ const& value) { value.visited_members; } ||
                            requires(value_type_ const& value) { value.telemetry.visited_members; } ||
                            requires(value_type_ const& value) { value.traversal_stats.visited_members; };

/** The one crossing for a fallible step: folds telemetry into @p stats when the payload carries
 *  any, finishes @p outcome only on failure, and answers whether @c .value is readable. */
template <typename value_type_>
inline bool crossed(expected<value_type_> const& result, usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,
                    char const* message = nullptr) noexcept {
    if constexpr (carries_telemetry<value_type_>) accumulate_stats(stats, result.value);
    return result || propagate_error(result.status, outcome, message);
}

/** The C key struct and the engine's @c utilities::uuid128_t share sixteen bytes: one bit cast. */
inline us::utilities::uuid128_t uuid_of(usearch_key_t key) noexcept {
    return std::bit_cast<us::utilities::uuid128_t>(key);
}

/** Admits every candidate, for a call that names no predicate. */
struct no_predicate_t {};

/** The C predicate over keys, asked per scored quad through arrays of its own key and admission
 *  types on the stack. */
struct cheap_predicate_t {
    usearch_predicate_t predicate;

    template <std::size_t extent_>
    void operator()(std::span<us::utilities::uuid128_t const, extent_> keys,
                    std::span<admission_t, extent_> admissions) const noexcept {
        std::array<usearch_key_t, extent_> c_keys;
        std::array<usearch_admission_t, extent_> c_admissions;
        for (std::size_t lane = 0; lane != extent_; ++lane) c_keys[lane] = std::bit_cast<usearch_key_t>(keys[lane]);
        predicate.callback(c_keys.data(), c_admissions.data(), extent_, predicate.state);
        for (std::size_t lane = 0; lane != extent_; ++lane)
            admissions[lane] = static_cast<admission_t>(c_admissions[lane]);
    }
};

/** The C predicate over keys, asked once per round a thread settles, over the keys it gathers. */
struct expensive_predicate_t {
    usearch_predicate_t predicate;
};

/** The caller's predicate, settled once per call and named by its cost, as C's
 *  @c usearch_predicate_cost_t names it. */
using predicate_t = std::variant<no_predicate_t, cheap_predicate_t, expensive_predicate_t>;

inline predicate_t predicate_of(usearch_predicate_t predicate) noexcept {
    if (!predicate.callback) return no_predicate_t {};
    if (predicate.cost == usearch_predicate_cost_expensive_k) return expensive_predicate_t {predicate};
    return cheap_predicate_t {predicate};
}

/** The C value lookup lifted onto the index's key vocabulary: one callback per scored batch, at
 *  the batch's static extent, a NULL row answering an empty view that refuses the call. A GPU walk
 *  takes it as bytes and calls it one key at a time on the device. */
struct value_lookup_t {
    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

    template <typename scalar_type_, std::size_t extent_>
    void operator()(std::span<us::utilities::uuid128_t const, extent_> keys,
                    std::span<nk::vector_view<scalar_type_>, extent_> rows) const noexcept {
        std::array<usearch_key_t, extent_> batch;
        std::array<void const*, extent_> found;
        for (std::size_t lane = 0; lane != extent_; ++lane) batch[lane] = std::bit_cast<usearch_key_t>(keys[lane]);
        callback(state, batch.data(), extent_, scalar, dimensions, found.data());
        for (std::size_t lane = 0; lane != extent_; ++lane)
            rows[lane] = found[lane]
                             ? nk::vector_view<scalar_type_> {static_cast<scalar_type_ const*>(found[lane]), dimensions}
                             : nk::vector_view<scalar_type_> {};
    }
};

/** Spells @c value_lookup_t into the name of an attend kernel instantiated over it. */
inline void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<value_lookup_t>) noexcept {
    symbol.append_name("value_lookup_t");
}

/** The allocator every per-call buffer of one executor thread comes from: its arena, rewound per
 *  verb. */
using thread_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;

/**
 *  @brief An expensive predicate: keys gather beside the engine's @p payload_type_ until the engine
 *      settles a round, then one call answers them all.
 *
 *  One per thread, its buffers reserved for the widest round its engine settles, the only buffers a
 *  predicate ever holds.
 */
template <typename payload_type_>
struct expensive_predicate {
    usearch_predicate_t predicate;
    growing_buffer<usearch_key_t, thread_allocator_t> keys;
    growing_buffer<payload_type_, thread_allocator_t> payloads;
    growing_buffer<usearch_admission_t, thread_allocator_t> admissions;

    expensive_predicate(usearch_predicate_t handler, thread_allocator_t const& allocator) noexcept
        : predicate(handler), keys(allocator), payloads(allocator), admissions(allocator) {}

    /** Reserves for rounds of @p round_capacity candidates. */
    status_t reserve(std::size_t round_capacity) noexcept {
        if (failed(keys.reserve_exactly(round_capacity)) || failed(payloads.reserve_exactly(round_capacity)) ||
            failed(admissions.reserve_exactly(round_capacity)))
            return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    template <std::size_t extent_>
    void push(std::span<us::utilities::uuid128_t const, extent_> asked,
              std::span<payload_type_ const, extent_> carried) noexcept {
        for (std::size_t lane = 0; lane != extent_; ++lane) {
            keys.push_back_reserved(std::bit_cast<usearch_key_t>(asked[lane]));
            payloads.push_back_reserved(carried[lane]);
        }
    }

    template <typename on_admitted_type_>
    void drain(on_admitted_type_&& on_admitted) noexcept {
        if (keys.empty()) return;
        admissions.resize_reserved(keys.size());
        predicate.callback(keys.data(), admissions.data(), keys.size(), predicate.state);
        for (std::size_t position = 0; position != keys.size(); ++position)
            if (admissions[position] == usearch_admit_k) on_admitted(payloads[position]);
        keys.clear();
        payloads.clear();
        admissions.clear();
    }
};

/** One expensive predicate per executor thread, for engines that drive their own threads: each
 *  thread gathers into its own buffers on its own arena and calls the handler once per round it
 *  settles. The engine opens the family right after rewinding its arenas, building the array in
 *  the first one, so a warm call allocates nothing. */
template <typename payload_type_>
struct expensive_predicates {
    using thread_predicate_t = expensive_predicate<payload_type_>;

    usearch_predicate_t predicate;
    growing_buffer<thread_predicate_t, thread_allocator_t> threads {thread_allocator_t {}};

    /** Builds one predicate per thread of @p executor over that thread's arena, the array in the
     *  first; the engine reserves each for its rounds. */
    template <typename executor_type_>
    status_t open(executor_type_& executor) noexcept {
        threads = growing_buffer<thread_predicate_t, thread_allocator_t> {thread_allocator_t {executor.arena(0)}};
        if (failed(threads.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            threads.push_back_reserved(thread_predicate_t {predicate, thread_allocator_t {executor.arena(thread)}});
        return status_t::success_k;
    }

    thread_predicate_t& of_thread(std::size_t thread) noexcept {
        assert(thread < threads.size() && "Engines open a per-thread family before asking it");
        return threads[thread];
    }
};

/** One call's reports: the C handler, serialized for the executor's threads, which the contract
 *  promises it never sees concurrently, thinned to the caller's steps, and the slice a batch verb
 *  hands its engine whole. */
struct progress_ticks_t {
    usearch_progress_t progress;
    std::size_t total;
    std::size_t done = 0;   // Counted under the mutex that already serializes the handler.
    std::size_t closed = 0; // The last step a forwarded tick closed.
    fu::spin_mutex_t mutex;

    progress_ticks_t(usearch_progress_t progress, std::size_t total) noexcept : progress(progress), total(total) {}

    /** Units each slice of a batch spans: a step's share with a handler, the whole batch
     *  without. */
    std::size_t slice() const noexcept {
        return progress.callback ? divide_round_up<std::size_t>(total, steps_()) : total;
    }

    continuation_t operator()(std::size_t) noexcept {
        if (!progress.callback) return continuation_t::continue_k;
        std::lock_guard guard {mutex};
        return forward_(++done, total);
    }

    /** The sibling for verbs that count their own units and know their own population - a
     *  refinement ticks the nodes visited so far, which no counter here could derive. */
    continuation_t operator()(std::size_t finished, std::size_t population) noexcept {
        if (!progress.callback) return continuation_t::continue_k;
        std::lock_guard guard {mutex};
        return forward_(finished, population);
    }

  private:
    std::size_t steps_() const noexcept { return progress.steps ? progress.steps : default_progress_steps_k; }

    /** Hands the C handler a tick that closes a new step, or the last one; the caller holds the
     *  mutex. */
    continuation_t forward_(std::size_t finished, std::size_t population) noexcept {
        std::size_t const step = finished >= population ? steps_() : finished / divide_round_up(population, steps_());
        if (step <= closed) return continuation_t::continue_k;
        closed = step;
        return progress.callback(finished, population, progress.state) == usearch_cancel_k ? continuation_t::cancel_k
                                                                                           : continuation_t::continue_k;
    }
};

#pragma endregion Scaffolding

#pragma region Composites

using dense::candidates_count_t;
using dense::insert_config_t;
using dense::refine_config_t;
using dense::search_config_t;

/** A split's knobs as the C layer lowers them: its refine's, and the seed ordering the parts. */
struct split_config_t {
    refine_config_t refine {};
    u64_t seed = 0;
};

/** The engine config a C call's refine options name. */
inline refine_config_t refine_config_of(usearch_refine_options_t options) noexcept {
    return {static_cast<candidates_count_t>(options.expansion_factor), options.memory_limit};
}

/** The variants behind a call's other handles, a split's receivers or a refinement's donors, read
 *  through the caller's own array, so a verb builds its one typed list straight from them. */
struct variants_view_t {
    void const* handles = nullptr;
    std::size_t count = 0;
    void* (*variant_at)(void const* handles, std::size_t position) noexcept = nullptr;

    std::size_t size() const noexcept { return count; }
    bool empty() const noexcept { return !count; }
    void* operator[](std::size_t position) const noexcept { return variant_at(handles, position); }
};

/** The engine config a C call's breadth and traversal name. */
inline search_config_t search_config_of(std::size_t expansion_factor, usearch_traversal_t traversal) noexcept {
    search_config_t config;
    config.expansion_factor = static_cast<candidates_count_t>(expansion_factor);
    config.traversal = traversal == usearch_traversal_scan_k ? traversal_t::scan_k : traversal_t::walk_k;
    return config;
}

/** Where an arena lives, and where the walks it serves compute: RAM, or one device's VRAM, the
 *  only residencies an arena takes. A RAM shard's memory domain travels beside it, never in it. */
struct arena_residency_t {
    usearch_residency_t residency = usearch_residency_ram_k;

    /** The vendor ordinal, on VRAM. */
    u32_t device = 0;

    bool operator==(arena_residency_t const&) const noexcept = default;
};

/** The residency every host walk and host arena names, whatever domain its memory sits in. */
inline constexpr arena_residency_t ram_residency_k {};

inline constexpr arena_residency_t vram_residency(std::size_t device) noexcept {
    return {usearch_residency_vram_k, static_cast<u32_t>(device)};
}

/** Writes what a refinement @p refined counts into the caller's @p report, if it names one. */
template <typename refined_type_>
inline void write_refine_report(usearch_refine_report_t* report, refined_type_ const& refined) noexcept {
    if (!report) return;
    report->reselected = refined.reselected_nodes;
    report->changed = refined.changed_nodes;
    report->islands_found = refined.islands_found;
    report->islands_reconnected = refined.islands_reconnected;
    report->reachability = {refined.reachability.audited_nodes, refined.reachability.reachable_nodes,
                            refined.reachability.sweeps, refined.reachability.back_sweeps};
}

#pragma endregion Composites

#pragma region Type Lists

/** How many storage dtypes the dense variants cover, from the front of the ladder bf16, u1, i8,
 *  e4m3, f16, u8, e5m2, i4, u4, e3m2, e2m3. CMake passes it, so only builds that bypass CMake take
 *  this default. */
#if !defined(USEARCH_DENSE_DTYPES)
#define USEARCH_DENSE_DTYPES 4
#endif
#if USEARCH_DENSE_DTYPES < 1 || USEARCH_DENSE_DTYPES > 11
#error "USEARCH_DENSE_DTYPES must be an integer from 1 to 11"
#endif

/** Types in order, the variants one basic class unit compiles in one residency. */
template <typename... types_>
struct type_list {
    static constexpr std::size_t size_k = sizeof...(types_);
};

/** Storage dtypes in ladder order, the one list every dense class unit folds its variants over. */
template <nk_dtype_t... dtypes_>
struct dtype_list {
    static constexpr std::size_t size_k = sizeof...(dtypes_);
    static constexpr std::array<nk_dtype_t, sizeof...(dtypes_)> dtypes_k {dtypes_...};

    /** The index of @p dtype in the list, or @c size_k when it is absent. */
    static constexpr std::size_t index_of(nk_dtype_t dtype) noexcept {
        std::size_t position = 0;
        while (position != size_k && dtypes_k[position] != dtype) ++position;
        return position;
    }
};

/** Takes dtypes off the front of a pack into @p taken_type_ until @p count_ are taken. */
template <std::size_t count_, typename taken_type_, nk_dtype_t... rest_>
struct first_dtypes_of {
    using type = taken_type_;
};
template <std::size_t count_, nk_dtype_t... taken_, nk_dtype_t next_, nk_dtype_t... rest_>
    requires(count_ != 0)
struct first_dtypes_of<count_, dtype_list<taken_...>, next_, rest_...> {
    using type = typename first_dtypes_of<count_ - 1, dtype_list<taken_..., next_>, rest_...>::type;
};

/** The leading @p count_ dtypes of the pack, as a @c dtype_list. */
template <std::size_t count_, nk_dtype_t... dtypes_>
using first_dtypes = typename first_dtypes_of<count_, dtype_list<>, dtypes_...>::type;

/** The dense storage dtypes this build compiles variants for. */
using dense_dtypes_t = first_dtypes<USEARCH_DENSE_DTYPES, nk_bf16_k, nk_u1_k, nk_i8_k, nk_e4m3_k, nk_f16_k, nk_u8_k,
                                    nk_e5m2_k, nk_i4_k, nk_u4_k, nk_e3m2_k, nk_e2m3_k>;

#pragma endregion Type Lists

#pragma region CPU Classes

/** The CPU families USearch ships tuned variants for - our own vocabulary, never an
 *  alias of a sister library's feature bits. Each class bundles the compute features,
 *  wait instruction, and atomics its silicon generation guarantees; explicit helpers
 *  lower a class onto NumKong, ForkUnion & StringZilla capability masks where those
 *  libraries are linked. */
enum class cpu_class_t : u32_t {

    /** No class detected yet, or none of the classes below. */
    unknown_k = 0,

    // Portable floor

    /** The portable floor: no SIMD assumptions, standard yields. */
    serial_k,

    // x86 classes

    /** Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    intel_haswell_k,

    /** Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI & its INT8 / IFMA / NE-CONVERT
     *  successors without AVX-512, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_sierra_k,

    /** Skylake-class x86: the AVX-512 foundation tier. */
    intel_skylake_k,

    /** Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    intel_icelake_k,

    /** Genoa-class x86 - Zen 4 and newer: native @c bf16 dot products over AVX-512. */
    amd_genoa_k,

    /** Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 &
     *  AMX-INT8 tiles. */
    intel_sapphire_k,

    /** Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles. */
    intel_granite_k,

    /** Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions and
     *  the Sierra vector bundle, @c TPAUSE waits, @c CMPCCXADD conditional adds. */
    intel_diamond_k,

    // Arm classes

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; pre-Armv8.6 phones fall
     *  here too. */
    aws_g2_k,

    /** Armv8.6+ NEON: dot products, @c bf16 & @c f16 fused multiplies. Named for Apple M2;
     *  Cortex-A710 & X2 - Snapdragon 8 Gen 1, AmpereOne - carry the same. */
    apple_m2_k,

    /** Apple M4-class Arm: the @c apple_m2_k bundle plus the SME outer-product tiles,
     *  @c BMOPA among them, without plain SVE. */
    apple_m4_k,

    /** Graviton5-class Arm - Neoverse V3, Armv9.2: the @c apple_m2_k bundle with waits that
     *  monitor through @c WFE instead of spinning; no build compiles it yet. */
    aws_g5_k,

    /** Vera-class Arm - NVIDIA's Armv9.2+ cores with @c FEAT_FP8 dot products. */
    nvidia_vera_k,

    // RISC-V classes

    /** Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23: the vector extension with
     *  @c Zvbb, @c Zawrs monitored waits, @c Zacas single-instruction compare-and-swap. */
    alibaba_c930_k,

    // WebAssembly classes

    /** WebAssembly every engine loads: 32-bit, one caller-inclusive thread, the fixed-width
     *  @c simd128 tier, so it runs on WebKit and in pages served without cross-origin isolation. */
    wasm32_v128_k,

    /** WebAssembly with relaxed SIMD, a shared memory for Web Worker pools and 64-bit addressing,
     *  bundled because every engine that ships one ships the other two. */
    wasm64_extended_k,

    // Monoversioned builds

    /** The one class of a monoversioned build: every capability @c USEARCH_MARCH implies or a
     *  preset vouches for, as the libraries compiled them. */
    monoversioned_k,
};

/** The GPU families USearch ships tuned variants for, the device twin of @c cpu_class_t: one
 *  class per generation of kernel kits, each compiled for its own architecture alone. A class
 *  implies its vendor, and an unknown newer device takes the newest class it satisfies within
 *  that vendor's classes. */
enum class gpu_class_t : u32_t {

    /** No class detected yet, or none of the classes below. */
    unknown_k = 0,

    // NVIDIA classes

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30. */
    nvidia_ampere_k,

    /** Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions. */
    nvidia_ada_k,

    /** Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    nvidia_hopper_k,

    /** Blackwell-class NVIDIA - B200, B300 & GB300: TMA @c gather4 scattered fetches
     *  and @c tcgen05 tensor-core GEMMs. */
    nvidia_blackwell_k,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark: Blackwell's conversions
     *  without @c tcgen05. */
    nvidia_blackwellrtx_k,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    nvidia_rubin_k,

    // AMD classes

    /** CDNA3 AMD - MI300X & MI325X: gfx942, 64-wide waves, FNUZ fp8. */
    amd_cdna3_k,

    /** CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8. */
    amd_cdna4_k,

    /** CDNA5 AMD - MI400: gfx1250, 32-wide waves. */
    amd_cdna5_k,

    // Apple classes

    /** Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2. */
    apple_metal7_k,

    /** Apple9-family GPUs - M3 & M4. */
    apple_metal9_k,

    /** Apple10-family GPUs - M5: the Neural Accelerator matrix units. */
    apple_metal10_k,
};

/** Lowers one CPU class onto NumKong's compute mask - cumulative within a platform, so the chains
 *  pick their best row per dtype under it. */
constexpr nk_capability_t numkong_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_haswell_k: return nk_cap_serial_k | nk_cap_haswell_k;
    case cpu_class_t::intel_sierra_k:
        return numkong_capabilities_of(cpu_class_t::intel_haswell_k) | nk_cap_alder_k | nk_cap_sierra_k;
    case cpu_class_t::intel_skylake_k: return numkong_capabilities_of(cpu_class_t::intel_haswell_k) | nk_cap_skylake_k;
    case cpu_class_t::intel_icelake_k: return numkong_capabilities_of(cpu_class_t::intel_skylake_k) | nk_cap_icelake_k;
    case cpu_class_t::amd_genoa_k: return numkong_capabilities_of(cpu_class_t::intel_icelake_k) | nk_cap_genoa_k;
    case cpu_class_t::intel_sapphire_k:
        return numkong_capabilities_of(cpu_class_t::amd_genoa_k) | nk_cap_sapphire_k | nk_cap_sapphireamx_k;
    case cpu_class_t::intel_granite_k:
        return numkong_capabilities_of(cpu_class_t::intel_sapphire_k) | nk_cap_graniteamx_k;
    case cpu_class_t::intel_diamond_k:
        return numkong_capabilities_of(cpu_class_t::intel_granite_k) | nk_cap_alder_k | nk_cap_sierra_k |
               nk_cap_diamond_k;
    case cpu_class_t::aws_g2_k: return nk_cap_serial_k | nk_cap_neon_k;
    case cpu_class_t::apple_m2_k:
    case cpu_class_t::aws_g5_k:
        return numkong_capabilities_of(cpu_class_t::aws_g2_k) | nk_cap_neonsdot_k | nk_cap_neonbfdot_k |
               nk_cap_neonfhm_k;
    case cpu_class_t::apple_m4_k:
        return numkong_capabilities_of(cpu_class_t::apple_m2_k) | nk_cap_sme_k | nk_cap_smebi32_k;
    case cpu_class_t::nvidia_vera_k: return numkong_capabilities_of(cpu_class_t::apple_m2_k) | nk_cap_neonfp8_k;
    case cpu_class_t::monoversioned_k: return nk_cpu_capabilities_compiled_();
    case cpu_class_t::alibaba_c930_k: return nk_cap_serial_k | nk_cap_rvv_k | nk_cap_rvvbb_k;
    case cpu_class_t::wasm32_v128_k: return nk_cap_serial_k | nk_cap_v128_k;
    case cpu_class_t::wasm64_extended_k:
        return numkong_capabilities_of(cpu_class_t::wasm32_v128_k) | nk_cap_v128relaxed_k;
    default: return nk_cap_serial_k;
    }
}

/** The class a refused rung degrades to - the next one down its own lineage, which is the class an
 *  variant must be distinct from to earn instantiation. On x86 the AVX-512 P-cores and the AVX-VNNI
 *  E-cores share the Haswell floor. */
constexpr cpu_class_t parent_class_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_diamond_k: return cpu_class_t::intel_granite_k;
    case cpu_class_t::intel_granite_k: return cpu_class_t::intel_sapphire_k;
    case cpu_class_t::intel_sapphire_k: return cpu_class_t::amd_genoa_k;
    case cpu_class_t::amd_genoa_k: return cpu_class_t::intel_icelake_k;
    case cpu_class_t::intel_icelake_k: return cpu_class_t::intel_skylake_k;
    case cpu_class_t::intel_skylake_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::intel_sierra_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::nvidia_vera_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::aws_g5_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m4_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m2_k: return cpu_class_t::aws_g2_k;
    case cpu_class_t::wasm64_extended_k: return cpu_class_t::wasm32_v128_k;
    default: return cpu_class_t::serial_k;
    }
}

/** Lowers one CPU class onto StringZilla's capability bits: each class spells every backend its
 *  kits may pick, and a class adding none spells its parent's. */
constexpr sz_capability_t stringzilla_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::serial_k: return sz_cap_serial_k;
    case cpu_class_t::intel_haswell_k: return sz_cap_serial_k | sz_cap_haswell_k | sz_cap_westmere_k;
    case cpu_class_t::intel_skylake_k:
        return stringzilla_capabilities_of(cpu_class_t::intel_haswell_k) | sz_cap_skylake_k;
    case cpu_class_t::intel_icelake_k:
        return stringzilla_capabilities_of(cpu_class_t::intel_skylake_k) | sz_cap_icelake_k | sz_cap_goldmont_k;
    case cpu_class_t::intel_sierra_k:
        return stringzilla_capabilities_of(cpu_class_t::intel_haswell_k) | sz_cap_goldmont_k;
    case cpu_class_t::aws_g2_k:
    case cpu_class_t::apple_m2_k: return sz_cap_serial_k | sz_cap_neon_k | sz_cap_neonaes_k | sz_cap_neonsha_k;
    case cpu_class_t::monoversioned_k: return sz_capabilities_compiled_cpu_mask();
    default: return stringzilla_capabilities_of(parent_class_of(cpu_class));
    }
}

/** The wait instruction one CPU class pairs: x86 spins PAUSE, the WAITPKG Xeons and the monitored
 *  Arm & RISC-V classes arm TPAUSE, WFE & WRS, WebAssembly threads sleep in @c memory.atomic.wait,
 *  everything else yields - as does any class whose @c FORKUNION_TARGET_<BIT> is off in the probe
 *  list. WFE sleeps on uncapped waits and yields on capped ones, so it needs no bit to confirm; the
 *  timed WFET only a monoversioned build reaches, where a preset vouches for it. */
template <cpu_class_t cpu_class_>
struct waiter_for_cpu_class {
    using type = fu::standard_yield_t;
};
#if FORKUNION_TARGET_X86_PAUSE
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_haswell_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_skylake_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_icelake_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::amd_genoa_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_sapphire_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_granite_k> {
    using type = fu::x86_pause_t;
};
#endif
#if FORKUNION_TARGET_X86_TPAUSE
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_sierra_k> {
    using type = fu::x86_tpause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_diamond_k> {
    using type = fu::x86_tpause_t;
};
#endif
#if FORKUNION_TARGET_ARM64_YIELD
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::apple_m2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::apple_m4_k> {
    using type = fu::arm64_yield_t;
};
#endif
#if FORKUNION_TARGET_ARM64_WFE
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g5_k> {
    using type = fu::arm64_wfe_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    using type = fu::arm64_wfe_t;
};
#endif
#if FORKUNION_TARGET_RISCV64_WRS
template <>
struct waiter_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    using type = fu::riscv64_wrs_t;
};
#endif
#if FORKUNION_TARGET_WASM_WAIT
template <>
struct waiter_for_cpu_class<cpu_class_t::wasm32_v128_k> {
    using type = fu::wasm_wait_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::wasm64_extended_k> {
    using type = fu::wasm_wait_t;
};
#endif

/** The waiter a CPU class pairs, spelled short for the variants' explicit instantiations. */
template <cpu_class_t cpu_class_>
using waiter_of = typename waiter_for_cpu_class<cpu_class_>::type;

/** The atomic reference one CPU class pairs: single-instruction LSE read-modify-writes under RCpc
 *  acquiring loads on the Armv8.3+ classes, conditional adds on the CMPCCXADD Xeons, @c amocas on
 *  Zacas RISC-V, and on WebAssembly threads a @c memory.atomic.notify after every releasing write,
 *  so the sleeping waiter wakes on it - the floors, the older classes, and any class whose
 *  @c FORKUNION_TARGET_<BIT> is off in the probe list take the standard one. */
template <cpu_class_t cpu_class_>
struct atomic_ref_for_cpu_class {
    template <typename value_type_>
    using type = fu::standard_atomic_ref<value_type_>;
};
#if FORKUNION_TARGET_X86_CMPCCXADD
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::intel_sierra_k> {
    template <typename value_type_>
    using type = fu::x86_cmpccxadd_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::intel_diamond_k> {
    template <typename value_type_>
    using type = fu::x86_cmpccxadd_atomic_ref<value_type_>;
};
#endif
#if FORKUNION_TARGET_ARM64_RCPC
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m2_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m4_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::aws_g5_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    template <typename value_type_>
    using type = fu::arm64_rcpc_atomic_ref<value_type_>;
};
#elif FORKUNION_TARGET_ARM64_LSE
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m2_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::apple_m4_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::aws_g5_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    template <typename value_type_>
    using type = fu::arm64_lse_atomic_ref<value_type_>;
};
#endif
#if FORKUNION_TARGET_RISCV64_ZACAS
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    template <typename value_type_>
    using type = fu::riscv64_zacas_atomic_ref<value_type_>;
};
#endif
#if FORKUNION_TARGET_WASM_WAIT
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::wasm32_v128_k> {
    template <typename value_type_>
    using type = fu::wasm_atomic_ref<value_type_>;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::wasm64_extended_k> {
    template <typename value_type_>
    using type = fu::wasm_atomic_ref<value_type_>;
};
#endif
#if USEARCH_TARGET_MONOVERSIONED
template <>
struct waiter_for_cpu_class<cpu_class_t::monoversioned_k> {
    using type = fu::preferred_yield_t;
};
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::monoversioned_k> {
    template <typename value_type_>
    using type = fu::preferred_atomic_ref<value_type_>;
};
#endif

/** The cache hints one CPU class pairs: the x86 classes demote with CLDEMOTE and the Arm classes
 *  prefetch with PRFM, all in hint or reserved-NOP space, so no class gains a bit to confirm. */
template <cpu_class_t cpu_class_>
struct cache_hints_for_cpu_class {
    using type = fu::standard_cache_hints_t;
};
#if FORKUNION_TARGET_X86_CLDEMOTE
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_haswell_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_skylake_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_icelake_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::amd_genoa_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_sapphire_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_granite_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_sierra_k> {
    using type = fu::x86_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::intel_diamond_k> {
    using type = fu::x86_cache_hints_t;
};
#endif
#if FORKUNION_TARGET_ARM64_PRFM
template <>
struct cache_hints_for_cpu_class<cpu_class_t::aws_g2_k> {
    using type = fu::arm64_prefetch_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::apple_m2_k> {
    using type = fu::arm64_prefetch_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::apple_m4_k> {
    using type = fu::arm64_prefetch_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::aws_g5_k> {
    using type = fu::arm64_prefetch_cache_hints_t;
};
template <>
struct cache_hints_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    using type = fu::arm64_prefetch_cache_hints_t;
};
#endif
#if USEARCH_TARGET_MONOVERSIONED
template <>
struct cache_hints_for_cpu_class<cpu_class_t::monoversioned_k> {
    using type = fu::preferred_cache_hints_t;
};
#endif

/** The line every engine's prefetcher steps by: this machine's cache line, or
 *  @c default_alignment_k where no platform source names one. */
inline std::size_t machine_cache_line_bytes() noexcept {
    std::size_t const machine_line_bytes = fu::cache_line_bytes();
    return machine_line_bytes ? machine_line_bytes : default_alignment_k;
}

/** The spin hints every core of an architecture carries - PAUSE, YIELD, Zihintpause: the runtime
 *  confirms nothing about them, and a class differing by one alone earns no rung. */
inline constexpr fu::capabilities_t baseline_hints_k = fu::capability_x86_pause_k | fu::capability_arm64_yield_k |
                                                       fu::capability_riscv64_pause_k;

/** Lowers one CPU class onto ForkUnion's mask - the bits the runtime must confirm before the
 *  class's waiter and atomic reference are trusted, read off the two types themselves. */
template <cpu_class_t cpu_class_>
constexpr fu::capabilities_t forkunion_capabilities_of() noexcept {
    constexpr fu::capabilities_t waits_k = waiter_of<cpu_class_>::capability_k;
    constexpr fu::capabilities_t atomics_k = atomic_ref_for_cpu_class<cpu_class_>::template type<u32_t>::capabilities_k;
    return static_cast<fu::capabilities_t>((waits_k | atomics_k) & ~baseline_hints_k);
}

/** Whether the build compiled the unit of @p cpu_class, as CMake decided and told the dispatch
 *  units through @c USEARCH_TARGET_<CLASS>; a ladder never picks a class without one. */
constexpr bool cpu_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if USEARCH_TARGET_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if USEARCH_TARGET_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if USEARCH_TARGET_INTEL_SIERRA
    case cpu_class_t::intel_sierra_k: return true;
#endif
#if USEARCH_TARGET_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if USEARCH_TARGET_INTEL_ICELAKE
    case cpu_class_t::intel_icelake_k: return true;
#endif
#if USEARCH_TARGET_AMD_GENOA
    case cpu_class_t::amd_genoa_k: return true;
#endif
#if USEARCH_TARGET_INTEL_SAPPHIRE
    case cpu_class_t::intel_sapphire_k: return true;
#endif
#if USEARCH_TARGET_INTEL_GRANITE
    case cpu_class_t::intel_granite_k: return true;
#endif
#if USEARCH_TARGET_INTEL_DIAMOND
    case cpu_class_t::intel_diamond_k: return true;
#endif
#if USEARCH_TARGET_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if USEARCH_TARGET_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if USEARCH_TARGET_APPLE_M4
    case cpu_class_t::apple_m4_k: return true;
#endif
#if USEARCH_TARGET_AWS_G5
    case cpu_class_t::aws_g5_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_VERA
    case cpu_class_t::nvidia_vera_k: return true;
#endif
#if USEARCH_TARGET_ALIBABA_C930
    case cpu_class_t::alibaba_c930_k: return true;
#endif
#if USEARCH_TARGET_WASM32_V128
    case cpu_class_t::wasm32_v128_k: return true;
#endif
#if USEARCH_TARGET_WASM64_EXTENDED
    case cpu_class_t::wasm64_extended_k: return true;
#endif
#if USEARCH_TARGET_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

/** ForkUnion's mask for a class named at run time, read off the template per class. */
constexpr fu::capabilities_t forkunion_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::intel_haswell_k: return forkunion_capabilities_of<cpu_class_t::intel_haswell_k>();
    case cpu_class_t::intel_sierra_k: return forkunion_capabilities_of<cpu_class_t::intel_sierra_k>();
    case cpu_class_t::intel_skylake_k: return forkunion_capabilities_of<cpu_class_t::intel_skylake_k>();
    case cpu_class_t::intel_icelake_k: return forkunion_capabilities_of<cpu_class_t::intel_icelake_k>();
    case cpu_class_t::amd_genoa_k: return forkunion_capabilities_of<cpu_class_t::amd_genoa_k>();
    case cpu_class_t::intel_sapphire_k: return forkunion_capabilities_of<cpu_class_t::intel_sapphire_k>();
    case cpu_class_t::intel_granite_k: return forkunion_capabilities_of<cpu_class_t::intel_granite_k>();
    case cpu_class_t::intel_diamond_k: return forkunion_capabilities_of<cpu_class_t::intel_diamond_k>();
    case cpu_class_t::aws_g2_k: return forkunion_capabilities_of<cpu_class_t::aws_g2_k>();
    case cpu_class_t::apple_m2_k: return forkunion_capabilities_of<cpu_class_t::apple_m2_k>();
    case cpu_class_t::apple_m4_k: return forkunion_capabilities_of<cpu_class_t::apple_m4_k>();
    case cpu_class_t::aws_g5_k: return forkunion_capabilities_of<cpu_class_t::aws_g5_k>();
    case cpu_class_t::nvidia_vera_k: return forkunion_capabilities_of<cpu_class_t::nvidia_vera_k>();
    case cpu_class_t::alibaba_c930_k: return forkunion_capabilities_of<cpu_class_t::alibaba_c930_k>();
    case cpu_class_t::wasm32_v128_k: return forkunion_capabilities_of<cpu_class_t::wasm32_v128_k>();
    case cpu_class_t::wasm64_extended_k: return forkunion_capabilities_of<cpu_class_t::wasm64_extended_k>();
    case cpu_class_t::monoversioned_k: return forkunion_capabilities_of<cpu_class_t::monoversioned_k>();
    default: return forkunion_capabilities_of<cpu_class_t::serial_k>();
    }
}

/** The C enum's class as the engines name it; @c usearch_cpu_class_unknown_k derives nothing, so
 *  it names the serial floor. */
constexpr cpu_class_t cpu_class_of(usearch_cpu_class_t cpu_class) noexcept {
    if (cpu_class == usearch_cpu_class_unknown_k || cpu_class > usearch_cpu_class_monoversioned_k)
        return cpu_class_t::serial_k;
    return static_cast<cpu_class_t>(cpu_class);
}

/** What one handle may use: the masks this machine runs when nothing is pinned, the pinned class's
 *  masks when it runs them. */
struct cpu_ceiling_t {
    nk_capability_t compute = nk_cap_serial_k;
    sz_capability_t strings = sz_cap_serial_k;
    fu::capabilities_t sync = fu::capabilities_unknown_k;
};

/** Whether @p rung fits under @p ceiling: its kits and waiters need nothing the ceiling lacks. */
constexpr bool cpu_rung_fits(cpu_class_t rung, cpu_ceiling_t ceiling) noexcept {
    return (numkong_capabilities_of(rung) & ~ceiling.compute) == 0 &&
           (stringzilla_capabilities_of(rung) & ~ceiling.strings) == 0 &&
           (forkunion_capabilities_of(rung) & ~ceiling.sync) == 0;
}

/** The allow-mask a pool spawns under @p ceiling: every thread and memory facility, and only the
 *  instruction bits the ceiling holds. */
constexpr fu_capabilities_t forkunion_allowed_of(cpu_ceiling_t ceiling) noexcept {
    return static_cast<fu_capabilities_t>((fu_capabilities_all_k & ~fu_capability_any_instruction_k) | ceiling.sync |
                                          baseline_hints_k);
}

/** CPU classes newest first, the order a ladder tries them in. */
template <cpu_class_t... classes_>
struct cpu_ladder {};

/** GPU classes of one vendor newest first, the order its ladder tries them in. */
template <gpu_class_t... classes_>
struct gpu_ladder {};

/** This architecture's ladder, the serial floor last, or the one class of a monoversioned build; a
 *  rung the build left out is skipped by the walk, never by the list. */
#if USEARCH_TARGET_MONOVERSIONED
using cpu_ladder_t = cpu_ladder<cpu_class_t::monoversioned_k>;
#elif NUMKONG_ARCH_X8664_
using cpu_ladder_t =
    cpu_ladder<cpu_class_t::intel_diamond_k, cpu_class_t::intel_granite_k, cpu_class_t::intel_sapphire_k,
               cpu_class_t::amd_genoa_k, cpu_class_t::intel_icelake_k, cpu_class_t::intel_skylake_k,
               cpu_class_t::intel_sierra_k, cpu_class_t::intel_haswell_k, cpu_class_t::serial_k>;
#elif NUMKONG_ARCH_ARM64_
using cpu_ladder_t = cpu_ladder<cpu_class_t::nvidia_vera_k, cpu_class_t::apple_m4_k, cpu_class_t::aws_g5_k,
                                cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k, cpu_class_t::serial_k>;
#elif NUMKONG_ARCH_RISCV64_
using cpu_ladder_t = cpu_ladder<cpu_class_t::alibaba_c930_k, cpu_class_t::serial_k>;
#elif NUMKONG_ARCH_WASM_
using cpu_ladder_t = cpu_ladder<cpu_class_t::wasm64_extended_k, cpu_class_t::wasm32_v128_k, cpu_class_t::serial_k>;
#else
using cpu_ladder_t = cpu_ladder<cpu_class_t::serial_k>;
#endif

/** The ceiling of a handle pinned to @p pinned, the one caller of NumKong's, StringZilla's and
 *  ForkUnion's detection; @c unsupported_k when this machine lacks the pinned class's masks, or
 *  when the build compiled no class under the ceiling, as a monoversioned build on an older CPU or
 *  under a pin below its one class. */
inline expected<cpu_ceiling_t> pick_cpu_ceiling(usearch_cpu_class_t pinned) noexcept {
    cpu_ceiling_t ceiling;
    if (nk_cpu_capabilities_enabled(&ceiling.compute) != nk_success_k) ceiling.compute = nk_cap_serial_k;
    if (sz_capabilities_enabled_cpu(&ceiling.strings) != sz_success_k) ceiling.strings = sz_cap_serial_k;
    ceiling.sync = fu::cpu_capabilities();
    if (pinned > usearch_cpu_class_monoversioned_k) return {ceiling, status_t::unsupported_k};
    bool runs = true;
    if (pinned != usearch_cpu_class_unknown_k) {
        cpu_class_t const pinned_class = cpu_class_of(pinned);
        cpu_ceiling_t const wanted {numkong_capabilities_of(pinned_class), stringzilla_capabilities_of(pinned_class),
                                    forkunion_capabilities_of(pinned_class)};
        runs = (ceiling.compute & wanted.compute) == wanted.compute &&
               (ceiling.strings & wanted.strings) == wanted.strings && (ceiling.sync & wanted.sync) == wanted.sync;
        ceiling = wanted;
    }
    bool const compiled = []<cpu_class_t... classes_>(cpu_ladder<classes_...>, cpu_ceiling_t under) noexcept {
        return ((cpu_class_compiled(classes_) && cpu_rung_fits(classes_, under)) || ...);
    }(cpu_ladder_t {}, ceiling);
    return {ceiling, runs && compiled ? status_t::success_k : status_t::unsupported_k};
}

/** One class's outcome row: its reference over the status, the claim and the named-call counts,
 *  its waiter between their loads. */
template <cpu_class_t cpu_class_>
outcome_verbs_t outcome_verbs_of() noexcept {
    using status_ref_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<usearch_status_t>;
    using status_reader_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<usearch_status_t const>;
    using claim_ref_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<call_claim_t>;
    using claim_reader_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<call_claim_t const>;
    using count_ref_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<u32_t>;
    using count_reader_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<u32_t const>;
    using wait_t = waiter_of<cpu_class_>;
    return {
        [](usearch_outcome_t* outcome, usearch_status_t status, std::memory_order order) noexcept {
            status_ref_t(outcome->status).store(status, order);
        },
        [](usearch_outcome_t const* outcome) noexcept {
            return status_reader_t(outcome->status).load(std::memory_order_acquire);
        },
        [](usearch_outcome_t const* outcome) noexcept {
            status_reader_t const status {outcome->status};
            usearch_status_t seen = status.load(std::memory_order_acquire);
            for (; seen == usearch_status_deferred_k; seen = status.load(std::memory_order_acquire))
                wait_t {}(&outcome->status, seen, 0u, fu::wait_capped_t {});
            return seen;
        },
        [](call_claim_t* claim) noexcept {
            return claim_ref_t(*claim).exchange(call_claim_t::held_k, std::memory_order_acquire);
        },
        [](call_claim_t* claim) noexcept {
            claim_ref_t(*claim).store(call_claim_t::free_k, std::memory_order_release);
        },
        [](call_claim_t const* claim) noexcept {
            claim_reader_t const held {*claim};
            for (call_claim_t observed; (observed = held.load(std::memory_order_acquire)) == call_claim_t::held_k;)
                wait_t {}(claim, observed, 0u, fu::wait_capped_t {});
        },
        [](u32_t* count) noexcept { count_ref_t(*count).store_add(1, std::memory_order_relaxed); },
        [](u32_t* count) noexcept { count_ref_t(*count).store_sub(1, std::memory_order_release); },
        [](u32_t const* count) noexcept {
            count_reader_t const named {*count};
            for (u32_t observed; (observed = named.load(std::memory_order_acquire)) != 0;)
                wait_t {}(count, observed, 0u, fu::wait_capped_t {});
        },
    };
}

inline outcome_verbs_t const& host_outcome_verbs() noexcept {
    static outcome_verbs_t const picked = []<cpu_class_t... classes_>(cpu_ladder<classes_...>) noexcept {
        cpu_ceiling_t const ceiling = pick_cpu_ceiling(usearch_cpu_class_unknown_k).value;
        outcome_verbs_t row = outcome_verbs_of<cpu_class_t::serial_k>();
        bool found = false;
        auto rung = [&]<cpu_class_t class_>() noexcept {
            if (!found && cpu_class_compiled(class_) && cpu_rung_fits(class_, ceiling))
                row = outcome_verbs_of<class_>(), found = true;
        };
        (rung.template operator()<classes_>(), ...);
        return row;
    }(cpu_ladder_t {});
    return picked;
}

/** Whether the dense NVMe variants compiled a unit for @p cpu_class, as their row in
 *  @c src/CMakeLists.txt told the dispatch unit through @c USEARCH_TARGET_NVME_<CLASS>: each kept
 *  class compiles as itself or the nearest ancestor the row lists. A storage walk shares a few
 *  counters and never waits, so the classes advancing by their waiter or reference alone take none,
 *  and a WebAssembly module reads no storage. */
constexpr bool nvme_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if USEARCH_TARGET_NVME_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_SIERRA
    case cpu_class_t::intel_sierra_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_ICELAKE
    case cpu_class_t::intel_icelake_k: return true;
#endif
#if USEARCH_TARGET_NVME_AMD_GENOA
    case cpu_class_t::amd_genoa_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_SAPPHIRE
    case cpu_class_t::intel_sapphire_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_GRANITE
    case cpu_class_t::intel_granite_k: return true;
#endif
#if USEARCH_TARGET_NVME_INTEL_DIAMOND
    case cpu_class_t::intel_diamond_k: return true;
#endif
#if USEARCH_TARGET_NVME_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if USEARCH_TARGET_NVME_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if USEARCH_TARGET_NVME_APPLE_M4
    case cpu_class_t::apple_m4_k: return true;
#endif
#if USEARCH_TARGET_NVME_NVIDIA_VERA
    case cpu_class_t::nvidia_vera_k: return true;
#endif
#if USEARCH_TARGET_NVME_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

/** Whether the basic modalities compiled a unit for @p cpu_class, as their rows in
 *  @c src/CMakeLists.txt told each dispatch unit through @c USEARCH_TARGET_BASIC_<CLASS>: each kept
 *  class compiles as itself or the nearest ancestor where a StringZilla or NumKong kit they inline
 *  advances. */
constexpr bool basic_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
#if USEARCH_TARGET_BASIC_SERIAL
    case cpu_class_t::serial_k: return true;
#endif
#if USEARCH_TARGET_BASIC_INTEL_HASWELL
    case cpu_class_t::intel_haswell_k: return true;
#endif
#if USEARCH_TARGET_BASIC_INTEL_SKYLAKE
    case cpu_class_t::intel_skylake_k: return true;
#endif
#if USEARCH_TARGET_BASIC_AWS_G2
    case cpu_class_t::aws_g2_k: return true;
#endif
#if USEARCH_TARGET_BASIC_APPLE_M2
    case cpu_class_t::apple_m2_k: return true;
#endif
#if USEARCH_TARGET_BASIC_ALIBABA_C930
    case cpu_class_t::alibaba_c930_k: return true;
#endif
#if USEARCH_TARGET_BASIC_WASM64_EXTENDED
    case cpu_class_t::wasm64_extended_k: return true;
#endif
#if USEARCH_TARGET_BASIC_MONOVERSIONED
    case cpu_class_t::monoversioned_k: return true;
#endif
    default: return false;
    }
}

#pragma endregion CPU Classes

#pragma region Residencies

/** Every I/O engine this build admits, yielding between polls, since one NVMe executor serves every
 *  CPU class. */
using default_io_engine_t = default_io_engine<fu::standard_yield_t>;

/** The executor over every engine this build admits, each thread's transport picked at spawn. */
using nvme_executor_t = nvme_executor<default_io_engine_t>;

/** Residency discrimination off the alternative's own executor type - no second tag exists to
 *  disagree with the variant. */
template <typename multi_type_>
inline constexpr bool is_ram_index = requires {
    requires std::is_same_v<typename multi_type_::executor_t, forkunion_executor_t>;
};

/** Whether an executor drives GPUs: every vendor's names the vendor it drives. */
template <typename executor_type_>
inline constexpr bool is_gpu_executor = requires { executor_type_::vendor_k; };

template <typename multi_type_>
inline constexpr bool is_vram_index = requires { requires is_gpu_executor<typename multi_type_::executor_t>; };

template <typename multi_type_>
inline constexpr bool is_nvme_index = requires {
    requires std::is_same_v<typename multi_type_::executor_t, nvme_executor_t>;
};

/** Where a variant runs and which executor it takes, the dense and basic modalities alike. */
enum class vendored_residency_t : u32_t {
    ram_k,
    nvme_k,
    vram_nvidia_k,
    vram_amd_k,
    vram_apple_k,
};

/** The executor each residency's verbs take. */
template <vendored_residency_t residency_>
struct vendored_residency_executor;
template <>
struct vendored_residency_executor<vendored_residency_t::ram_k> {
    using type = forkunion_executor_t;
};
template <>
struct vendored_residency_executor<vendored_residency_t::nvme_k> {
    using type = nvme_executor_t;
};
template <>
struct vendored_residency_executor<vendored_residency_t::vram_nvidia_k> {
    using type = cuda_executor_t;
};
template <>
struct vendored_residency_executor<vendored_residency_t::vram_amd_k> {
    using type = hip_executor_t;
};
template <>
struct vendored_residency_executor<vendored_residency_t::vram_apple_k> {
    using type = metal_executor_t;
};

/** The residency whose verbs take @p executor_type_. */
template <typename executor_type_>
inline constexpr vendored_residency_t vendored_residency_of =
    std::is_same_v<executor_type_, forkunion_executor_t> ? vendored_residency_t::ram_k
    : std::is_same_v<executor_type_, nvme_executor_t>    ? vendored_residency_t::nvme_k
    : std::is_same_v<executor_type_, cuda_executor_t>    ? vendored_residency_t::vram_nvidia_k
    : std::is_same_v<executor_type_, hip_executor_t>     ? vendored_residency_t::vram_amd_k
                                                         : vendored_residency_t::vram_apple_k;

/** The first compute capability one class's cubins run on; the class covers the rest of that major
 *  version up to the next row, the family its @c f or @c a target covers. */
struct nvidia_class_span_t {
    gpu_class_t gpu_class;
    cuda_arch_t first;
};

/** Every NVIDIA class and the GPUs its unit runs natively; a new family is a new row. */
inline constexpr nvidia_class_span_t nvidia_class_spans_k[] = {
    {gpu_class_t::nvidia_ampere_k, cuda_arch_ampere_k},
    {gpu_class_t::nvidia_ada_k, cuda_arch_ada_k},
    {gpu_class_t::nvidia_hopper_k, cuda_arch_hopper_k},
    {gpu_class_t::nvidia_blackwell_k, cuda_arch_blackwell_k},
    {gpu_class_t::nvidia_blackwellrtx_k, cuda_arch_blackwellrtx_k},
};

/** The class whose unit carries cubins for compute capability @p arch: the last row of its major
 *  version at or below it, else the Ampere unit, whose PTX JIT-compiles for any newer GPU, and none
 *  older than Ampere. */
constexpr gpu_class_t nvidia_class_of(cuda_arch_t arch) noexcept {
    if (arch < cuda_arch_ampere_k) return gpu_class_t::unknown_k;
    gpu_class_t picked = gpu_class_t::nvidia_ampere_k;
    for (nvidia_class_span_t const& span : nvidia_class_spans_k)
        if (span.first <= arch && span.first / 100 == arch / 100) picked = span.gpu_class;
    return picked;
}

/** The class whose kernels run on an AMD GPU of @c gfx version @p major. @p minor: 9.4 is MI300's
 *  @c gfx942, 9.5 is MI350's @c gfx950, and 12.5 is MI400's @c gfx1250 and @c gfx1251; any other
 *  has no code object to load. */
constexpr gpu_class_t amd_class_of(u32_t major, u32_t minor) noexcept {
    if (major == 9 && minor == 4) return gpu_class_t::amd_cdna3_k;
    if (major == 9 && minor == 5) return gpu_class_t::amd_cdna4_k;
    if (major == 12 && minor == 5) return gpu_class_t::amd_cdna5_k;
    return gpu_class_t::unknown_k;
}

/** The class whose unit runs on an Apple GPU of the newest family @p family: Apple7 and Apple8
 *  share the floor, and a GPU older than Apple7 runs none. */
constexpr gpu_class_t apple_class_of(u32_t family) noexcept {
    if (family >= 10) return gpu_class_t::apple_metal10_k;
    if (family == 9) return gpu_class_t::apple_metal9_k;
    if (family >= 7) return gpu_class_t::apple_metal7_k;
    return gpu_class_t::unknown_k;
}

/** The class whose unit runs on @p device, read off the architecture its driver reported; unknown
 *  where no unit does. */
constexpr gpu_class_t gpu_class_of(gpu_device_t const& device) noexcept {
    switch (device.vendor) {
    case gpu_vendor_t::nvidia_k:
        return nvidia_class_of(cuda_arch_of(device.architecture_major, device.architecture_minor));
    case gpu_vendor_t::amd_k: return amd_class_of(device.architecture_major, device.architecture_minor);
    case gpu_vendor_t::apple_k: return apple_class_of(device.architecture_major);
    default: return gpu_class_t::unknown_k;
    }
}

/** The GPU side of the process, built at its first VRAM call: every vendor's driver, and the
 *  devices some class runs on, NVIDIA first, then AMD, then Apple, a device's position here being
 *  its USearch ordinal. */
struct gpu_runtime_t {
    gpu_drivers_t drivers;
    gpu_device_t devices[3 * gpu_devices_capacity_k];
    std::size_t devices_count = 0;

    gpu_runtime_t() noexcept {
        std::size_t const reported = list_gpu_devices(drivers, devices);
        for (std::size_t position = 0; position != reported; ++position)
            if (gpu_class_of(devices[position]) != gpu_class_t::unknown_k) devices[devices_count++] = devices[position];
    }
};

/** The one runtime every VRAM call passes its drivers down from. */
inline gpu_runtime_t& gpu_runtime() noexcept {
    static gpu_runtime_t runtime;
    return runtime;
}

/** Every vendor's driver, loaded at the first VRAM call. */
inline gpu_drivers_t& gpu_drivers() noexcept { return gpu_runtime().drivers; }

/** Every GPU some class runs on, as the ABI numbers them. */
inline std::span<gpu_device_t const> gpu_devices() noexcept {
    gpu_runtime_t const& runtime = gpu_runtime();
    return {runtime.devices, runtime.devices_count};
}

/** The checkpoint in @p bytes and where it sits: on the NVIDIA or AMD device whose memory holds it,
 *  which no host read reaches, or on the host, as host pages and Apple's unified memory are. */
inline checkpoint_source_t checkpoint_source_of(bytes_view_t bytes) noexcept {
    gpu_drivers_t const& drivers = gpu_drivers();
    int const nvidia = cuda_device_of(drivers.cuda, bytes.data());
    return {bytes, nvidia >= 0 ? nvidia : hip_device_of(drivers.hip, bytes.data())};
}

/** Copies the NVIDIA or AMD device bytes @p source into the host's @p destination through the
 *  driver whose device holds them, returning once they arrived, as a checkpoint's headers cross. */
inline status_t copy_off_device(bytes_span_t destination, bytes_view_t source) noexcept {
    if (destination.size() != source.size()) return status_t::config_mismatch_k;
    gpu_drivers_t const& drivers = gpu_drivers();
    if (int const nvidia = cuda_device_of(drivers.cuda, source.data()); nvidia >= 0)
        return cuda_set_device(drivers.cuda, nvidia) == cuda_success_k &&
                       cuda_copy(drivers.cuda, destination.data(), source.data(), source.size()) == cuda_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    if (int const amd = hip_device_of(drivers.hip, source.data()); amd >= 0)
        return hip_set_device(drivers.hip, amd) == hip_success_k &&
                       drivers.hip.copy(destination.data(), source.data(), source.size(), hip_copy_device_to_host_k) ==
                           hip_success_k
                   ? status_t::success_k
                   : status_t::device_error_k;
    return status_t::device_error_k;
}

/** Lowers one GPU class onto NumKong's GPU capabilities. On NVIDIA, the Ampere & Ada bits
 *  accumulate, while Hopper, Blackwell & RTX Blackwell each run only their own kits, and Rubin
 *  rides the Blackwell kits until its own land. AMD and Apple generations accumulate over their
 *  vendor's floor. */
constexpr nk_capability_t numkong_capabilities_of(gpu_class_t gpu_class) noexcept {
    nk_capability_t const ampere = nk_cap_cuda_k | nk_cap_ampere_k, ada = ampere | nk_cap_ada_k;
    nk_capability_t const cdna3 = nk_cap_rocm_k | nk_cap_cdna3_k, cdna4 = cdna3 | nk_cap_cdna4_k;
    nk_capability_t const metal9 = nk_cap_metal_k | nk_cap_apple9_k;
    switch (gpu_class) {
    case gpu_class_t::nvidia_ada_k: return ada;
    case gpu_class_t::nvidia_hopper_k: return ada | nk_cap_hopper_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_rubin_k: return ada | nk_cap_blackwell_k;
    case gpu_class_t::nvidia_blackwellrtx_k: return ada | nk_cap_blackwellrtx_k;
    case gpu_class_t::amd_cdna3_k: return cdna3;
    case gpu_class_t::amd_cdna4_k: return cdna4;
    case gpu_class_t::amd_cdna5_k: return cdna4 | nk_cap_cdna5_k;
    case gpu_class_t::apple_metal7_k: return nk_cap_metal_k;
    case gpu_class_t::apple_metal9_k: return metal9;
    case gpu_class_t::apple_metal10_k: return metal9 | nk_cap_apple10_k;
    case gpu_class_t::nvidia_ampere_k: return ampere;
    case gpu_class_t::unknown_k: return nk_cap_serial_k;
    }
    return nk_cap_serial_k;
}

/** The ceiling an NVIDIA device pass instantiates: its class's, read off the one NumKong target the
 *  class compiles under. */
inline constexpr nk_capability_t cuda_class_capabilities_k = numkong_capabilities_of(
    NUMKONG_TARGET_BLACKWELLRTX ? gpu_class_t::nvidia_blackwellrtx_k
    : NUMKONG_TARGET_BLACKWELL  ? gpu_class_t::nvidia_blackwell_k
    : NUMKONG_TARGET_HOPPER     ? gpu_class_t::nvidia_hopper_k
    : NUMKONG_TARGET_ADA        ? gpu_class_t::nvidia_ada_k
                                : gpu_class_t::nvidia_ampere_k);

/** Devices @p vram names, one when it names none. */
inline std::size_t named_count(usearch_vram_options_t const& vram) noexcept {
    return vram.devices && vram.devices_count ? vram.devices_count : 1;
}

/** The USearch device ordinal shard @p shard of @p vram sits on: the named devices taken
 *  round-robin, device zero when it names none. */
inline std::size_t named_ordinal(usearch_vram_options_t const& vram, std::size_t shard) noexcept {
    return vram.devices && vram.devices_count ? static_cast<std::size_t>(vram.devices[shard % vram.devices_count]) : 0;
}

/** Whether a device of @p available capabilities runs every kit one GPU class lowers onto. */
constexpr bool numkong_admits(nk_capability_t available, gpu_class_t gpu_class) noexcept {
    nk_capability_t const wanted = numkong_capabilities_of(gpu_class);
    return (available & wanted) == wanted;
}

/** The capabilities a GPU ladder walks under: the devices' class @p device_class, narrowed to
 *  @p pinned when a caller names one; zero when the devices do not run the pinned class. */
constexpr nk_capability_t gpu_pin_capabilities(gpu_class_t device_class, usearch_gpu_class_t pinned) noexcept {
    nk_capability_t const device = numkong_capabilities_of(device_class);
    if (pinned == usearch_gpu_class_unknown_k) return device;
    if (pinned > usearch_gpu_class_apple_metal10_k) return 0;
    gpu_class_t const pinned_class = static_cast<gpu_class_t>(pinned);
    return numkong_admits(device, pinned_class) ? numkong_capabilities_of(pinned_class) : 0;
}

/** Whether the build compiled device code for @p gpu_class, as CMake decided and told the dispatch
 *  units through @c USEARCH_TARGET_<VENDOR>_<CLASS>; a ladder never picks a class without one. */
constexpr bool gpu_class_compiled(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
#if USEARCH_TARGET_NVIDIA_AMPERE
    case gpu_class_t::nvidia_ampere_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_ADA
    case gpu_class_t::nvidia_ada_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_HOPPER
    case gpu_class_t::nvidia_hopper_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_BLACKWELL
    case gpu_class_t::nvidia_blackwell_k: return true;
#endif
#if USEARCH_TARGET_NVIDIA_BLACKWELLRTX
    case gpu_class_t::nvidia_blackwellrtx_k: return true;
#endif
#if USEARCH_TARGET_AMD_CDNA3
    case gpu_class_t::amd_cdna3_k: return true;
#endif
#if USEARCH_TARGET_AMD_CDNA4
    case gpu_class_t::amd_cdna4_k: return true;
#endif
#if USEARCH_TARGET_AMD_CDNA5
    case gpu_class_t::amd_cdna5_k: return true;
#endif
#if USEARCH_TARGET_APPLE_METAL7
    case gpu_class_t::apple_metal7_k: return true;
#endif
#if USEARCH_TARGET_APPLE_METAL9
    case gpu_class_t::apple_metal9_k: return true;
#endif
#if USEARCH_TARGET_APPLE_METAL10
    case gpu_class_t::apple_metal10_k: return true;
#endif
    default: return false;
    }
}

/** Whether the basic modalities compiled device code for @p gpu_class: their Ampere rung is the
 *  vendor-wide @c nvidia unit, which CMake gates as @c USEARCH_TARGET_NVIDIA. */
constexpr bool basic_gpu_class_compiled(gpu_class_t gpu_class) noexcept {
    if (gpu_class != gpu_class_t::nvidia_ampere_k) return gpu_class_compiled(gpu_class);
#if USEARCH_TARGET_NVIDIA
    return true;
#else
    return false;
#endif
}

// The device listing crosses the ABI by casts, so the C vendors and classes mirror the engine's.
static_assert(static_cast<u32_t>(gpu_vendor_t::apple_k) == usearch_gpu_vendor_apple_k);
static_assert(static_cast<u32_t>(gpu_class_t::apple_metal10_k) == usearch_gpu_class_apple_metal10_k);

#pragma endregion Residencies

#pragma region Calls

/** Every argument structure that names an executor, so a delegated call keeps its arguments by
 *  value; whatever the copy points to stays the caller's until final. */
using deferred_arguments_t = std::variant< //

    /* No call */
    std::monostate, //

    /* Loading */
    usearch_dense_load_ram_t, usearch_dense_load_ram_buffer_t, usearch_dense_view_nvme_t,                //
    usearch_dense_load_vram_t, usearch_dense_load_vram_buffer_t,                                         //
    usearch_sequence_load_ram_t, usearch_sequence_load_ram_buffer_t, usearch_sequence_view_nvme_t,       //
    usearch_sequence_load_vram_t, usearch_sequence_load_vram_buffer_t,                                   //
    usearch_shapes_load_ram_t, usearch_shapes_load_ram_buffer_t, usearch_shapes_view_nvme_t,             //
    usearch_shapes_load_vram_t, usearch_shapes_load_vram_buffer_t,                                       //
    usearch_geospatial_load_ram_t, usearch_geospatial_load_ram_buffer_t, usearch_geospatial_view_nvme_t, //
    usearch_geospatial_load_vram_t, usearch_geospatial_load_vram_buffer_t,                               //

    /* Building */
    usearch_dense_reserve_t, usearch_dense_clear_t, usearch_dense_insert_t,                //
    usearch_sequence_reserve_t, usearch_sequence_clear_t, usearch_sequence_insert_t,       //
    usearch_shapes_reserve_t, usearch_shapes_clear_t, usearch_shapes_insert_t,             //
    usearch_geospatial_reserve_t, usearch_geospatial_clear_t, usearch_geospatial_insert_t, //

    /* Refining */
    usearch_dense_refine_t, usearch_dense_health_t, usearch_dense_absorb_t, usearch_dense_split_t,             //
    usearch_sequence_refine_t, usearch_sequence_health_t, usearch_sequence_absorb_t, usearch_sequence_split_t, //
    usearch_shapes_refine_t, usearch_shapes_health_t, usearch_shapes_absorb_t, usearch_shapes_split_t,         //
    usearch_geospatial_refine_t, usearch_geospatial_health_t, usearch_geospatial_absorb_t,                     //
    usearch_geospatial_split_t, usearch_dense_compact_t,                                                       //

    /* Refinements */
    usearch_dense_refinement_init_t, usearch_dense_refinement_next_t, usearch_dense_refinement_finish_t,          //
    usearch_sequence_refinement_init_t, usearch_sequence_refinement_next_t, usearch_sequence_refinement_finish_t, //
    usearch_shapes_refinement_init_t, usearch_shapes_refinement_next_t, usearch_shapes_refinement_finish_t,       //
    usearch_geospatial_refinement_init_t, usearch_geospatial_refinement_next_t,                                   //
    usearch_geospatial_refinement_finish_t,                                                                       //

    /* Reading */
    usearch_dense_search_t, usearch_dense_search_stream_next_t, usearch_dense_count_t, usearch_dense_get_t,      //
    usearch_dense_cursor_next_t, usearch_dense_sample_t, usearch_dense_describe_t,                               //
    usearch_sequence_search_t, usearch_sequence_count_t, usearch_sequence_get_t, usearch_sequence_cursor_next_t, //
    usearch_sequence_sample_t, usearch_sequence_describe_t,                                                      //
    usearch_shapes_search_t, usearch_shapes_count_t, usearch_shapes_get_t, usearch_shapes_cursor_next_t,         //
    usearch_shapes_sample_t, usearch_shapes_describe_t,                                                          //
    usearch_geospatial_search_t, usearch_geospatial_within_t, usearch_geospatial_count_t,                        //
    usearch_geospatial_get_t, usearch_geospatial_cursor_next_t, usearch_geospatial_sample_t,                     //
    usearch_geospatial_describe_t,                                                                               //

    /* Persisting */
    usearch_dense_save_t, usearch_dense_save_buffer_t, usearch_dense_metadata_read_t,                //
    usearch_dense_metadata_read_buffer_t,                                                            //
    usearch_sequence_save_t, usearch_sequence_save_buffer_t, usearch_sequence_metadata_read_t,       //
    usearch_sequence_metadata_read_buffer_t,                                                         //
    usearch_shapes_save_t, usearch_shapes_save_buffer_t, usearch_shapes_metadata_read_t,             //
    usearch_shapes_metadata_read_buffer_t,                                                           //
    usearch_geospatial_save_t, usearch_geospatial_save_buffer_t, usearch_geospatial_metadata_read_t, //
    usearch_geospatial_metadata_read_buffer_t,                                                       //

    /* Products */
    usearch_dense_multiply_t, usearch_dense_attend_t, usearch_dense_maxsim_t>;

/** The counts of the handles a call names - its index or target, cursor, refinement and arena -
 *  each raised at its entry and lowered at its end; null where it names none. */
using named_calls_t = std::array<u32_t*, 5>;

/** The one call that holds a USearch executor, from its claim to its end. */
struct holding_call_t {

    /** Free, or held by the call from its claim to its end. */
    call_claim_t claim = call_claim_t::free_k;

    /** The holding call's arguments; the entry runs the body when it meets this address again. */
    void const* arguments = nullptr;

    /** A delegated call's copy, which pool thread 0 runs, since the caller's structure may go out
     *  of scope. */
    deferred_arguments_t delegated_arguments;

    /** The caller's outcome, published last. */
    usearch_outcome_t* outcome = nullptr;

    /** The status the body recorded, which @c end_call publishes. */
    usearch_outcome_t recorded {usearch_status_internal_k, "The call finished without reporting a status"};

    /** The counts the entry raised, which @c end_call lowers. */
    named_calls_t named_calls {};

    /** The thread running the body: an eager call's caller, or a delegated one's pool thread 0. */
    std::thread::id body_thread;
};

/** The state behind @c usearch_executor_t: the discovered topology a RAM or NVMe executor was
 *  spawned over, the call holding it, and one residency's executor; the variant is the kind tag, so
 *  no second word can disagree. The executor is declared last, so its pool dies before the
 *  topology. */
struct usearch_executor_handle_t {
    std::unique_ptr<void, decltype(&fu_topology_delete)> topology {nullptr, &fu_topology_delete};

    /** The one call that holds the executor, from its claim to its end. */
    holding_call_t holding_call;

    std::variant<forkunion_executor_t, nvme_executor_t, gpu_executor_t> executor;
};

/** Requests AMX tile state alone when @p ceiling holds it, granted process-wide on Linux; more bits
 *  would rewrite FPCR. */
inline void request_tile_state(cpu_ceiling_t ceiling) noexcept {
    if (!(ceiling.compute & nk_cap_sapphireamx_k)) return;
    [[maybe_unused]] nk_status_t const requested = nk_cpu_configure_thread(nk_cap_sapphireamx_k);
}

/** Discovers the topology and spawns @p threads over it into @p handle under @p ceiling, near the
 *  one memory domain @p ram names on a machine of several, answering @p handle, or null once the
 *  outcome records why not; one thread is the caller. */
inline usearch_executor_t spawn_ram_executor(usearch_executor_handle_t& handle, std::size_t threads,
                                             cpu_ceiling_t ceiling, usearch_ram_options_t const& ram,
                                             usearch_outcome_t* outcome) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_status_device_k, outcome, "Topology discovery failed"), nullptr;
    handle.topology.reset(topology);
    // A pool spawns near one memory domain or across all of them, so several named ones take all.
    std::size_t domains_count = 0;
    bool const near_one = ram.memory_domains && ram.memory_domains_count == 1 &&
                          fu_memory_domains_count(topology, &domains_count) == fu_success_k && domains_count > 1;
    auto [spawned, status] = near_one ? forkunion_executor_t::spawn_near(
                                            topology, static_cast<fu_memory_domain_id_t>(ram.memory_domains[0]),
                                            threads, forkunion_allowed_of(ceiling))
                                      : forkunion_executor_t::spawn(topology, threads, forkunion_allowed_of(ceiling));
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    request_tile_state(ceiling);
    handle.executor.emplace<forkunion_executor_t>(std::move(spawned));
    return &handle;
}

/** What an NVMe executor spawns by: the fewest hardware queues of any disk or SPDK controller under
 *  its paths, zero when none answers, and the memory domain every one of them sits nearest, or
 *  @c no_memory_domain_k when they differ or none is known. */
struct nvme_media_shape_t {
    std::size_t fewest_queues = 0;
    fu_memory_domain_id_t memory_domain = no_memory_domain_k;
};

/** The shape of the media under @p paths, opened as views open them. */
inline nvme_media_shape_t nvme_media_shape(std::span<usearch_bytes_view_t const> paths) noexcept {
    nvme_media_shape_t shape;
    for (std::size_t position = 0; position != paths.size(); ++position) {
        std::string_view const named {reinterpret_cast<char const*>(paths[position].data), paths[position].length};
        io_device_queues_t queues;
        fu_memory_domain_id_t domain = no_memory_domain_k;
#if USEARCH_WITH_LINUX_SPDK
        if (names_transport_id(named)) {
            auto [name_space, open_status] = spdk_medium_t::open<fu::standard_yield_t>(named);
            if (succeeded(open_status)) queues = device_queues_of(name_space), domain = memory_domain_of(name_space);
            name_space.release<fu::standard_yield_t>();
        }
#endif
        if (!names_transport_id(named)) {
            file_medium_t const medium = file_medium_t::open(named).value;
            queues = device_queues_of(medium), domain = memory_domain_of(medium);
        }
        if (queues.hardware_queues && (!shape.fewest_queues || queues.hardware_queues < shape.fewest_queues))
            shape.fewest_queues = queues.hardware_queues;
        shape.memory_domain = !position || shape.memory_domain == domain ? domain : no_memory_domain_k;
    }
    return shape;
}

/** The threads an NVMe executor spawns when its caller names none: ForkUnion's default, capped by
 *  @p queues, the fewest hardware queues of its media, so no two threads share a queue; zero,
 *  ForkUnion's own default, when the cores are unknown. */
inline std::size_t nvme_default_threads(fu_topology_t topology, std::size_t queues) noexcept {
    auto const [cores, cores_status] = forkunion_executor_t::default_threads(topology);
    if (failed(cores_status)) return 0;
    return queues ? min_of(cores, queues) : cores;
}

/** Discovers the topology and spawns @p threads into @p handle under @p ceiling, each opening the
 *  engine @p config names, answering @p handle, or null once the outcome records why not; zero
 *  threads takes @ref nvme_default_threads over the media under @p paths. */
inline usearch_executor_t spawn_nvme_executor(usearch_executor_handle_t& handle, std::size_t threads,
                                              std::span<usearch_bytes_view_t const> paths,
                                              io_engine_config_t const& config, cpu_ceiling_t ceiling,
                                              usearch_outcome_t* outcome) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_status_device_k, outcome, "Topology discovery failed"), nullptr;
    handle.topology.reset(topology);
    nvme_media_shape_t const shape = nvme_media_shape(paths);
    std::size_t const queues = shape.fewest_queues;
    // Each SPDK thread holds a queue pair of its own, and a controller grants only so many.
    if (config.transport == io_transport_t::linux_spdk_k && queues && threads > queues)
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "More threads than the SPDK controller granted I/O queue pairs"),
               nullptr;
    if (!threads) threads = nvme_default_threads(topology, queues);
    // Drives on one socket keep their issuers there; drives on several spread them machine-wide.
    std::size_t domains_count = 0;
    fu_memory_domain_id_t const near_domain = fu_memory_domains_count(topology, &domains_count) == fu_success_k &&
                                                      domains_count > 1
                                                  ? shape.memory_domain
                                                  : no_memory_domain_k;
    auto [spawned, status] = nvme_executor_t::spawn(topology, threads, config, forkunion_allowed_of(ceiling),
                                                    near_domain);
    if (failed(status))
        return propagate_error(status, outcome, "No I/O engine opened on the executor's threads"), nullptr;
    request_tile_state(ceiling);
    handle.executor.emplace<nvme_executor_t>(std::move(spawned));
    return &handle;
}

/** The executor a null handle runs RAM work on: the calling thread, built into @p call_executor
 *  under the machine's own ceiling, since a pool of one never waits on another thread. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
inline usearch_executor_t call_executor_for(multi_type_&, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    return spawn_ram_executor(call_executor, 1, pick_cpu_ceiling(usearch_cpu_class_unknown_k).value,
                              usearch_ram_options_t {}, outcome);
}

/** The executor a null handle runs NVMe work on: one thread of synchronous reads. */
template <typename multi_type_>
    requires is_nvme_index<multi_type_>
inline usearch_executor_t call_executor_for(multi_type_&, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    // Zero leaves each call's depth to the block queues of the media it reads.
    io_engine_config_t config;
    config.transport = io_transport_t::blocking_k;
    return spawn_nvme_executor(call_executor, 1, {}, config, pick_cpu_ceiling(usearch_cpu_class_unknown_k).value,
                               outcome);
}

/** Applies @p visitor to whichever executor @p executor holds, a GPU one included; an empty GPU
 *  alternative answers @c device_error_k. */
template <typename visitor_type_>
inline status_t visit_executor(usearch_executor_handle_t& executor, visitor_type_&& visitor) noexcept {
    return std::visit(
        [&]<typename executor_type_>(executor_type_& typed) noexcept {
            if constexpr (!std::is_same_v<executor_type_, gpu_executor_t>) return visitor(typed);
            else
                return std::visit(
                    [&]<typename gpu_type_>(gpu_type_& gpu) noexcept {
                        if constexpr (std::is_same_v<gpu_type_, std::monostate>) return status_t::device_error_k;
                        else return visitor(gpu);
                    },
                    typed);
        },
        executor.executor);
}

/** The residency's executor inside @p executor 's variant: GPU executors nest one level deeper. */
template <typename executor_type_>
inline executor_type_* residency_executor(usearch_executor_handle_t& executor) noexcept {
    if constexpr (std::is_same_v<executor_type_, forkunion_executor_t> ||
                  std::is_same_v<executor_type_, nvme_executor_t>)
        return std::get_if<executor_type_>(&executor.executor);
    else {
        gpu_executor_t* const gpu = std::get_if<gpu_executor_t>(&executor.executor);
        return gpu ? std::get_if<executor_type_>(gpu) : nullptr;
    }
}

/** The typed executor behind the C handle, or null with @c usearch_status_device_k on a kind
 *  mismatch. A null handle builds @p call_executor for this call alone, through the overload of
 *  @c call_executor_for its residency names, the VRAM ones defined beside the dense and the basic
 *  GPU variants. */
template <typename multi_type_>
inline typename multi_type_::executor_t* executor_of(multi_type_& variant, usearch_executor_t executor,
                                                     usearch_executor_handle_t& call_executor,
                                                     usearch_outcome_t* outcome) noexcept {
    if (!executor) executor = call_executor_for(variant, call_executor, outcome);
    if (!executor) return nullptr;
    typename multi_type_::executor_t* const typed = residency_executor<typename multi_type_::executor_t>(*executor);
    if (!typed)
        propagate_error(usearch_status_device_k, outcome, "The executor's kind must match the handle's residency");
    return typed;
}

/** Whether @p traversal names a walk, a scan, or nothing, which derives the walk, and @p predicate
 *  leaves the walks at @p residency alone, since a host callback cannot narrow a device walk; false
 *  once @p outcome records why not. */
inline bool accepts_traversal(arena_residency_t residency, usearch_traversal_t traversal, usearch_predicate_t predicate,
                              usearch_outcome_t* outcome) noexcept {
    if (traversal != usearch_traversal_unknown_k && traversal != usearch_traversal_walk_k &&
        traversal != usearch_traversal_scan_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown traversal");
    if (predicate.callback && residency != ram_residency_k)
        return propagate_error(usearch_status_unsupported_k, outcome, "A host predicate cannot narrow a device walk");
    return true;
}

/** What @c usearch_arena_t owns: memory that one call at a time rewinds and refills, as host pages
 *  or on one device, and the call holding it. */
struct usearch_arena_handle_t {
    growing_arena<forkunion_allocator_t> memory {forkunion_allocator_t {no_memory_domain_k}};

    /** Device memory on the first stream of the executor the arena was made for, or none. */
    gpu_arena_t device_memory;

    /** The executor whose first stream the device memory rides, the only one whose calls it serves;
     *  null for host pages, which serve any. */
    usearch_executor_t executor = nullptr;

    /** Where the memory lives: host pages, or a device's VRAM. */
    arena_residency_t residency = ram_residency_k;

    /** The outcome of the call holding the arena, from its claim until just before its final store;
     *  touched through a reference, since a delegated call finishes on a pool thread. */
    usearch_outcome_t const* holder = nullptr;

    /** Executor calls naming the arena that have not ended, which its free waits out. */
    u32_t named_calls = 0;
};

/** Holds @p arena for the call whose outcome is @p call, answering whether it was free or already
 *  that call's; a null arena holds nothing, left for the verb to refuse. */
inline bool hold_arena(usearch_arena_t arena, usearch_outcome_t const* call) noexcept {
    if (!arena) return true;
    usearch_outcome_t const* holder = nullptr;
    return fu::standard_atomic_ref<usearch_outcome_t const*>(arena->holder)
               .compare_exchange_strong(holder, call, std::memory_order_acquire, std::memory_order_acquire) ||
           holder == call;
}

/** Lets go of @p arena when the call whose outcome is @p call holds it. */
inline void release_arena(usearch_arena_t arena, usearch_outcome_t const* call) noexcept {
    if (arena)
        fu::standard_atomic_ref<usearch_outcome_t const*>(arena->holder)
            .compare_exchange_strong(call, nullptr, std::memory_order_release, std::memory_order_relaxed);
}

/** Where @c usearch_gpu_devices lists the device @p arena lives on, or 0 for host pages; the arena
 *  keeps its vendor's own ordinal, which the listing maps back. */
inline std::size_t arena_device_ordinal(usearch_arena_handle_t const& arena) noexcept {
    gpu_executor_t const* const devices = arena.executor ? std::get_if<gpu_executor_t>(&arena.executor->executor)
                                                         : nullptr;
    if (!devices) return 0;
    gpu_vendor_t const vendor = std::visit(
        []<typename executor_type_>(executor_type_ const&) noexcept {
            if constexpr (is_gpu_executor<executor_type_>) return executor_type_::vendor_k;
            else return gpu_vendor_t::unknown_k;
        },
        *devices);
    std::span<gpu_device_t const> const listed = gpu_devices();
    for (std::size_t ordinal = 0; ordinal != listed.size(); ++ordinal)
        if (listed[ordinal].vendor == vendor && listed[ordinal].vendor_ordinal == arena.residency.device)
            return ordinal;
    return 0;
}

/** Whether @p address lies in memory @p arena handed out, which its next rewind reuses or frees. */
inline bool arena_contains(usearch_arena_handle_t const& arena, void const* address) noexcept {
    return arena.device_memory ? arena.device_memory.contains(address) : arena.memory.contains(address);
}

/** Where @ref claim_arena wants @p arena for a verb filling host pages over an index at
 *  @p residency : a host arena keeps them, and any other must sit on that device to take them over
 *  by @ref publish. */
inline arena_residency_t staging_residency(usearch_arena_t arena, arena_residency_t residency) noexcept {
    return arena && arena->residency == ram_residency_k ? ram_residency_k : residency;
}

/** Holds @p arena for the call whose outcome is @p call and rewinds it, freeing the previous call's
 *  outputs, so an input of @p inputs still inside it is refused, as is an arena elsewhere than
 *  @p residency or device memory another @p executor made. The call's @c call_outcome_t lets go of
 *  it before the final store. */
inline usearch_arena_handle_t* claim_arena(usearch_arena_t arena, usearch_outcome_t const* call,
                                           usearch_executor_t executor, arena_residency_t residency,
                                           std::initializer_list<void const*> inputs,
                                           usearch_outcome_t* outcome) noexcept {
    if (!arena)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The arena must not be null"), nullptr;
    if (arena->residency != residency)
        return propagate_error(usearch_status_device_k, outcome, "The arena lives where the call does not compute"),
               nullptr;
    if (arena->device_memory && arena->executor != executor)
        return propagate_error(usearch_status_device_k, outcome,
                               "A device arena serves calls on the executor that made it"),
               nullptr;
    if (!hold_arena(arena, call))
        return propagate_error(usearch_status_contended_k, outcome, "The arena serves another call"), nullptr;
    for (void const* input : inputs)
        if (input && arena_contains(*arena, input))
            return propagate_error(usearch_status_invalid_argument_k, outcome,
                                   "An input lies inside the arena it rewinds"),
                   nullptr;
    arena->device_memory.clear();
    arena->memory.clear();
    return arena;
}

/** Carves @p count values of @p value_type_ out of a claimed arena's host pages, @p staging, where
 *  every verb computes, even one whose outputs @ref publish moves to a device; null when they
 *  cannot grow. */
template <typename value_type_>
inline value_type_* carve(growing_arena<forkunion_allocator_t>& staging, std::size_t count) noexcept {
    return reinterpret_cast<value_type_*>(staging.allocate(count * sizeof(value_type_), default_alignment_k));
}

/** Carves @p count values of @p value_type_ out of a claimed arena's @p device_memory, for outputs
 *  the device writes in place. */
template <typename value_type_>
inline value_type_* carve(gpu_arena_t& device_memory, std::size_t count) noexcept {
    return device_memory.template allocate<value_type_>(count);
}

/** Where the @p count values @ref carve put at @p output reach the caller: in place on host pages,
 *  or one upload to the arena's device queued on its executor's first stream, which the call
 *  awaits once through @ref drained, the host pages it reads living until then. */
template <typename value_type_>
inline expected<value_type_*> publish(usearch_arena_handle_t& arena, value_type_* output, std::size_t count) noexcept {
    if (!arena.device_memory || !output || !count) return {output, status_t::success_k};
    using uploaded_t = std::remove_const_t<value_type_>;
    uploaded_t* const uploaded = carve<uploaded_t>(arena.device_memory, count);
    if (!uploaded) return {nullptr, status_t::bad_alloc_k};
    return {uploaded, arena.device_memory.upload(uploaded, output, count * sizeof(value_type_))};
}

/** The first failure of the steps of one call, or success. */
inline status_t first_failure(std::initializer_list<status_t> statuses) noexcept {
    for (status_t const status : statuses)
        if (failed(status)) return status;
    return status_t::success_k;
}

/** Keeps the first failure of @p status, awaiting @p arena 's uploads either way. */
inline status_t drained(usearch_arena_handle_t& arena, status_t status) noexcept {
    status_t const waited = arena.device_memory.wait();
    return failed(status) ? status : waited;
}

/** The arena @p arguments name for their outputs, or null for a verb that names none. */
template <typename arguments_type_>
inline usearch_arena_t arena_of(arguments_type_ const& arguments) noexcept {
    if constexpr (requires(usearch_arena_t named) { named = arguments.arena; }) return arguments.arena;
    else return nullptr;
}

/** The executor whose holding call @p arguments are, so the body records for @c end_call; null
 *  for a call without an executor or one that never claimed it. */
template <typename arguments_type_>
inline usearch_executor_t holding_executor_of(arguments_type_ const& arguments) noexcept {
    if constexpr (requires(usearch_executor_t named) { named = arguments.executor; }) {
        usearch_executor_t const executor = arguments.executor;
        if (!executor) return nullptr;
        void const* const holding =
            fu::standard_atomic_ref<void const*>(executor->holding_call.arguments).load(std::memory_order_relaxed);
        return holding == &arguments ? executor : nullptr;
    }
    else return nullptr;
}

/**
 *  @brief A verb's outcome while it runs: every step records into a private one, and the end lets
 *      go of the arena, then publishes the record itself, or leaves it to @c end_call.
 *
 *  Declared first in a verb, it unwinds last, so a waiter that sees the final status may reuse the
 *  arena at once. A path that forgot to report finishes @c usearch_status_internal_k rather than
 *  hanging its waiters.
 */
class call_outcome_t {
  public:
    template <typename arguments_type_>
    explicit call_outcome_t(arguments_type_ const& arguments) noexcept
        : published_(arguments.outcome), holding_(holding_executor_of(arguments)), arena_(arena_of(arguments)) {
        if (!holding_) propagate_pending(published_);
    }
    ~call_outcome_t() noexcept {
        release_arena(arena_, published_);
        if (holding_) holding_->holding_call.recorded = recorded_;
        else propagate_error(recorded_.status, published_, recorded_.message);
    }
    call_outcome_t(call_outcome_t const&) = delete;
    call_outcome_t& operator=(call_outcome_t const&) = delete;

    /** The private record every step reports into. */
    operator usearch_outcome_t*() noexcept { return &recorded_; }

  private:
    usearch_outcome_t* published_;

    /** The executor the call holds, whose @c end_call publishes the record, or null. */
    usearch_executor_t holding_;
    usearch_arena_t arena_;
    usearch_outcome_t recorded_ {usearch_status_internal_k, "The call finished without reporting a status"};
};

/** Whether the C hints @p given of @p count queries refuse to parse, once @p outcome records why;
 *  @c NULL hints parse as none. */
inline bool rejects_hints(usearch_matches_view_t const* given, std::size_t count, usearch_outcome_t* outcome) noexcept {
    if (!given || !count) return false;
    if (!given->counts || (given->columns && !given->keys))
        return !propagate_error(usearch_status_invalid_argument_k, outcome, "The hints must name keys and counts");
    for (std::size_t query = 0; query != count; ++query)
        if (given->counts[query] > given->columns)
            return !propagate_error(usearch_status_shape_mismatch_k, outcome, "A query counts more hints than columns");
    return false;
}

/** Where a ranked run is written inside a claimed arena: @c wanted keys and scores per query, and a
 *  count per query. */
struct matches_outputs_t {
    usearch_key_t* keys = nullptr;
    usearch_distance_t* scores = nullptr;
    std::size_t* counts = nullptr;
    std::size_t wanted = 0;

    /** The view the call publishes once every query published. */
    usearch_matches_view_t view() const noexcept { return {keys, scores, counts, wanted}; }
};

/** The outputs of @p count queries of @p wanted matches, carved out of a claimed arena's host pages
 *  or device memory, @p memory; a zero @p wanted carves only the counts. */
template <typename memory_type_>
inline expected<matches_outputs_t> carve_matches(memory_type_& memory, std::size_t count, std::size_t wanted) noexcept {
    matches_outputs_t outputs;
    outputs.wanted = wanted;
    outputs.counts = carve<std::size_t>(memory, count);
    if (wanted) {
        outputs.keys = carve<usearch_key_t>(memory, count * wanted);
        outputs.scores = carve<usearch_distance_t>(memory, count * wanted);
    }
    bool const carved = outputs.counts && (!wanted || (outputs.keys && outputs.scores));
    return {outputs, carved ? status_t::success_k : status_t::bad_alloc_k};
}

/** The @p count queries @ref carve_matches put in @p outputs as the caller reads them, where
 *  @p arena hands outputs out, once its uploads drained. */
inline expected<usearch_matches_view_t> publish_matches(usearch_arena_handle_t& arena, matches_outputs_t const& outputs,
                                                        std::size_t count) noexcept {
    std::size_t const entries = count * outputs.wanted;
    auto const [counts, counts_status] = publish(arena, outputs.counts, count);
    auto const [keys, keys_status] = publish(arena, outputs.keys, entries);
    auto const [scores, scores_status] = publish(arena, outputs.scores, entries);
    status_t const published = drained(arena, first_failure({counts_status, keys_status, scores_status}));
    return {{keys, scores, counts, outputs.wanted}, published};
}

/** Where a cursor pull gathers its entries' links, straight in the call's arena: their keys end to
 *  end, and the offset where each entry's run starts, one more than the entries, as the C tape
 *  reads them out. */
struct links_outputs_t {
    growing_buffer<us::utilities::uuid128_t, thread_allocator_t> keys;
    growing_buffer<std::size_t, thread_allocator_t> offsets;

    explicit links_outputs_t(growing_arena<forkunion_allocator_t>& memory) noexcept
        : keys(thread_allocator_t {memory}), offsets(thread_allocator_t {memory}) {}

    /** Opens the first run; @c bad_alloc_k when the arena cannot grow. */
    status_t open() noexcept { return offsets.push_back(0); }

    /** The C tape over the gathered links. */
    usearch_keys_tape_view_t tape() const noexcept {
        static_assert(sizeof(us::utilities::uuid128_t) == sizeof(usearch_key_t), "Links land as the walk reads them");
        return {reinterpret_cast<usearch_key_t const*>(keys.data()), offsets.data()};
    }
};

/** Where the host @p links a pull gathered reach the caller, as @ref publish moves them. */
inline expected<usearch_keys_tape_view_t> publish_links(usearch_arena_handle_t& arena,
                                                        links_outputs_t const& links) noexcept {
    usearch_keys_tape_view_t const gathered = links.tape();
    auto const [keys, keys_status] = publish(arena, gathered.keys, links.keys.size());
    auto const [offsets, offsets_status] = publish(arena, gathered.offsets, links.offsets.size());
    return {{keys, offsets}, first_failure({keys_status, offsets_status})};
}

/** Where a sample is written on a claimed arena's host pages, @c wanted rows each: the plan writes
 *  its keys, ordinals, parents, f32 parent distances and coordinates, and @ref publish_sample
 *  widens the distances into the parents' scores and counts. */
struct sample_staging_t {
    usearch_key_t* keys = nullptr;
    std::size_t* ordinals = nullptr;
    usearch_key_t* parents = nullptr;
    f32_t* parent_distances = nullptr;
    usearch_distance_t* scores = nullptr;
    std::size_t* counts = nullptr;
    f32_t* coordinates = nullptr;
    std::size_t wanted = 0;

    /** The plan's outputs over these rows, in the key type @p key_type_ the index stores. */
    template <typename key_type_>
    sample_outputs<key_type_> outputs() const noexcept {
        static_assert(sizeof(key_type_) == sizeof(usearch_key_t), "Keys land in the arena as the plan writes them");
        return {{reinterpret_cast<key_type_*>(keys), wanted},
                {ordinals, wanted},
                {reinterpret_cast<key_type_*>(parents), wanted},
                {parent_distances, wanted},
                {coordinates, wanted * 3}};
    }
};

/** Carves the staging of @p wanted draws out of the host pages @p memory. */
inline expected<sample_staging_t> carve_sample(growing_arena<forkunion_allocator_t>& memory,
                                               std::size_t wanted) noexcept {
    sample_staging_t staging;
    staging.wanted = wanted;
    staging.keys = carve<usearch_key_t>(memory, wanted);
    staging.ordinals = carve<std::size_t>(memory, wanted);
    staging.parents = carve<usearch_key_t>(memory, wanted);
    staging.parent_distances = carve<f32_t>(memory, wanted);
    staging.scores = carve<usearch_distance_t>(memory, wanted);
    staging.counts = carve<std::size_t>(memory, wanted);
    staging.coordinates = carve<f32_t>(memory, wanted * 3);
    bool const carved = staging.keys && staging.ordinals && staging.parents && staging.parent_distances &&
                        staging.scores && staging.counts && staging.coordinates;
    return {staging, carved || !wanted ? status_t::success_k : status_t::bad_alloc_k};
}

/** The @p count keys of @p keys as the plan reads them: the caller's own when packed, otherwise
 *  copied onto the host pages @p memory. */
template <typename key_type_>
inline expected<std::span<key_type_ const>> view_keys(usearch_keys_view_t keys, std::size_t count,
                                                      growing_arena<forkunion_allocator_t>& memory) noexcept {
    static_assert(sizeof(key_type_) == sizeof(usearch_key_t), "The plan reads the caller's keys as they are");
    if (count <= 1 || keys.stride == sizeof(usearch_key_t))
        return {{reinterpret_cast<key_type_ const*>(keys.data), count}, status_t::success_k};
    key_type_* const packed = carve<key_type_>(memory, count);
    if (!packed) return {{}, status_t::bad_alloc_k};
    for (std::size_t offset = 0; offset != count; ++offset)
        std::memcpy(packed + offset, reinterpret_cast<byte_t const*>(keys.data) + offset * keys.stride,
                    sizeof(usearch_key_t));
    return {{packed, count}, status_t::success_k};
}

/** The @p count rows of three f32 values of @p rows as the plan reads them: the caller's own when
 *  packed, otherwise copied onto the host pages @p memory. */
inline expected<std::span<f32_t const>> view_coordinates(usearch_rows_view_t rows, std::size_t count,
                                                         growing_arena<forkunion_allocator_t>& memory) noexcept {
    std::size_t const row_bytes = 3 * sizeof(f32_t);
    if (count <= 1 || rows.stride == row_bytes)
        return {{static_cast<f32_t const*>(rows.data), count * 3}, status_t::success_k};
    f32_t* const packed = carve<f32_t>(memory, count * 3);
    if (!packed) return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row)
        std::memcpy(packed + row * 3, static_cast<byte_t const*>(rows.data) + row * rows.stride, row_bytes);
    return {{packed, count * 3}, status_t::success_k};
}

/** A sample's shown keys, their coordinates and its focus, viewed in place where packed and copied
 *  onto the host pages @p memory otherwise. */
template <typename key_type_, typename arguments_type_>
inline expected<sample_inputs<key_type_>> view_sample(arguments_type_ const& arguments,
                                                      growing_arena<forkunion_allocator_t>& memory) noexcept {
    auto const [shown, shown_status] = view_keys<key_type_>(arguments.shown, arguments.shown_count, memory);
    auto const [coordinates, coordinates_status] = view_coordinates(arguments.shown_coordinates, arguments.shown_count,
                                                                    memory);
    auto const [focus, focus_status] = view_keys<key_type_>(arguments.focus, arguments.focus_count, memory);
    if (failed(first_failure({shown_status, coordinates_status, focus_status}))) return {{}, status_t::bad_alloc_k};
    return {{shown, coordinates, focus}, status_t::success_k};
}

/** Resets every output of a sample's @p arguments, so a refused call leaves nothing behind. */
template <typename arguments_type_>
inline void reset_sample(arguments_type_ const& arguments) noexcept {
    *arguments.keys = {};
    *arguments.entry_ordinals = nullptr;
    *arguments.parents = {};
    *arguments.coordinates = {};
    *arguments.written = 0;
    if (arguments.stats) *arguments.stats = {};
}

/** Whether a sample's @p arguments name their shown keys and focus as the header asks; false once
 *  @p outcome records why not. */
template <typename arguments_type_>
inline bool accepts_sample(arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
    usearch_rows_view_t const& coordinates = arguments.shown_coordinates;
    if ((arguments.shown_count && !arguments.shown.data) || (arguments.focus_count && !arguments.focus.data))
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The shown keys and focus must name keys");
    if ((arguments.shown.stride && arguments.shown.stride < sizeof(usearch_key_t)) ||
        (arguments.focus.stride && arguments.focus.stride < sizeof(usearch_key_t)))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key");
    if (arguments.shown_count &&
        (!coordinates.data || coordinates.columns != 3 || coordinates.scalar != usearch_scalar_f32_k ||
         (coordinates.stride && coordinates.stride < 3 * sizeof(f32_t))))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The shown coordinates must be three f32 values per key");
    return true;
}

/** Hands the @p sampled draws of @p staging out through a sample's @p arguments, their parents'
 *  distances widened into the scores; the self-parent of a first draw without shown points counts
 *  as no parent. */
template <typename arguments_type_>
inline void publish_sample(arguments_type_ const& arguments, sample_staging_t const& staging,
                           sample_result_t const& sampled) noexcept {
    std::size_t const written = sampled.written;
    if (arguments.stats)
        arguments.stats->visited_members += sampled.visited_members,
            arguments.stats->computed_distances += sampled.computed_distances;
    for (std::size_t draw = 0; draw != written; ++draw) {
        bool const own = staging.parent_distances[draw] == 0 &&
                         !std::memcmp(&staging.parents[draw], &staging.keys[draw], sizeof(usearch_key_t));
        staging.scores[draw] = static_cast<usearch_distance_t>(staging.parent_distances[draw]);
        staging.counts[draw] = own ? 0 : 1;
    }
    if (!written) return;
    *arguments.keys = {staging.keys, sizeof(usearch_key_t)};
    *arguments.entry_ordinals = staging.ordinals;
    *arguments.parents = {staging.parents, staging.scores, staging.counts, 1};
    *arguments.coordinates = {staging.coordinates, 3 * sizeof(f32_t), 3, usearch_scalar_f32_k};
    *arguments.written = written;
}

/** Writes the @p required bytes of a checkpoint into @p arena through @p save, as
 *  <tt>save(bytes_span_t) → status_t</tt>, and points @p monolith at its published copy. */
template <typename save_type_>
inline void save_into_arena(usearch_arena_handle_t& arena, expected<std::size_t> required, save_type_&& save,
                            usearch_bytes_view_t& monolith, usearch_outcome_t* outcome) noexcept {
    if (failed(required.status)) return propagate_error(required.status, outcome), void();
    std::byte* const checkpoint = carve<std::byte>(arena.memory, required.value);
    if (!checkpoint)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
    if (status_t const saved = save(bytes_span_t {checkpoint, required.value}); failed(saved))
        return propagate_error(saved, outcome), void();
    // A device checkpoint is written on the host, then uploaded to the arena's device.
    auto const [published, publish_status] = publish(arena, checkpoint, required.value);
    if (status_t const uploaded = drained(arena, publish_status); failed(uploaded))
        return propagate_error(uploaded, outcome, "Failed to copy the checkpoint to the device"), void();
    monolith = {reinterpret_cast<uint8_t const*>(published), required.value};
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** The run read as text - a path, which is bytes that happen to decode. The ABI spells a byte
 *  @c uint8_t and the language spells it @c char, so the two never meet implicitly. */
inline std::string_view as_text(usearch_bytes_view_t bytes) noexcept {
    return {reinterpret_cast<char const*>(bytes.data), bytes.length};
}

/** The run read as the engine's bytes - a buffer the ABI spells in @c uint8_t. */
inline bytes_view_t as_bytes(usearch_bytes_view_t bytes) noexcept {
    return {reinterpret_cast<byte_t const*>(bytes.data), bytes.length};
}

#pragma endregion Calls

#pragma region Holding Calls

/** The host threads behind @p executor, or null for a VRAM one, which has none. */
inline forkunion_executor_t* threads_of(usearch_executor_handle_t& executor) noexcept {
    if (forkunion_executor_t* const threads = std::get_if<forkunion_executor_t>(&executor.executor)) return threads;
    if (nvme_executor_t* const nvme = std::get_if<nvme_executor_t>(&executor.executor)) return &nvme->threads();
    return nullptr;
}

/** Whether @p executor hands deferred calls to its thread 0: a caller-exclusive pool does, while an
 *  inclusive one, a single thread, WebAssembly and a VRAM executor run them at submission. */
inline bool delegates(usearch_executor_handle_t& executor) noexcept {
    forkunion_executor_t const* const threads = threads_of(executor);
    return threads && threads->exclusivity() == fu_caller_exclusive_k;
}

/** Whether the calling thread runs @p executor 's holding call body, as a callback the body calls
 *  on its own thread does. */
inline bool runs_holding_call(usearch_executor_handle_t& executor) noexcept {
    std::thread::id const body_thread =
        fu::standard_atomic_ref<std::thread::id>(executor.holding_call.body_thread).load(std::memory_order_relaxed);
    return body_thread == std::this_thread::get_id();
}

/** Waits out the tail of the last delegated call's slice on thread 0, after its status went out. */
inline void await_delegated_call(usearch_executor_handle_t& executor) noexcept {
    if (forkunion_executor_t* const threads = threads_of(executor)) threads->unsafe_await();
}

/** Raises the count of every handle @p arguments name and answers where they are, so @c end_call
 *  lowers the same ones; an output slot for a handle the call creates names nothing yet. */
template <typename arguments_type_>
inline named_calls_t raise_named_calls(arguments_type_ const& arguments) noexcept {
    named_calls_t named {};
    u32_t** next = named.data();
    auto const name = [&](auto* handle) noexcept {
        if (!handle) return;
        host_outcome_verbs().raise_named(&handle->named_calls);
        *next++ = &handle->named_calls;
    };
    if constexpr (requires { arguments.index->named_calls; }) name(arguments.index);
    if constexpr (requires { arguments.target->named_calls; }) name(arguments.target);
    if constexpr (requires { arguments.cursor->named_calls; }) name(arguments.cursor);
    if constexpr (requires { arguments.refinement->named_calls; }) name(arguments.refinement);
    if constexpr (requires { arguments.arena->named_calls; }) name(arguments.arena);
    return named;
}

/** Returns once no executor call naming the handle that owns @p named_calls is left: what every
 *  free and every consumed donor waits for. */
inline void wait_named_calls(u32_t const& named_calls) noexcept { host_outcome_verbs().wait_named(&named_calls); }

/** Ends the call holding @p executor: its handle counts, the executor claim, then its status. A
 *  delegated call ends inside thread 0's slice, so the next call awaits that dispatch first. */
inline void end_call(usearch_executor_handle_t& executor) noexcept {
    holding_call_t& call = executor.holding_call;
    usearch_outcome_t* const outcome = call.outcome;
    usearch_outcome_t const recorded = call.recorded;
    for (u32_t* const named : call.named_calls)
        if (named) host_outcome_verbs().lower_named(named);
    fu::standard_atomic_ref<void const*>(call.arguments).store(nullptr, std::memory_order_relaxed);
    fu::standard_atomic_ref<std::thread::id>(call.body_thread).store(std::thread::id {}, std::memory_order_relaxed);
    call.delegated_arguments.template emplace<std::monostate>();
    host_outcome_verbs().release_claim(&call.claim);
    propagate_error(recorded.status, outcome, recorded.message);
}

/** Marks @p arguments as @p call 's body, run on the calling thread. */
inline void hold_body(holding_call_t& call, void const* arguments) noexcept {
    fu::standard_atomic_ref<void const*>(call.arguments).store(arguments, std::memory_order_relaxed);
    fu::standard_atomic_ref<std::thread::id>(call.body_thread)
        .store(std::this_thread::get_id(), std::memory_order_relaxed);
}

/**
 *  @brief Every verb's entry. A call on an executor claims it and runs its body now, or dispatches
 *      a copy to thread 0 when it is deferred on a caller-exclusive pool; either way it answers
 *      true, and @c end_call publishes the status.
 *
 *  The body itself, met again under its own arguments, answers false, so the verb runs it in place.
 *  So does a call with no executor, which is final when it returns, whatever its completion says.
 *  A second call on a held executor finishes @c usearch_status_contended_k, and nothing queues.
 */
template <auto verb_, typename arguments_type_>
inline bool submitted(arguments_type_* arguments) noexcept {
    usearch_executor_t const executor = arguments->executor;
    if (!executor || holding_executor_of(*arguments)) return false;
    holding_call_t& call = executor->holding_call;
    if (host_outcome_verbs().take_claim(&call.claim) == call_claim_t::held_k)
        return propagate_error(usearch_status_contended_k, arguments->outcome, "Another call holds the executor"), true;
    await_delegated_call(*executor);
    call.outcome = arguments->outcome;
    call.named_calls = raise_named_calls(*arguments);
    // Before the body or the dispatch, never after, so it never overwrites a final status.
    propagate_pending(arguments->outcome);
    if (arguments->completion != usearch_completion_deferred_k || !delegates(*executor)) {
        hold_body(call, arguments);
        verb_(arguments);
        return end_call(*executor), true;
    }
    arguments_type_& copy = call.delegated_arguments.template emplace<arguments_type_>(*arguments);
    fu::standard_atomic_ref<void const*>(call.arguments).store(&copy, std::memory_order_relaxed);
    fu_for_threads_t const delegated_body = [](fu_lambda_context_t context, std::size_t thread, std::size_t) noexcept {
        if (thread) return;
        usearch_executor_handle_t& held = *static_cast<usearch_executor_t>(context);
        arguments_type_* const delegated = std::get_if<arguments_type_>(&held.holding_call.delegated_arguments);
        hold_body(held.holding_call, delegated);
        verb_(delegated);
        end_call(held);
    };
    if (failed(threads_of(*executor)->unsafe_for_threads(delegated_body, executor)))
        return call.recorded = {usearch_status_contended_k, "The pool runs another dispatch"}, end_call(*executor),
               true;
    return true;
}

/** Returns once no call holds @p executor, on the class waiter. */
inline void wait_while_claimed(usearch_executor_handle_t& executor) noexcept {
    host_outcome_verbs().wait_claim(&executor.holding_call.claim);
}

#pragma endregion Holding Calls
