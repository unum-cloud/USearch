/**
 *  @file src/lib.hpp
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What every modality's C surface shares: the outcome and progress crossings, the CPU
 *      classes, the executor and arena handles, and the deferred calls every verb enters through.
 *
 *  Each modality's dispatch unit includes this, so one verb reads the same on every modality. The
 *  registry of deferred calls and the executor, arena and outcome verbs are defined once, in
 *  @c dense_index.cuh, so a wait or a free finds the calls of every modality naming its handle.
 */
#pragma once

#include <cassert> // `assert` - a per-thread family is opened before it is asked
#include <cstddef> // `std::size_t`

#include <array>            // `std::array` - the predicate's batch
#include <bit>              // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <memory>           // `std::unique_ptr` - the executor's topology
#include <mutex>            // `std::lock_guard` over the ticks' lock and the deferred tables
#include <span>             // `std::span` - key batches crossing the ABI
#include <string_view>      // `std::string_view` - checkpoint paths
#include <type_traits>      // `std::is_same_v` - the residency discrimination
#include <variant>          // `std::variant`, `std::visit`

#include <forkunion.h>                // `fu_topology_new` - the executor factories' discovery
#include <forkunion/atomics.hpp>      // `fu::standard_atomic_ref` & kin - the classes' atomic references
#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/types.hpp>        // `fu::spin_mutex_t` - the ticks' lock
#include <numkong/capabilities.h>     // `nk_capability_t`, `nk_cpu_configure_thread` - compute masks, AMX grant
#include <numkong/vector.hpp>         // `nk::vector_view` - the rows a value lookup resolves

#include "usearch/usearch.h"                 // every modality's argument structures
#include "usearch/utilities/nvme_device.hpp" // `nvme_executor_t`, `io_engine_config_t`
#include "usearch/utilities/ram_device.hpp"  // `forkunion_executor_t`, `forkunion_allocator_t`, `no_memory_domain_k`
#include "usearch/utilities/vram_device.hpp" // `gpu_executor_t`, `gpu_arena_t`

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

inline usearch_scalar_t usearch_scalar_of(nk_dtype_t dtype) noexcept {
    for (scalar_dtype_pair_t const& pair : scalar_dtype_pairs_k)
        if (pair.dtype == dtype) return pair.scalar;
    return usearch_scalar_unknown_k;
}

/** Reports a handler hears when the caller leaves @c usearch_progress_t::steps zero, a tenth of the
 *  call apart. */
inline constexpr std::size_t default_progress_steps_k = 10;

/** The verbs over the words a call's waiters watch - its outcome's status and the deferred
 *  registry's count of finished calls - lowered onto one CPU class's reference and waiter. A
 *  release store is the same instruction on every class; the wait is what differs, and it watches
 *  the address, so no store notifies. */
struct outcome_verbs_t {
    void (*publish)(usearch_outcome_t*, usearch_status_t, std::memory_order) noexcept;
    usearch_status_t (*load)(usearch_outcome_t const*) noexcept;

    /** Loops on the class waiter while the status reads deferred. */
    usearch_status_t (*wait)(usearch_outcome_t const*) noexcept;

    /** Counts one more finished call, with a release. */
    void (*count_finished)(u32_t*) noexcept;

    /** Waits while the count still reads the value seen. */
    void (*wait_finished)(u32_t const*, u32_t) noexcept;
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
    else if (stats) {
        stats->visited_members += result.visited_members;
        stats->computed_distances += result.computed_distances;
        stats->restarts += result.restarts;
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

/** Whether a result carries traversal telemetry - one probe both crossings read. */
template <typename value_type_>
concept carries_telemetry = requires(value_type_ const& value) { value.visited_members; } ||
                            requires(value_type_ const& value) { value.telemetry.visited_members; };

/** The one crossing for a fallible step: folds telemetry into @p stats when the payload carries
 *  any, finishes @p outcome only on failure, and answers whether @c .value is readable. */
template <typename value_type_>
inline bool crossed(expected<value_type_> const& result, usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,
                    char const* message = nullptr) noexcept {
    if constexpr (carries_telemetry<value_type_>) accumulate_stats(stats, result.value);
    return result || propagate_error(result.status, outcome, message);
}

/** The C key struct and the engine's @c utilities::uuid_t share sixteen bytes - one bit cast in. */
inline us::utilities::uuid_t uuid_of(usearch_key_t key) noexcept { return std::bit_cast<us::utilities::uuid_t>(key); }

/** The predicate lifted onto the index's key vocabulary, asked a quad at a time through arrays of
 *  its own key and admission types on the stack. */
struct predicate_t {
    usearch_predicate_t predicate;

    template <std::size_t extent_>
    void operator()(std::span<us::utilities::uuid_t const, extent_> keys,
                    std::span<admission_t, extent_> admissions) const noexcept {
        std::array<usearch_key_t, extent_> c_keys;
        std::array<usearch_admission_t, extent_> c_admissions;
        for (std::size_t lane = 0; lane != extent_; ++lane) c_keys[lane] = std::bit_cast<usearch_key_t>(keys[lane]);
        predicate.callback(c_keys.data(), c_admissions.data(), extent_, predicate.state);
        for (std::size_t lane = 0; lane != extent_; ++lane)
            admissions[lane] = static_cast<admission_t>(c_admissions[lane]);
    }
};

/** The C value lookup lifted onto the index's key vocabulary: one callback per scored batch, at
 *  the batch's static extent, a NULL row answering an empty view that refuses the call. A GPU walk
 *  takes it as bytes and calls it one key at a time on the device. */
struct value_lookup_t {
    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

    template <typename scalar_type_, std::size_t extent_>
    void operator()(std::span<us::utilities::uuid_t const, extent_> keys,
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
    void push(std::span<us::utilities::uuid_t const, extent_> asked,
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

/** Types in order, the variants one basic class unit compiles in one home. */
template <typename... types_>
struct type_list {
    static constexpr std::size_t size_k = sizeof...(types_);
};

/** Storage dtypes in ladder order, the one list every dense class unit folds its variants over. */
template <nk_dtype_t... dtypes_>
struct dtype_list {
    static constexpr std::size_t size_k = sizeof...(dtypes_);
    static constexpr std::array<nk_dtype_t, sizeof...(dtypes_)> dtypes_k {dtypes_...};

    /** The position of @p dtype in the list, or @c size_k when it is absent. */
    static constexpr std::size_t position_of(nk_dtype_t dtype) noexcept {
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
    case cpu_class_t::nvidia_vera_k: return numkong_capabilities_of(cpu_class_t::aws_g5_k) | nk_cap_neonfp8_k;
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
    case cpu_class_t::nvidia_vera_k: return cpu_class_t::aws_g5_k;
    case cpu_class_t::aws_g5_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m4_k: return cpu_class_t::apple_m2_k;
    case cpu_class_t::apple_m2_k: return cpu_class_t::aws_g2_k;
    case cpu_class_t::wasm64_extended_k: return cpu_class_t::wasm32_v128_k;
    default: return cpu_class_t::serial_k;
    }
}

/** The wait instruction one CPU class pairs: x86 spins PAUSE, the WAITPKG Xeons and the monitored
 *  Arm & RISC-V classes arm TPAUSE, WFET & WRS, WebAssembly threads sleep in @c memory.atomic.wait,
 *  everything else yields - as does any class whose @c FORKUNION_TARGET_<BIT> is off in the probe
 *  list. */
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
#if FORKUNION_TARGET_ARM64_WFET
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g5_k> {
    using type = fu::arm64_wfet_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    using type = fu::arm64_wfet_t;
};
#endif
#if FORKUNION_TARGET_RISCV64_WRS
template <>
struct waiter_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    using type = fu::risc5_wrs_t;
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
    using type = fu::risc5_zacas_atomic_ref<value_type_>;
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

/** The line every engine's prefetcher steps by: this machine's cache line, or
 *  @c default_alignment_k where no platform source names one. */
inline std::size_t machine_cache_line_bytes() noexcept {
    std::size_t const machine_line_bytes = fu::cache_line_bytes();
    return machine_line_bytes ? machine_line_bytes : default_alignment_k;
}

/** The spin hints every core of an architecture carries - PAUSE, YIELD, Zihintpause: the runtime
 *  confirms nothing about them, and a class differing by one alone earns no rung. */
inline constexpr fu::capabilities_t baseline_hints_k = fu::capability_x86_pause_k | fu::capability_arm64_yield_k |
                                                       fu::capability_risc5_pause_k;

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
    case cpu_class_t::serial_k: return true;
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
    default: return false;
    }
}

/** Whether every compute bit one CPU class lowers onto is present in @p available. */
inline bool numkong_admits(nk_capability_t available, cpu_class_t cpu_class) noexcept {
    nk_capability_t const wanted = numkong_capabilities_of(cpu_class);
    return (available & wanted) == wanted;
}

/** Whether the runtime confirms every ForkUnion bit one CPU class needs. */
template <cpu_class_t cpu_class_>
inline bool forkunion_admits() noexcept {
    constexpr fu::capabilities_t wanted = forkunion_capabilities_of<cpu_class_>();
    return (fu::runtime_capabilities() & wanted) == wanted;
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
    default: return forkunion_capabilities_of<cpu_class_t::serial_k>();
    }
}

/** Whether @p rung runs wherever @p pinned does: its NumKong and ForkUnion masks are subsets of
 *  the pinned class's, so a pin admits exactly the rungs a machine of that class would. */
constexpr bool cpu_class_within(cpu_class_t rung, cpu_class_t pinned) noexcept {
    nk_capability_t const rung_compute = numkong_capabilities_of(rung),
                          pinned_compute = numkong_capabilities_of(pinned);
    fu::capabilities_t const rung_sync = forkunion_capabilities_of(rung),
                             pinned_sync = forkunion_capabilities_of(pinned);
    return (rung_compute & pinned_compute) == rung_compute && (rung_sync & pinned_sync) == rung_sync;
}

/** The C enum's class as the engines name it; @c usearch_cpu_class_unknown_k derives nothing, so
 *  it names the serial floor. */
constexpr cpu_class_t cpu_class_of(usearch_cpu_class_t cpu_class) noexcept {
    if (cpu_class == usearch_cpu_class_unknown_k || cpu_class > usearch_cpu_class_wasm64_extended_k)
        return cpu_class_t::serial_k;
    return static_cast<cpu_class_t>(cpu_class);
}

/** Whether this machine runs @p pinned, the class a caller told an index to dispatch as: every
 *  NumKong bit in @p available and every ForkUnion bit the runtime confirms; no pin always runs. */
inline bool cpu_pin_admitted(usearch_cpu_class_t pinned, nk_capability_t available) noexcept {
    if (pinned == usearch_cpu_class_unknown_k) return true;
    if (pinned > usearch_cpu_class_wasm64_extended_k) return false;
    cpu_class_t const pinned_class = cpu_class_of(pinned);
    fu::capabilities_t const wanted = forkunion_capabilities_of(pinned_class);
    return numkong_admits(available, pinned_class) && (fu::runtime_capabilities() & wanted) == wanted;
}

/** Whether a ladder may take @p rung under @p pinned: every rung when nothing is pinned. */
constexpr bool cpu_pin_takes(usearch_cpu_class_t pinned, cpu_class_t rung) noexcept {
    return pinned == usearch_cpu_class_unknown_k || cpu_class_within(rung, cpu_class_of(pinned));
}

/** CPU classes newest first, the order a ladder tries them in. */
template <cpu_class_t... classes_>
struct cpu_ladder {};

/** GPU classes of one vendor newest first, the order its ladder tries them in. */
template <gpu_class_t... classes_>
struct gpu_ladder {};

/** This architecture's ladder, the serial floor last; a rung the build left out is skipped by the
 *  walk, never by the list. */
#if NUMKONG_ARCH_X8664_
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

/** One class's outcome row: its reference over the status and the finish count, its waiter between
 *  their loads. */
template <cpu_class_t cpu_class_>
outcome_verbs_t outcome_verbs_of() noexcept {
    using status_ref_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<usearch_status_t>;
    using status_reader_t = typename atomic_ref_for_cpu_class<cpu_class_>::template type<usearch_status_t const>;
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
        [](u32_t* count) noexcept { count_ref_t(*count).add(1, std::memory_order_release); },
        [](u32_t const* count, u32_t seen) noexcept {
            count_reader_t const finished {*count};
            while (finished.load(std::memory_order_acquire) == seen) wait_t {}(count, seen, 0u, fu::wait_capped_t {});
        },
    };
}

inline outcome_verbs_t const& host_outcome_verbs() noexcept {
    static outcome_verbs_t const picked = []<cpu_class_t... classes_>(cpu_ladder<classes_...>) noexcept {
        outcome_verbs_t row = outcome_verbs_of<cpu_class_t::serial_k>();
        bool found = false;
        auto rung = [&]<cpu_class_t class_>() noexcept {
            if constexpr (class_ != cpu_class_t::serial_k)
                if (!found && forkunion_admits<class_>()) row = outcome_verbs_of<class_>(), found = true;
        };
        (rung.template operator()<classes_>(), ...);
        return row;
    }(cpu_ladder_t {});
    return picked;
}

/** Whether the dense NVMe variants compiled a unit for @p cpu_class, as
 *  @c usearch_dense_nvme_classes_ in @c src/CMakeLists.txt lists them: a storage walk shares a few
 *  counters and never waits, so the classes advancing by their waiter or reference alone take none
 *  and their parent's unit serves, and a WebAssembly module reads no storage. */
constexpr bool nvme_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::aws_g5_k:
    case cpu_class_t::alibaba_c930_k:
    case cpu_class_t::wasm32_v128_k:
    case cpu_class_t::wasm64_extended_k: return false;
    default: return cpu_class_compiled(cpu_class);
    }
}

/** Whether the basic modalities compiled a unit for @p cpu_class, as @c usearch_basic_cpu_classes_
 *  in @c src/CMakeLists.txt lists them: the classes where a StringZilla or NumKong kit they inline
 *  advances. */
constexpr bool basic_class_compiled(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::serial_k:
    case cpu_class_t::intel_haswell_k:
    case cpu_class_t::intel_skylake_k:
    case cpu_class_t::aws_g2_k:
    case cpu_class_t::apple_m2_k:
    case cpu_class_t::alibaba_c930_k:
    case cpu_class_t::wasm64_extended_k: return cpu_class_compiled(cpu_class);
    default: return false;
    }
}

#pragma endregion CPU Classes

#pragma region Residencies

/** Residency discrimination off the alternative's own executor type - no second tag exists to
 *  disagree with the variant. */
template <typename multi_type_>
inline constexpr bool is_ram_index = std::is_same_v<typename multi_type_::executor_t, forkunion_executor_t>;

/** Whether an executor drives GPUs: every vendor's names the vendor it drives. */
template <typename executor_type_>
inline constexpr bool is_gpu_executor = requires { executor_type_::vendor_k; };

template <typename multi_type_>
inline constexpr bool is_vram_index = is_gpu_executor<typename multi_type_::executor_t>;

template <typename multi_type_>
inline constexpr bool is_nvme_index = std::is_same_v<typename multi_type_::executor_t, nvme_executor_t>;

/** Where a variant runs and which executor it takes, the dense and basic modalities alike. */
enum class dense_home_t : u32_t {
    ram_k,
    nvme_k,
    vram_nvidia_k,
    vram_amd_k,
    vram_apple_k,
};

/** The executor each home's verbs take. */
template <dense_home_t home_>
struct dense_home_executor;
template <>
struct dense_home_executor<dense_home_t::ram_k> {
    using type = forkunion_executor_t;
};
template <>
struct dense_home_executor<dense_home_t::nvme_k> {
    using type = nvme_executor_t;
};
template <>
struct dense_home_executor<dense_home_t::vram_nvidia_k> {
    using type = cuda_executor_t;
};
template <>
struct dense_home_executor<dense_home_t::vram_amd_k> {
    using type = hip_executor_t;
};
template <>
struct dense_home_executor<dense_home_t::vram_apple_k> {
    using type = metal_executor_t;
};

/** The home whose verbs take @p executor_type_. */
template <typename executor_type_>
inline constexpr dense_home_t home_of = std::is_same_v<executor_type_, forkunion_executor_t> ? dense_home_t::ram_k
                                        : std::is_same_v<executor_type_, nvme_executor_t>    ? dense_home_t::nvme_k
                                        : std::is_same_v<executor_type_, cuda_executor_t> ? dense_home_t::vram_nvidia_k
                                        : std::is_same_v<executor_type_, hip_executor_t>  ? dense_home_t::vram_amd_k
                                                                                          : dense_home_t::vram_apple_k;

/** Copies @p source into @p staged when it lives on an NVIDIA or AMD device, whose memory no host
 *  read reaches, and points @p source at the copy; host pages and Apple's unified memory are read
 *  in place. */
inline status_t stage_on_host(bytes_view_t& source, growing_buffer<byte_t>& staged) noexcept {
    int const nvidia = cuda_device_of(source.data()), amd = nvidia < 0 ? hip_device_of(source.data()) : -1;
    if (nvidia < 0 && amd < 0) return status_t::success_k;
    if (failed(staged.resize_uninitialized(source.size()))) return status_t::bad_alloc_k;
    bool const copied = nvidia >= 0 ? cuda_set_device(nvidia) == cuda_success_k &&
                                          cuda_copy(staged.data(), source.data(), source.size()) == cuda_success_k
                                    : hip_set_device(amd) == hip_success_k &&
                                          hip_api().copy(staged.data(), source.data(), source.size(),
                                                         hip_copy_device_to_host_k) == hip_success_k;
    if (!copied) return status_t::device_error_k;
    source = to_span(staged);
    return status_t::success_k;
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

// The device listing crosses the ABI by casts, so the C vendors and classes mirror the engine's.
static_assert(static_cast<u32_t>(gpu_vendor_t::apple_k) == usearch_gpu_vendor_apple_k);
static_assert(static_cast<u32_t>(gpu_class_t::apple_metal10_k) == usearch_gpu_class_apple_metal10_k);

#pragma endregion Residencies

#pragma region Calls

/** Every argument structure that names an executor, so a deferred submission keeps its call by
 *  value; whatever the copy points to stays the caller's until final. */
using deferred_arguments_t = std::variant<                                                               //
    usearch_dense_load_ram_t, usearch_dense_load_ram_buffer_t, usearch_dense_view_nvme_t,                //
    usearch_dense_load_vram_t, usearch_dense_load_vram_buffer_t,                                         //
    usearch_dense_reserve_t, usearch_dense_clear_t, usearch_dense_insert_t,                              //
    usearch_dense_refine_t, usearch_dense_health_t, usearch_dense_absorb_t, usearch_dense_split_t,       //
    usearch_dense_refinement_init_t, usearch_dense_refinement_next_t, usearch_dense_refinement_finish_t, //
    usearch_dense_search_t, usearch_dense_count_t, usearch_dense_get_t,                                  //
    usearch_dense_search_stream_next_t, usearch_dense_cursor_next_t, usearch_dense_sample_t,             //
    usearch_dense_save_t, usearch_dense_save_buffer_t,                                                   //
    usearch_dense_metadata_read_t, usearch_dense_metadata_read_buffer_t, usearch_dense_describe_t,       //
    usearch_dense_multiply_t, usearch_dense_attend_t, usearch_dense_maxsim_t,                            //
    usearch_sequence_load_ram_t, usearch_sequence_load_vram_t, usearch_sequence_view_nvme_t,             //
    usearch_sequence_reserve_t, usearch_sequence_clear_t, usearch_sequence_insert_t,                     //
    usearch_sequence_refine_t, usearch_sequence_absorb_t, usearch_sequence_split_t,                      //
    usearch_sequence_refinement_init_t, usearch_sequence_refinement_next_t,                              //
    usearch_sequence_refinement_finish_t,                                                                //
    usearch_sequence_search_t, usearch_sequence_count_t, usearch_sequence_get_t,                         //
    usearch_sequence_cursor_next_t, usearch_sequence_sample_t,                                           //
    usearch_sequence_save_t, usearch_sequence_metadata_read_t, usearch_sequence_describe_t,              //
    usearch_shapes_load_ram_t, usearch_shapes_load_vram_t, usearch_shapes_view_nvme_t,                   //
    usearch_shapes_reserve_t, usearch_shapes_clear_t, usearch_shapes_insert_t,                           //
    usearch_shapes_refine_t, usearch_shapes_absorb_t, usearch_shapes_split_t,                            //
    usearch_shapes_refinement_init_t, usearch_shapes_refinement_next_t,                                  //
    usearch_shapes_refinement_finish_t,                                                                  //
    usearch_shapes_search_t, usearch_shapes_count_t, usearch_shapes_get_t,                               //
    usearch_shapes_cursor_next_t, usearch_shapes_sample_t,                                               //
    usearch_shapes_save_t, usearch_shapes_metadata_read_t, usearch_shapes_describe_t,                    //
    usearch_geospatial_load_ram_t, usearch_geospatial_load_vram_t, usearch_geospatial_view_nvme_t,       //
    usearch_geospatial_reserve_t, usearch_geospatial_clear_t, usearch_geospatial_insert_t,               //
    usearch_geospatial_refine_t, usearch_geospatial_absorb_t, usearch_geospatial_split_t,                //
    usearch_geospatial_refinement_init_t, usearch_geospatial_refinement_next_t,                          //
    usearch_geospatial_refinement_finish_t,                                                              //
    usearch_geospatial_search_t, usearch_geospatial_within_t,                                            //
    usearch_geospatial_count_t, usearch_geospatial_get_t,                                                //
    usearch_geospatial_cursor_next_t, usearch_geospatial_sample_t,                                       //
    usearch_geospatial_save_t, usearch_geospatial_metadata_read_t, usearch_geospatial_describe_t>;

/** Where one slot of a deferred table stands. */
enum class deferred_stage_t : unsigned char {

    /** Nothing in it; a submission may take it. */
    free_k,

    /** A submitted call waits in it for its wait, a drain, a blocking call or a free. */
    held_k,

    /** Some thread took the call and runs it. */
    running_k,
};

/** One slot of an executor's deferred table: the copied call, the verb that runs it, and its place
 *  in line. */
struct deferred_call_t {
    deferred_arguments_t arguments;
    void (*verb)(void*) = nullptr;

    /** Submission order, counted from one across every executor. */
    std::size_t ticket = 0;
    deferred_stage_t stage = deferred_stage_t::free_k;
};

/** Slots a deferred table holds when its constructor names no limit. */
inline constexpr std::size_t deferred_limit_default_k = 64;

/** Reads every NVMe engine keeps in flight: deep enough to cover a device's latency, which the C
 *  interface leaves to the library. */
inline constexpr std::size_t nvme_queue_depth_k = 64;

/** The state behind @c usearch_executor_t: the discovered topology a RAM or NVMe executor was
 *  spawned over, its deferred table, and one residency's executor; the variant is the kind tag, so
 *  no second word can disagree. The executor is declared last, so its pool dies before the
 *  topology. */
struct usearch_executor_handle_t {
    std::unique_ptr<void, decltype(&fu_topology_delete)> topology {nullptr, &fu_topology_delete};

    /** One slot per call it tracks at once, sized at creation, so a submission never allocates. */
    growing_buffer<deferred_call_t> deferred {global_allocator};

    /** The next listed executor, so a wait or a free finds the calls naming its handle. */
    usearch_executor_handle_t* next_listed = nullptr;

    std::variant<forkunion_executor_t, nvme_executor_t, gpu_executor_t> executor;
};

/** Requests AMX tile state alone, granted process-wide on Linux; more bits would rewrite FPCR. */
inline void request_tile_state() noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k || !(available & nk_cap_sapphireamx_k)) return;
    [[maybe_unused]] nk_status_t const requested = nk_cpu_configure_thread(nk_cap_sapphireamx_k);
}

/** Discovers the topology and spawns @p threads over it into @p handle, answering @p handle, or
 *  null once the outcome records why not; one thread is the caller. */
inline usearch_executor_t spawn_ram_executor(usearch_executor_handle_t& handle, std::size_t threads,
                                             usearch_outcome_t* outcome) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_status_device_k, outcome, "Topology discovery failed"), nullptr;
    handle.topology.reset(topology);
    auto [spawned, status] = forkunion_executor_t::spawn(topology, threads);
    if (failed(status)) return propagate_error(status, outcome), nullptr;
    request_tile_state();
    handle.executor = std::move(spawned);
    return &handle;
}

/** Discovers the topology and spawns @p threads into @p handle, each opening the engine @p config
 *  names, answering @p handle, or null once the outcome records why not. */
inline usearch_executor_t spawn_nvme_executor(usearch_executor_handle_t& handle, std::size_t threads,
                                              io_engine_config_t const& config, usearch_outcome_t* outcome) noexcept {
    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k)
        return propagate_error(usearch_status_device_k, outcome, "Topology discovery failed"), nullptr;
    handle.topology.reset(topology);
    auto [spawned, status] = nvme_executor_t::spawn(topology, threads, config);
    if (failed(status))
        return propagate_error(status, outcome, "No I/O engine opened on the executor's threads"), nullptr;
    request_tile_state();
    handle.executor = std::move(spawned);
    return &handle;
}

/** The executor a null handle runs RAM work on: the calling thread, built into @p call_executor. */
template <typename multi_type_>
    requires is_ram_index<multi_type_>
inline usearch_executor_t call_executor_for(multi_type_&, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    return spawn_ram_executor(call_executor, 1, outcome);
}

/** The executor a null handle runs NVMe work on: one thread of synchronous reads. */
template <typename multi_type_>
    requires is_nvme_index<multi_type_>
inline usearch_executor_t call_executor_for(multi_type_&, usearch_executor_handle_t& call_executor,
                                            usearch_outcome_t* outcome) noexcept {
    io_engine_config_t config;
    config.queue_depth = nvme_queue_depth_k;
    config.transport = io_transport_t::blocking_k;
    return spawn_nvme_executor(call_executor, 1, config, outcome);
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

/** The device an arena of host pages names, which no GPU ordinal does. */
inline constexpr int host_device_k = -1;

/** Whether @p traversal names a walk, a scan, or nothing, which derives the walk, and @p predicate
 *  leaves @p device 's walks alone, since a host callback cannot narrow a device walk; false once
 *  @p outcome records why not. */
inline bool accepts_traversal(int device, usearch_traversal_t traversal, usearch_predicate_t predicate,
                              usearch_outcome_t* outcome) noexcept {
    if (traversal != usearch_traversal_unknown_k && traversal != usearch_traversal_walk_k &&
        traversal != usearch_traversal_scan_k)
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown traversal");
    if (predicate.callback && device != host_device_k)
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

    /** The GPU ordinal the memory lives on, or @c host_device_k. */
    int device = host_device_k;

    /** The outcome of the call holding the arena, from its claim or deferred submission until just
     *  before its final store; touched through a reference, since a held call finishes on whichever
     *  thread runs it. */
    usearch_outcome_t const* holder = nullptr;
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

/** Whether @p address lies in memory @p arena handed out, which its next rewind reuses or frees. */
inline bool arena_contains(usearch_arena_handle_t const& arena, void const* address) noexcept {
    return arena.device_memory ? arena.device_memory.contains(address) : arena.memory.contains(address);
}

/** Holds @p arena for the call whose outcome is @p call and rewinds it, freeing the previous call's
 *  outputs, so an input of @p inputs still inside it is refused, as is an arena on another device
 *  than @p device or device memory another @p executor made. The call's @c call_outcome_t lets go
 *  of it before the final store. */
inline usearch_arena_handle_t* claim_arena(usearch_arena_t arena, usearch_outcome_t const* call,
                                           usearch_executor_t executor, int device,
                                           std::initializer_list<void const*> inputs,
                                           usearch_outcome_t* outcome) noexcept {
    if (!arena)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The arena must not be null"), nullptr;
    if (arena->device != device)
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
    value_type_* const landed = carve<value_type_>(arena.device_memory, count);
    if (!landed) return {nullptr, status_t::bad_alloc_k};
    return {landed, arena.device_memory.upload(landed, output, count * sizeof(value_type_))};
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

/**
 *  @brief A verb's outcome while it runs: the caller's reads deferred from the start, every step
 *      records into a private one, and the end lets go of the arena before the one final store.
 *
 *  Declared first in a verb, it unwinds last, so a waiter that sees the final status may reuse the
 *  arena at once. A path that forgot to report finishes @c usearch_status_internal_k rather than
 *  hanging its waiters.
 */
class call_outcome_t {
  public:
    explicit call_outcome_t(usearch_outcome_t* published, usearch_arena_t arena = nullptr) noexcept
        : published_(published), arena_(arena) {
        propagate_pending(published_);
    }
    ~call_outcome_t() noexcept {
        release_arena(arena_, published_);
        propagate_error(recorded_.status, published_, recorded_.message);
    }
    call_outcome_t(call_outcome_t const&) = delete;
    call_outcome_t& operator=(call_outcome_t const&) = delete;

    /** The private record every step reports into. */
    operator usearch_outcome_t*() noexcept { return &recorded_; }

  private:
    usearch_outcome_t* published_;
    usearch_arena_t arena_;
    usearch_outcome_t recorded_ {usearch_status_internal_k, "The call finished without reporting a status"};
};

/** Where a ranked run lands inside a claimed arena: @c wanted keys and scores per row, and a count
 *  per row. */
struct matches_landing_t {
    usearch_key_t* keys = nullptr;
    usearch_distance_t* scores = nullptr;
    std::size_t* counts = nullptr;
    std::size_t wanted = 0;

    /** The view the call publishes once every row landed. */
    usearch_matches_view_t view() const noexcept { return {keys, scores, counts, wanted}; }
};

/** The landing of @p count rows of @p wanted matches, carved out of a claimed arena's host pages or
 *  device memory, @p memory; a zero @p wanted carves only the counts. */
template <typename memory_type_>
inline expected<matches_landing_t> carve_matches(memory_type_& memory, std::size_t count, std::size_t wanted) noexcept {
    matches_landing_t landing;
    landing.wanted = wanted;
    landing.counts = carve<std::size_t>(memory, count);
    if (wanted) {
        landing.keys = carve<usearch_key_t>(memory, count * wanted);
        landing.scores = carve<usearch_distance_t>(memory, count * wanted);
    }
    bool const carved = landing.counts && (!wanted || (landing.keys && landing.scores));
    return {landing, carved ? status_t::success_k : status_t::bad_alloc_k};
}

/** The @p count rows @ref carve_matches put in @p landing as the caller reads them, where @p arena
 *  hands outputs out, once its uploads drained. */
inline expected<usearch_matches_view_t> publish_matches(usearch_arena_handle_t& arena, matches_landing_t const& landing,
                                                        std::size_t count) noexcept {
    std::size_t const entries = count * landing.wanted;
    auto const [counts, counts_status] = publish(arena, landing.counts, count);
    auto const [keys, keys_status] = publish(arena, landing.keys, entries);
    auto const [scores, scores_status] = publish(arena, landing.scores, entries);
    status_t const landed = drained(arena, first_failure({counts_status, keys_status, scores_status}));
    return {{keys, scores, counts, landing.wanted}, landed};
}

/** The ragged run of @p count rows @ref carve put on host pages, @p keys and @p scores cut by
 *  @p offsets, as the caller reads it where @p arena hands outputs out, once its uploads drained;
 *  an empty run hands out no keys or scores. */
inline expected<usearch_matches_tape_view_t> publish_matches(usearch_arena_handle_t& arena, usearch_key_t* keys,
                                                             usearch_distance_t* scores, std::size_t* offsets,
                                                             std::size_t count) noexcept {
    std::size_t const written = offsets[count];
    auto const [landed_keys, keys_status] = publish(arena, written ? keys : nullptr, written);
    auto const [landed_scores, scores_status] = publish(arena, written ? scores : nullptr, written);
    auto const [landed_offsets, offsets_status] = publish(arena, offsets, count + 1);
    status_t const landed = drained(arena, first_failure({keys_status, scores_status, offsets_status}));
    return {{landed_keys, landed_scores, landed_offsets}, landed};
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

#pragma region Deferred Calls

/** Every listed executor and the lock every deferred table is read under, so a wait or a free finds
 *  the calls naming its handle without asking each executor. */
struct deferred_registry_t {
    std::mutex guard;

    /** Counted up as each taken call becomes final, under @c guard; waiters read it unlocked. */
    u32_t finished = 0;

    /** The last ticket handed out. */
    std::size_t tickets = 0;

    /** The newest listed executor. */
    usearch_executor_handle_t* executors = nullptr;
};

/** The one registry every modality's calls list in, defined beside the executor verbs. */
extern deferred_registry_t deferred_registry;

/** The held call whose arguments this thread passes back into their verb, which then runs them in
 *  place rather than holding them again. */
extern thread_local void const* resuming_call;

/** Sizes @p executor 's deferred table at @p limit slots, zero taking the default, and lists it. */
inline status_t list_executor(usearch_executor_handle_t& executor, std::size_t limit) noexcept {
    if (failed(executor.deferred.resize(limit ? limit : deferred_limit_default_k))) return status_t::bad_alloc_k;
    std::lock_guard<std::mutex> lock(deferred_registry.guard);
    executor.next_listed = deferred_registry.executors;
    deferred_registry.executors = &executor;
    return status_t::success_k;
}

/** Takes @p executor off the list; the caller already finished every call it held. */
inline void delist_executor(usearch_executor_handle_t& executor) noexcept {
    std::lock_guard<std::mutex> lock(deferred_registry.guard);
    for (usearch_executor_handle_t** link = &deferred_registry.executors; *link; link = &(*link)->next_listed)
        if (*link == &executor) {
            *link = executor.next_listed;
            return;
        }
}

/** Whether the call in @p slot reads or writes @p handle: its outcome, its executor, or an index,
 *  cursor, stream or arena it names. An output slot for a handle the call creates holds nothing
 *  yet. */
inline bool deferred_uses(deferred_call_t const& slot, void const* handle) noexcept {
    return std::visit(
        [&](auto const& arguments) noexcept {
            auto const names = [&](auto named) noexcept {
                if constexpr (std::is_pointer_v<std::remove_pointer_t<decltype(named)>>) return false;
                else return static_cast<void const*>(named) == handle;
            };
            bool used = static_cast<void const*>(arguments.outcome) == handle ||
                        static_cast<void const*>(arguments.executor) == handle;
            if constexpr (requires { arguments.index; }) used = used || names(arguments.index);
            if constexpr (requires { arguments.cursor; }) used = used || names(arguments.cursor);
            if constexpr (requires { arguments.refinement; }) used = used || names(arguments.refinement);
            if constexpr (requires { arguments.stream; }) used = used || names(arguments.stream);
            if constexpr (requires { arguments.arena; }) used = used || names(arguments.arena);
            return used;
        },
        slot.arguments);
}

/** Runs every call @p executor holds with a ticket up to @p last on this thread, oldest first. */
inline void run_held(usearch_executor_handle_t& executor, std::size_t last) noexcept {
    for (;;) {
        deferred_call_t* oldest = nullptr;
        {
            std::lock_guard<std::mutex> lock(deferred_registry.guard);
            for (deferred_call_t& slot : executor.deferred)
                if (slot.stage == deferred_stage_t::held_k && slot.ticket <= last &&
                    (!oldest || slot.ticket < oldest->ticket))
                    oldest = &slot;
            if (!oldest) return;
            oldest->stage = deferred_stage_t::running_k;
        }
        std::visit(
            [&](auto& arguments) noexcept {
                resuming_call = &arguments;
                oldest->verb(&arguments);
            },
            oldest->arguments);
        std::lock_guard<std::mutex> lock(deferred_registry.guard);
        oldest->stage = deferred_stage_t::free_k;
        host_outcome_verbs().count_finished(&deferred_registry.finished);
    }
}

/** Waits until no call @p executor took with a ticket up to @p last still runs on another
 *  thread. */
inline void wait_running(usearch_executor_handle_t& executor, std::size_t last) noexcept {
    for (;;) {
        u32_t seen = 0;
        {
            std::lock_guard<std::mutex> lock(deferred_registry.guard);
            bool running = false;
            for (deferred_call_t const& slot : executor.deferred)
                running = running || (slot.stage == deferred_stage_t::running_k && slot.ticket <= last);
            if (!running) return;
            seen = deferred_registry.finished;
        }
        host_outcome_verbs().wait_finished(&deferred_registry.finished, seen);
    }
}

/** Returns once no deferred call reads or writes @p handle, running the held ones on this thread
 *  together with every call held before them on the same executor - what every wait and every free
 *  does first. */
inline void finish_calls_using(void const* handle) noexcept {
    if (!handle) return;
    std::unique_lock<std::mutex> lock(deferred_registry.guard);
    for (;;) {
        usearch_executor_handle_t* holding = nullptr;
        std::size_t last = 0, running = 0;
        for (usearch_executor_handle_t* executor = deferred_registry.executors; executor && !holding;
             executor = executor->next_listed)
            for (deferred_call_t const& slot : executor->deferred)
                if (slot.stage == deferred_stage_t::held_k && deferred_uses(slot, handle))
                    holding = executor, last = max_of(last, slot.ticket);
                else if (slot.stage == deferred_stage_t::running_k && deferred_uses(slot, handle)) ++running;
        if (holding) {
            // Running a held call takes this lock itself, so it runs outside it.
            lock.unlock();
            run_held(*holding, last);
            lock.lock();
        }
        else if (running) {
            u32_t const seen = deferred_registry.finished;
            lock.unlock();
            host_outcome_verbs().wait_finished(&deferred_registry.finished, seen);
            lock.lock();
        }
        else return;
    }
}

/**
 *  @brief Every verb's entry. A deferred call on an executor is copied into the executor's table,
 *      holding its arena, and answers true; a blocking one first runs the calls the executor holds
 *      and answers false.
 *
 *  A call the runner of a held call passes back answers false at once, so the verb runs it in
 *  place, and so does a call with no executor, which is final when it returns whatever its
 *  completion says.
 */
template <auto verb_, typename arguments_type_>
inline bool submitted(arguments_type_* arguments) noexcept {
    if (resuming_call == arguments) return resuming_call = nullptr, false;
    usearch_executor_t const executor = arguments->executor;
    if (!executor) return false;
    if (arguments->completion != usearch_completion_deferred_k) return run_held(*executor, ~std::size_t {0}), false;
    usearch_outcome_t* const outcome = arguments->outcome;
    propagate_pending(outcome);
    usearch_arena_t arena = nullptr;
    if constexpr (requires { arguments->arena; }) arena = arguments->arena;
    if (!hold_arena(arena, outcome))
        return propagate_error(usearch_status_contended_k, outcome, "The arena serves another call"), true;
    std::lock_guard<std::mutex> lock(deferred_registry.guard);
    for (deferred_call_t& slot : executor->deferred)
        if (slot.stage == deferred_stage_t::free_k) {
            slot.arguments.template emplace<arguments_type_>(*arguments);
            slot.verb = [](void* erased) { verb_(static_cast<arguments_type_*>(erased)); };
            slot.ticket = ++deferred_registry.tickets;
            slot.stage = deferred_stage_t::held_k;
            return true;
        }
    // A refused submission never runs, so nothing else would let go of the arena it took.
    release_arena(arena, outcome);
    propagate_error(usearch_status_contended_k, outcome, "The deferred table is full");
    return true;
}

/** Lists a spawned @p executor with a deferred table of @p deferred_limit slots and hands it out
 *  through @p output, or frees it once @p outcome records why not. */
inline void hand_out_executor(usearch_executor_t executor, std::size_t deferred_limit, usearch_executor_t* output,
                              usearch_outcome_t* outcome) noexcept {
    if (failed(list_executor(*executor, deferred_limit)))
        return delete executor,
               propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the deferred table"),
               void();
    *output = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Deferred Calls
