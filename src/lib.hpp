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

#include <cassert> // `assert` - a predicate judges a quad at a time
#include <cstddef> // `std::size_t`

#include <array>              // `std::array` - the predicate's batch
#include <atomic>             // `std::atomic`, `std::atomic_ref` - the arena's holder and the outcome's status word
#include <bit>                // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <condition_variable> // `std::condition_variable` - waiting out deferred calls other threads run
#include <initializer_list>   // `std::initializer_list` - the inputs a call checks against its arena
#include <memory>             // `std::unique_ptr` - the executor's topology
#include <mutex>              // `std::lock_guard` over the ticks' lock and the deferred tables
#include <span>               // `std::span` - key batches crossing the ABI
#include <string_view>        // `std::string_view` - checkpoint paths
#include <type_traits>        // `std::is_same_v` - the residency discrimination
#include <variant>            // `std::variant`, `std::visit`

#include <forkunion.h>                // `fu_topology_new` - the executor factories' discovery
#include <forkunion/atomics.hpp>      // `fu::standard_atomic_ref` & kin - the classes' atomic references
#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/types.hpp>        // `fu::spin_mutex_t` - the ticks' lock
#include <numkong/capabilities.h>     // `nk_capability_t` - the CPU classes' compute masks

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

/** Codenames of the dispatchable CPU tiers, composed into the capability string once. */
struct numkong_capability_name_t {
    nk_capability_t bit;
    char const* name;
};
inline constexpr numkong_capability_name_t numkong_capability_names_k[] = {
    {nk_cap_serial_k, "serial"},
    {nk_cap_neon_k, "neon"},
    {nk_cap_neonsdot_k, "neonsdot"},
    {nk_cap_neonbfdot_k, "neonbfdot"},
    {nk_cap_neonfhm_k, "neonfhm"},
    {nk_cap_neonfp8_k, "neonfp8"},
    {nk_cap_sme_k, "sme"},
    {nk_cap_smebi32_k, "smebi32"},
    {nk_cap_haswell_k, "haswell"},
    {nk_cap_skylake_k, "skylake"},
    {nk_cap_icelake_k, "icelake"},
    {nk_cap_genoa_k, "genoa"},
    {nk_cap_sapphire_k, "sapphire"},
    {nk_cap_sapphireamx_k, "sapphireamx"},
    {nk_cap_graniteamx_k, "graniteamx"},
    {nk_cap_alder_k, "alder"},
    {nk_cap_sierra_k, "sierra"},
    {nk_cap_diamond_k, "diamond"},
    {nk_cap_rvv_k, "rvv"},
    {nk_cap_rvvbb_k, "rvvbb"},
    {nk_cap_v128_k, "v128"},
    {nk_cap_v128relaxed_k, "v128relaxed"},
};

/** Marks @p outcome as running; no waiter watches for this store, so it is relaxed. */
inline void propagate_pending(usearch_outcome_t* outcome) noexcept {
    outcome->message = nullptr;
    std::atomic_ref<usearch_status_t> {outcome->status}.store(usearch_status_deferred_k, std::memory_order_relaxed);
}

/** The one final write to @p outcome: the message, then the status every waiter acquires. The call
 *  touches nothing it was passed after this, and the message is null exactly on success. */
inline bool propagate_error(usearch_status_t status, usearch_outcome_t* outcome, char const* message) noexcept {
    bool const succeeded = status == usearch_status_success_k;
    outcome->message = succeeded ? nullptr : message;
    std::atomic_ref<usearch_status_t> final_status {outcome->status};
    final_status.store(status, std::memory_order_release);
    final_status.notify_all();
    return succeeded;
}

/** The engine-side overload: the C++ mirror shares the ABI's values, so crossing is a cast, and an
 *  unnamed @p message falls back to the status's own static text. */
inline bool propagate_error(status_t status, usearch_outcome_t* outcome, char const* message = nullptr) noexcept {
    return propagate_error(static_cast<usearch_status_t>(status), outcome,
                           message ? message : status_to_string(status));
}

/** Adds one fan-out's telemetry into the caller's stats, whichever modality's they are -
 *  accumulating, so chunked verbs report the whole call rather than the last chunk. */
template <typename stats_type_, typename result_type_>
inline void accumulate_stats(stats_type_* stats, result_type_ const& result) noexcept {
    if (!stats) return;
    stats->visited_members += result.visited_members;
    stats->computed_distances += result.computed_distances;
    stats->restarts += result.restarts;
}

/** Whether a result carries traversal telemetry - one probe both crossings read. */
template <typename value_type_>
concept carries_telemetry = requires(value_type_ const& value) { value.visited_members; };

/** The one crossing for a fallible step: folds telemetry into @p stats when the payload carries
 *  any, finishes @p outcome only on failure, and answers whether @c .value is readable. */
template <typename stats_type_, typename value_type_>
inline bool crossed(expected<value_type_> const& result, stats_type_* stats, usearch_outcome_t* outcome,
                    char const* message = nullptr) noexcept {
    if constexpr (carries_telemetry<value_type_>) accumulate_stats(stats, result.value);
    return result || propagate_error(result.status, outcome, message);
}

/** The C key struct and the engine's @c utilities::uuid_t share sixteen bytes - one bit cast in. */
inline us::utilities::uuid_t uuid_of(usearch_key_t key) noexcept { return std::bit_cast<us::utilities::uuid_t>(key); }

/** The C predicate lifted onto the index's key vocabulary, judging every quad the walk scores in
 *  one call through buffers of the C types. */
struct key_predicate_t {
    usearch_predicate_t predicate;

    void operator()(std::span<us::utilities::uuid_t const> keys, std::span<admission_t> admissions) const noexcept {
        if (keys.empty()) return;
        std::array<usearch_key_t, 4> c_keys;
        std::array<usearch_admission_t, 4> c_admissions;
        assert(keys.size() <= c_keys.size() && "Walks judge a quad at a time");
        for (std::size_t index = 0; index != keys.size(); ++index)
            c_keys[index] = std::bit_cast<usearch_key_t>(keys[index]);
        predicate.callback(c_keys.data(), c_admissions.data(), keys.size(), predicate.state);
        for (std::size_t index = 0; index != keys.size(); ++index)
            admissions[index] = static_cast<admission_t>(c_admissions[index]);
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
 *  arm must be distinct from to earn instantiation. On x86 the AVX-512 P-cores and the AVX-VNNI
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
 *  Arm & RISC-V classes arm TPAUSE, WFET & WRS, everything else yields - as does any class whose
 *  @c FORKUNION_TARGET_<BIT> is off in the probe list. */
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
#if FORKUNION_TARGET_RISC5_WRS
template <>
struct waiter_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    using type = fu::risc5_wrs_t;
};
#endif

/** The waiter a CPU class pairs, spelled short for the arms' explicit instantiations. */
template <cpu_class_t cpu_class_>
using waiter_of = typename waiter_for_cpu_class<cpu_class_>::type;

/** The atomic reference one CPU class pairs: single-instruction LSE read-modify-writes under RCpc
 *  acquiring loads on the Armv8.3+ classes, conditional adds on the CMPCCXADD Xeons, @c amocas on
 *  Zacas RISC-V - the floors, the older classes, and any class whose @c FORKUNION_TARGET_<BIT> is
 *  off in the probe list take the standard one. */
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
#if FORKUNION_TARGET_RISC5_ZACAS
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    template <typename value_type_>
    using type = fu::risc5_zacas_atomic_ref<value_type_>;
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

/** The vendor a VRAM arm's devices belong to, and @c unknown_k for every other residency. */
template <typename multi_type_>
constexpr gpu_vendor_t gpu_vendor_of_index() noexcept {
    if constexpr (is_vram_index<multi_type_>) return multi_type_::vendor_k;
    else return gpu_vendor_t::unknown_k;
}

template <typename multi_type_>
inline constexpr bool is_nvme_index = std::is_same_v<typename multi_type_::executor_t, nvme_executor_t>;

/** Visits the arm of @p index with @p mutate when it is writable; an NVMe view serves its
 *  checkpoint in place, so it refuses every mutation with @c usearch_status_read_only_k. */
template <typename handle_type_, typename mutate_type_>
void visit_writable(handle_type_& index, usearch_outcome_t* outcome, mutate_type_&& mutate) noexcept {
    std::visit(
        [&]<typename variant_type_>(variant_type_& variant) noexcept {
            if constexpr (is_nvme_index<variant_type_>)
                propagate_error(usearch_status_read_only_k, outcome, "An NVMe view never mutates");
            else mutate(variant);
        },
        index.variants);
}

/** Whether @p alternative_type_ is listed in @p variant_type_ - the membership fact the ladder
 *  asserts against the explicit list. */
template <typename alternative_type_, typename variant_type_>
inline constexpr bool variant_carries = false;
template <typename alternative_type_, typename... alternative_types_>
inline constexpr bool variant_carries<alternative_type_, std::variant<alternative_types_...>> =
    (std::is_same_v<alternative_type_, alternative_types_> || ...);

#pragma endregion Residencies

#pragma region Calls

/** Every argument structure that names an executor, so a deferred submission keeps its call by
 *  value; whatever the copy points to stays the caller's until final. */
using deferred_arguments_t = std::variant<
    usearch_dense_load_ram_t, usearch_dense_load_vram_t, usearch_dense_load_ram_buffer_t,
    usearch_dense_load_vram_buffer_t, usearch_dense_view_nvme_t, usearch_dense_reserve_t, usearch_dense_clear_t,
    usearch_dense_insert_t, usearch_dense_refine_t, usearch_dense_health_t, usearch_dense_search_t,
    usearch_dense_count_t, usearch_dense_get_t, usearch_dense_merge_t, usearch_dense_cursor_next_t,
    usearch_dense_search_stream_next_t, usearch_dense_save_t, usearch_dense_save_buffer_t,
    usearch_dense_metadata_read_t, usearch_dense_metadata_read_buffer_t, usearch_dense_describe_t,
    usearch_dense_multiply_t, usearch_dense_attend_t, usearch_dense_maxsim_t,
    usearch_dense_sample_t, //
    usearch_sequence_load_ram_t, usearch_sequence_view_nvme_t, usearch_sequence_reserve_t, usearch_sequence_clear_t,
    usearch_sequence_insert_t, usearch_sequence_refine_t, usearch_sequence_search_t, usearch_sequence_count_t,
    usearch_sequence_get_t, usearch_sequence_merge_t, usearch_sequence_cursor_next_t, usearch_sequence_sample_t,
    usearch_sequence_save_t, usearch_sequence_metadata_read_t, usearch_sequence_describe_t, //
    usearch_shapes_load_ram_t, usearch_shapes_load_vram_t, usearch_shapes_view_nvme_t, usearch_shapes_reserve_t,
    usearch_shapes_clear_t, usearch_shapes_insert_t, usearch_shapes_refine_t, usearch_shapes_search_t,
    usearch_shapes_count_t, usearch_shapes_get_t, usearch_shapes_merge_t, usearch_shapes_cursor_next_t,
    usearch_shapes_sample_t, usearch_shapes_save_t, usearch_shapes_metadata_read_t,
    usearch_shapes_describe_t, //
    usearch_geospatial_load_ram_t, usearch_geospatial_load_vram_t, usearch_geospatial_reserve_t,
    usearch_geospatial_clear_t, usearch_geospatial_insert_t, usearch_geospatial_refine_t, usearch_geospatial_search_t,
    usearch_geospatial_within_t, usearch_geospatial_count_t, usearch_geospatial_get_t, usearch_geospatial_merge_t,
    usearch_geospatial_cursor_next_t, usearch_geospatial_sample_t, usearch_geospatial_save_t,
    usearch_geospatial_metadata_read_t, usearch_geospatial_describe_t>;

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
 *  @c call_executor_for its residency names, the VRAM one defined beside the dense arms. */
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

/** What @c usearch_arena_t owns: memory that one call at a time rewinds and refills, as host pages
 *  or on one device, and the call holding it. */
struct usearch_arena_handle_t {
    growing_arena<forkunion_allocator_t> memory {forkunion_allocator_t {no_memory_domain_k}};

    /** Device memory on the first device of the executor the arena was made for, or none. */
    gpu_arena_t device_memory;

    /** The GPU ordinal the memory lives on, or @c host_device_k. */
    int device = host_device_k;

    /** The outcome of the call holding the arena, from its claim or deferred submission until just
     *  before its final store; atomic, since a held call finishes on whichever thread runs it. */
    std::atomic<usearch_outcome_t const*> holder {nullptr};
};

/** Holds @p arena for the call whose outcome is @p call, answering whether it was free or already
 *  that call's; a null arena holds nothing, left for the verb to refuse. */
inline bool hold_arena(usearch_arena_t arena, usearch_outcome_t const* call) noexcept {
    if (!arena) return true;
    usearch_outcome_t const* holder = nullptr;
    return arena->holder.compare_exchange_strong(holder, call, std::memory_order_acquire) || holder == call;
}

/** Lets go of @p arena when the call whose outcome is @p call holds it. */
inline void release_arena(usearch_arena_t arena, usearch_outcome_t const* call) noexcept {
    if (arena) arena->holder.compare_exchange_strong(call, nullptr, std::memory_order_release);
}

/** Whether @p address lies in memory @p arena handed out, which its next rewind reuses or frees. */
inline bool arena_contains(usearch_arena_handle_t const& arena, void const* address) noexcept {
    return arena.device_memory ? arena.device_memory.contains(address) : arena.memory.contains(address);
}

/** Holds @p arena for the call whose outcome is @p call and rewinds it, freeing the previous call's
 *  outputs, so an input of @p inputs still inside it is refused, as is an arena on another device
 *  than @p device. The call's @c call_outcome_t lets go of it before the final store. */
inline usearch_arena_handle_t* claim_arena(usearch_arena_t arena, usearch_outcome_t const* call, int device,
                                           std::initializer_list<void const*> inputs,
                                           usearch_outcome_t* outcome) noexcept {
    if (!arena)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The arena must not be null"), nullptr;
    if (arena->device != device)
        return propagate_error(usearch_status_device_k, outcome, "The arena lives where the call does not compute"),
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

/** Hands out the @p count values @ref carve put at @p output in @p arena: in place on host pages,
 *  or re-pointed at one upload to its device. */
template <typename value_type_>
inline status_t publish(usearch_arena_handle_t& arena, value_type_*& output, std::size_t count) noexcept {
    if (!arena.device_memory || !output || !count) return status_t::success_k;
    value_type_* const landed = arena.device_memory.template allocate<value_type_>(count);
    if (!landed) return status_t::bad_alloc_k;
    if (status_t const uploaded = arena.device_memory.upload(landed, output, count * sizeof(value_type_));
        failed(uploaded))
        return uploaded;
    output = landed;
    return status_t::success_k;
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

/** Carves @p count rows of @p wanted matches out of a claimed arena's host pages, @p staging; a
 *  zero @p wanted carves only the counts. False once @p outcome records why not. */
inline bool carve_matches(growing_arena<forkunion_allocator_t>& staging, std::size_t count, std::size_t wanted,
                          matches_landing_t& landing, usearch_outcome_t* outcome) noexcept {
    landing.wanted = wanted;
    landing.counts = carve<std::size_t>(staging, count);
    if (wanted) {
        landing.keys = carve<usearch_key_t>(staging, count * wanted);
        landing.scores = carve<usearch_distance_t>(staging, count * wanted);
    }
    if (landing.counts && (!wanted || (landing.keys && landing.scores))) return true;
    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
}

/** Hands out the @p count rows @ref carve_matches put in @p landing where @p arena hands outputs
 *  out. False once @p outcome records why not. */
inline bool publish_matches(usearch_arena_handle_t& arena, std::size_t count, matches_landing_t& landing,
                            usearch_outcome_t* outcome) noexcept {
    std::size_t const entries = count * landing.wanted;
    status_t status = publish(arena, landing.counts, count);
    if (succeeded(status)) status = publish(arena, landing.keys, entries);
    if (succeeded(status)) status = publish(arena, landing.scores, entries);
    return succeeded(status) || propagate_error(status, outcome, "Failed to land the matches on the device");
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

    /** Signalled whenever a taken call becomes final. */
    std::condition_variable finished;

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
        deferred_registry.finished.notify_all();
    }
}

/** Waits until no call @p executor took with a ticket up to @p last still runs on another
 *  thread. */
inline void wait_running(usearch_executor_handle_t& executor, std::size_t last) noexcept {
    std::unique_lock<std::mutex> lock(deferred_registry.guard);
    deferred_registry.finished.wait(lock, [&]() noexcept {
        for (deferred_call_t const& slot : executor.deferred)
            if (slot.stage == deferred_stage_t::running_k && slot.ticket <= last) return false;
        return true;
    });
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
        else if (running) deferred_registry.finished.wait(lock);
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
