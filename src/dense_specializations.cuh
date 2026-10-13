/**
 *  @file src/dense_specializations.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What one dense arm is spelled with and compiled through: the scalar and capability
 *      tables, the CPU and GPU classes with their waiters, and the
 *      @c specialized_index view every class unit instantiates.
 *
 *  Every class unit includes this to instantiate its arms; @c dense_index.cuh adds the variant
 *  grid, the handle and the exported C entry points on top of it.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cstddef> // `std::size_t`

#include <array>       // `std::array` - the value lookup's batch
#include <atomic>      // `std::atomic_ref` - the outcome's status word
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <mutex>       // `std::lock_guard` over the ticks' lock
#include <span>        // `std::span` - the search verbs' landing matrices
#include <string_view> // `std::string_view` - kernel tokens & checkpoint paths
#include <type_traits> // `std::is_same_v` - the residency discrimination

#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/types.hpp>        // `fu::spin_mutex_t` - the ticks' lock

#include "usearch/dense_index.h"
#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/dense/ram_indexes.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/dense/vram_nvidia_indexes.hpp"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/dense/vram_amd_indexes.hpp"
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/dense/vram_apple_indexes.hpp"
#endif

/** How many storage dtypes the arms cover, from the front of the ladder bf16, u1, i8, e4m3, f16,
 *  u8, e5m2, i4, u4, e3m2, e2m3. CMake passes it, so only builds that bypass CMake take this
 *  default. */
#if !defined(USEARCH_DENSE_DTYPES)
#define USEARCH_DENSE_DTYPES 4
#endif
#if USEARCH_DENSE_DTYPES < 1 || USEARCH_DENSE_DTYPES > 11
#error "USEARCH_DENSE_DTYPES must be an integer from 1 to 11"
#endif

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

/** A kernel kind lifted back onto the C metric vocabulary. */
inline usearch_dense_metric_t usearch_metric_of(nk_kernel_kind_t kernel) noexcept {
    switch (kernel) {
    case nk_kernel_dot_k: return usearch_dense_metric_inner_k;
    case nk_kernel_angular_k: return usearch_dense_metric_angular_k;
    case nk_kernel_euclidean_k: return usearch_dense_metric_euclidean_k;
    case nk_kernel_hamming_k: return usearch_dense_metric_hamming_k;
    case nk_kernel_jaccard_k: return usearch_dense_metric_jaccard_k;
    default: return usearch_dense_metric_unknown_k;
    }
}

/** Rows one progress tick or one staged conversion spans - a few megabytes at any width in use,
 *  and wide enough to bury a fan-out's per-thread setup. */
inline constexpr std::size_t increment_rows_k = 4096;

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
 *  Zacas RISC-V - the floors, the older classes, and any class whose @c FORKUNION_TARGET_<BIT> is off in
 *  the probe list take the standard one. */
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

/** Codenames of the GPU classes a VRAM alternative dispatches onto. */
struct gpu_class_name_t {
    gpu_class_t gpu_class;
    char const* name;
};
inline constexpr gpu_class_name_t gpu_class_names_k[] = {
    {gpu_class_t::nvidia_ampere_k, "nvidia-ampere"},
    {gpu_class_t::nvidia_ada_k, "nvidia-ada"},
    {gpu_class_t::nvidia_hopper_k, "nvidia-hopper"},
    {gpu_class_t::nvidia_blackwell_k, "nvidia-blackwell"},
    {gpu_class_t::nvidia_blackwellrtx_k, "nvidia-blackwellrtx"},
    {gpu_class_t::nvidia_rubin_k, "nvidia-rubin"},
    {gpu_class_t::amd_cdna3_k, "amd-cdna3"},
    {gpu_class_t::amd_cdna4_k, "amd-cdna4"},
    {gpu_class_t::amd_cdna5_k, "amd-cdna5"},
    {gpu_class_t::apple_metal7_k, "apple-metal7"},
    {gpu_class_t::apple_metal9_k, "apple-metal9"},
    {gpu_class_t::apple_metal10_k, "apple-metal10"},
};

/** Lowers one GPU class onto NumKong's GPU capabilities. On NVIDIA, as
 *  @c nk_cuda_capabilities_detected reports them, the Ampere & Ada bits accumulate, while Hopper,
 *  Blackwell & RTX Blackwell each run only their own kits, and Rubin rides the Blackwell kits until
 *  its own land. AMD and Apple generations accumulate over their vendor's floor. */
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

/** Adds one fan-out's telemetry into the caller's stats - accumulating, so chunked verbs report
 *  the whole call rather than the last chunk. */
template <typename result_type_>
inline void accumulate_stats(usearch_dense_traversal_stats_t* stats, result_type_ const& result) noexcept {
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
template <typename value_type_>
inline bool crossed(expected<value_type_> const& result, usearch_dense_traversal_stats_t* stats,
                    usearch_outcome_t* outcome, char const* message = nullptr) noexcept {
    if constexpr (carries_telemetry<value_type_>) accumulate_stats(stats, result.value);
    return result || propagate_error(result.status, outcome, message);
}

/** The C key struct and the engine's @c utilities::uuid_t share sixteen bytes - one bit cast in. */
inline us::utilities::uuid_t uuid_of(usearch_key_t key) noexcept { return std::bit_cast<us::utilities::uuid_t>(key); }

#pragma endregion Scaffolding

#pragma region Specialized Index

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

/** One RAM alternative: the CPU class alone names the rung, lowering onto its NumKong mask, its waiter,
 *  its atomic reference and its cache hints. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using ram_indexes = us::dense::ram_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(cpu_class_),
                                           waiter_of<cpu_class_>, atomic_ref_for_cpu_class<cpu_class_>::template type,
                                           typename cache_hints_for_cpu_class<cpu_class_>::type>;

/** One NVMe alternative: scoring compiles per CPU class, while the metric comes from the parts'
 *  headers and the transport is the executor's runtime choice. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using nvme_indexes = us::dense::nvme_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(cpu_class_)>;

/** One NVIDIA alternative: the class names the rung and its capability ceiling; its unit holds the
 *  kernels. */
#if USEARCH_ARCH_CUDA_
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_nvidia_indexes =
    us::dense::vram_nvidia_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(gpu_class_)>;
#endif

/** One AMD alternative: the class names the rung and its capability ceiling; its unit holds the
 *  code object. */
#if USEARCH_ARCH_ROCM_
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_amd_indexes =
    us::dense::vram_amd_indexes<us::utilities::uuid_t, dtype_, numkong_capabilities_of(gpu_class_)>;
#endif

/** One Apple alternative, serving every Apple class until a family's kernels differ. */
#if USEARCH_ARCH_METAL_
template <nk_dtype_t dtype_>
using vram_apple_indexes = us::dense::vram_apple_indexes<us::utilities::uuid_t, dtype_>;
#endif

/** The C predicate lifted onto the index's key vocabulary. */
struct key_predicate_t {
    usearch_predicate_t callback;
    void* state;

    admission_t operator()(us::utilities::uuid_t key) const noexcept {
        return static_cast<admission_t>(callback(std::bit_cast<usearch_key_t>(key), state));
    }
};

/** The C value lookup lifted onto the index's key vocabulary: one callback per scored batch, a NULL row
 *  answering an empty view that refuses the call. */
struct value_lookup_t {
    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

    template <typename scalar_type_>
    void operator()(std::span<us::utilities::uuid_t const> keys,
                    std::span<nk::vector_view<scalar_type_>> rows) const noexcept {
        std::array<usearch_key_t, 4> batch;
        std::array<void const*, 4> found;
        for (std::size_t lane = 0; lane != keys.size(); ++lane) batch[lane] = std::bit_cast<usearch_key_t>(keys[lane]);
        callback(state, batch.data(), keys.size(), scalar, dimensions, found.data());
        for (std::size_t lane = 0; lane != keys.size(); ++lane)
            rows[lane] = found[lane]
                             ? nk::vector_view<scalar_type_> {static_cast<scalar_type_ const*>(found[lane]), dimensions}
                             : nk::vector_view<scalar_type_> {};
    }
};

/** The C value lookup called on the device, one scored key at a time inside a GPU walk. */
struct device_value_lookup_t {

    /** Names the lookup to the NVIDIA attend kernels the class units instantiate over it. */
    inline static constexpr u32_t device_lookup_k = 1;

    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

#if defined(__CUDACC__)
    __device__ void const* operator()(us::utilities::uuid_t key) const noexcept {
        usearch_key_t const scored = std::bit_cast<usearch_key_t>(key);
        void const* row = nullptr;
        callback(state, &scored, 1, scalar, dimensions, &row);
        return row;
    }
#endif
};

#if USEARCH_ARCH_CUDA_

/** What each shard of a VRAM attend resolves rows through: the C lookup behind the shard's keys. */
using nvidia_attend_lookup_t = us::dense::cuda_keyed_lookup<us::utilities::uuid_t, device_value_lookup_t>;

template <>
struct us::dense::cuda_value_lookup_of<nvidia_attend_lookup_t::device_lookup_k> {
    using type = nvidia_attend_lookup_t;
};
#endif

/** Serializes shard ticks from the executor's threads onto the C handler, which the contract
 *  promises is never concurrent. */
struct shard_ticks_t {
    usearch_progress_callback_t progress;
    void* state;
    std::size_t total;
    std::size_t done = 0; // Counted under the mutex that already serializes the handler.
    fu::spin_mutex_t mutex;

    shard_ticks_t(usearch_progress_callback_t progress, void* state, std::size_t total) noexcept
        : progress(progress), state(state), total(total) {}

    continuation_t operator()(std::size_t) noexcept {
        if (!progress) return continuation_t::continue_k;
        std::lock_guard guard {mutex};
        return forward_(++done, total);
    }

    /** The sibling for verbs that count their own units and know their own population - a
     *  refinement ticks the nodes visited so far, which no counter here could derive. */
    continuation_t operator()(std::size_t finished, std::size_t population) noexcept {
        if (!progress) return continuation_t::continue_k;
        std::lock_guard guard {mutex};
        return forward_(finished, population);
    }

  private:

    /** Hands one tick to the C handler; the caller holds the mutex. */
    continuation_t forward_(std::size_t finished, std::size_t population) noexcept {
        return progress(finished, population, state) == usearch_cancel_k ? continuation_t::cancel_k
                                                                         : continuation_t::continue_k;
    }
};

/** One insertion's handler, told of each chunk once it lands. Every row is claimed before the first
 *  report and a dispensed row cannot be withdrawn, so the handler's answer is ignored. */
struct insertion_progress_t {
    usearch_progress_callback_t progress;
    void* progress_state;

    /** Tells the handler, if any, that @p landed of @p total rows landed. */
    void report(std::size_t landed, std::size_t total) const noexcept {
        if (progress) progress(landed, total, progress_state);
    }
};

/** One arm's verbs, fixed to the C layer's callback types and wrapped as non-template members of a
 *  view over the index: an explicit instantiation compiles exactly these bodies in the arm's unit,
 *  and an @c extern one keeps every arm body out of the dispatch unit. The members are defined
 *  outside the class because in-class ones would be inline, which an explicit instantiation
 *  declaration does not restrain. */
template <typename index_type_>
struct specialized_index {
    using key_t = typename index_type_::key_t;
    using executor_t = typename index_type_::executor_t;
    using reserve_capacity_t = typename index_type_::reserve_capacity_t;
    using insert_config_t = typename index_type_::insert_config_t;
    using insert_result_t = typename index_type_::insert_result_t;
    using search_config_t = typename index_type_::search_config_t;
    using search_result_t = typename index_type_::search_result_t;
    using refine_config_t = typename index_type_::refine_config_t;
    using refine_result_t = typename index_type_::refine_result_t;
    using health_result_t = typename index_type_::health_result_t;
    using rows_view_t = us::dense::rows_view_t;
    using rows_span_t = us::dense::rows_span_t;
    using keys_view_t = nk::vector_view<key_t>;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;

    /** What an attend resolves value rows through: a host callback per batch on RAM, a device call per key on GPU. */
    using attend_lookup_t = std::conditional_t<is_vram_index<index_type_>, device_value_lookup_t, value_lookup_t>;

    index_type_& index;

    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept
        requires is_ram_index<index_type_>;
    status_t emplace(int device) noexcept
        requires is_vram_index<index_type_>;

    /** The device @p shard sits on; the first one's is what the capability string names. */
    int device(std::size_t shard) const noexcept
        requires is_vram_index<index_type_>;
    std::size_t shards_count() const noexcept;
    std::size_t dimensions() const noexcept;
    std::size_t connectivity() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t memory_usage() const noexcept;
    expected<std::size_t> storage_requirements() const noexcept;
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept;
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) noexcept;
    status_t reserve(reserve_capacity_t config) noexcept;
    status_t clear() noexcept;

    /** Lands @p rows under @p keys through one plan, a step per chunk, and publishes them all at once. */
    expected<insert_result_t> add(keys_view_t keys, rows_view_t rows, insertion_progress_t const& progress,
                                  executor_t& executor, insert_config_t config) noexcept;
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config, shard_ticks_t& ticks) noexcept;
    expected<health_result_t> health(executor_t& executor) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     dummy_predicate_t) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     key_predicate_t& predicate) noexcept
        requires is_ram_index<index_type_>;
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys,
                                       scores_span_t products, counts_span_t counts) noexcept;
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, keys_span_t keys, scores_span_t scores,
                                     counts_span_t counts) noexcept;
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup,
                                     f32_t score_scale, rows_span_t outputs,
                                     std::span<us::dense::softmax_state_t> carry, executor_t& executor,
                                     search_config_t config) noexcept;
    status_t save(std::string_view path, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t save(bytes_span_t buffer, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t load(bytes_view_t buffer, executor_t& executor, shard_ticks_t& ticks) noexcept;

    /** The C layer's save config: the path's suffix picks the layout, the residency the contract -
     *  device storage cannot be settlement-probed, so a GPU handle certifies. */
    static save_config_t save_config_of(std::string_view path) noexcept;
};

template <typename index_type_>
status_t specialized_index<index_type_>::emplace(fu_memory_domain_id_t memory_domain) noexcept
    requires is_ram_index<index_type_>
{
    return index.emplace(memory_domain);
}
template <typename index_type_>
status_t specialized_index<index_type_>::emplace(int device) noexcept
    requires is_vram_index<index_type_>
{
    return index.emplace(device);
}
template <typename index_type_>
int specialized_index<index_type_>::device(std::size_t shard) const noexcept
    requires is_vram_index<index_type_>
{
    return index.shard(shard).device_specs().ordinal;
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::shards_count() const noexcept {
    return index.shards_count();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::dimensions() const noexcept {
    return index.dimensions();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::connectivity() const noexcept {
    return index.connectivity();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::size() const noexcept {
    return index.size();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::capacity() const noexcept {
    return index.capacity();
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::memory_usage() const noexcept {
    return index.memory_usage();
}
template <typename index_type_>
expected<std::size_t> specialized_index<index_type_>::storage_requirements() const noexcept {
    return index.storage_requirements(save_config_of({}));
}
template <typename index_type_>
status_t specialized_index<index_type_>::count(std::span<key_t const> keys, std::span<std::size_t> entries,
                                               executor_t& executor) noexcept {
    return index.count(keys, entries, executor);
}
template <typename index_type_>
status_t specialized_index<index_type_>::get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted,
                                             std::span<std::size_t> entries, executor_t& executor) noexcept {
    return index.get(keys, vectors, wanted, entries, executor);
}
template <typename index_type_>
status_t specialized_index<index_type_>::reserve(reserve_capacity_t config) noexcept {
    return index.reserve(config).status;
}
template <typename index_type_>
status_t specialized_index<index_type_>::clear() noexcept {
    return index.clear();
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::insert_result_t> specialized_index<index_type_>::add(
    keys_view_t keys, rows_view_t rows, insertion_progress_t const& progress, executor_t& executor,
    insert_config_t config) noexcept {
    std::size_t const count = keys.size();
    // Only a handler splits the call, and one plan still publishes every chunk at once.
    std::size_t const chunk = progress.progress ? min_of(count, increment_rows_k) : count;
    auto [plan, status] = index.plan_insert(count, executor, config);
    if (failed(status)) return {{}, status};
    for (std::size_t landed = 0; landed != count;) {
        std::size_t const taking = min_of(chunk, count - landed);
        if constexpr (is_ram_index<index_type_>)
            plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking));
        else if (failed(plan.insert(landed, keys[nk::range(landed, landed + taking)], rows.slice(landed, taking))))
            break;
        landed += taking;
        progress.report(landed, count);
    }
    // After a failed device step, `finish` publishes nothing and reports the failure the plan kept.
    return plan.finish();
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::refine_result_t> specialized_index<index_type_>::refine(
    executor_t& executor, refine_config_t config, shard_ticks_t& ticks) noexcept {
    return index.refine(executor, config,
                        [&](std::size_t swept, std::size_t population) noexcept { return ticks(swept, population); });
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::health_result_t> specialized_index<index_type_>::health(
    executor_t& executor) noexcept {
    return index.health(executor);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, dummy_predicate_t) noexcept {
    return index.search(queries, executor, config, keys, distances, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, key_predicate_t& predicate) noexcept
    requires is_ram_index<index_type_>
{
    return index.search_if(queries, executor, config, keys, distances, counts, predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t products,
    counts_span_t counts) noexcept {
    return index.multiply(left, executor, config, keys, products, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    rows_view_t queries, std::size_t rows_per_query, executor_t& executor, search_config_t config, keys_span_t keys,
    scores_span_t scores, counts_span_t counts) noexcept {
    return index.maxsim(queries, rows_per_query, executor, config, keys, scores, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<us::dense::softmax_state_t> carry, executor_t& executor, search_config_t config) noexcept {
    return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
}
template <typename index_type_>
status_t specialized_index<index_type_>::save(std::string_view path, executor_t& executor,
                                              shard_ticks_t& ticks) noexcept {
    return index.save(path, save_config_of(path), executor, ticks);
}
template <typename index_type_>
status_t specialized_index<index_type_>::save(bytes_span_t buffer, executor_t& executor,
                                              shard_ticks_t& ticks) noexcept {
    return index.save(buffer, save_config_of({}), executor, ticks);
}
template <typename index_type_>
status_t specialized_index<index_type_>::load(std::string_view path, executor_t& executor,
                                              shard_ticks_t& ticks) noexcept {
    return index.load(path, load_config_t {}, executor, ticks);
}
template <typename index_type_>
status_t specialized_index<index_type_>::load(bytes_view_t buffer, executor_t& executor,
                                              shard_ticks_t& ticks) noexcept {
    return index.load(buffer, load_config_t {}, executor, ticks);
}
template <typename index_type_>
save_config_t specialized_index<index_type_>::save_config_of(std::string_view path) noexcept {
    save_config_t config;
    config.layout = path.ends_with(".json") ? checkpoint_layout_t::distributed_k : checkpoint_layout_t::monolith_k;
    if constexpr (is_vram_index<index_type_>) config.synchronization = synchronization_t::immutable_k;
    return config;
}

/** One NVMe arm's verbs: the read verbs alone, since a view of parts on storage never mutates,
 *  defined outside the class for the same reason as the primary's. */
template <typename index_type_>
    requires is_nvme_index<index_type_>
struct specialized_index<index_type_> {
    using key_t = typename index_type_::key_t;
    using executor_t = typename index_type_::executor_t;
    using search_config_t = typename index_type_::search_config_t;
    using search_result_t = typename index_type_::search_result_t;
    using health_result_t = typename index_type_::health_result_t;
    using rows_view_t = us::dense::rows_view_t;
    using rows_span_t = us::dense::rows_span_t;
    using keys_span_t = nk::matrix_span<key_t>;
    using scores_span_t = nk::matrix_span<f64_t>;
    using counts_span_t = std::span<std::size_t>;
    using attend_lookup_t = value_lookup_t;

    index_type_& index;

    status_t view(std::string_view path, std::span<std::string_view const> paths) noexcept;
    std::size_t shards_count() const noexcept;
    std::size_t dimensions() const noexcept;
    std::size_t connectivity() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t memory_usage() const noexcept;
    expected<health_result_t> health(executor_t& executor) noexcept;
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept;
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted, std::span<std::size_t> entries,
                 executor_t& executor) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     dummy_predicate_t) noexcept;
    expected<search_result_t> search(rows_view_t queries, executor_t& executor, search_config_t config,
                                     keys_span_t keys, scores_span_t distances, counts_span_t counts,
                                     key_predicate_t& predicate) noexcept;
    expected<search_result_t> multiply(rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys,
                                       scores_span_t products, counts_span_t counts) noexcept;
    expected<search_result_t> maxsim(rows_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, keys_span_t keys, scores_span_t scores,
                                     counts_span_t counts) noexcept;
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup,
                                     f32_t score_scale, rows_span_t outputs,
                                     std::span<us::dense::softmax_state_t> carry, executor_t& executor,
                                     search_config_t config) noexcept;
};

template <typename index_type_>
    requires is_nvme_index<index_type_>
status_t specialized_index<index_type_>::view(std::string_view path, std::span<std::string_view const> paths) noexcept {
    return index.view(path, paths);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::shards_count() const noexcept {
    return index.shards_count();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::dimensions() const noexcept {
    return index.dimensions();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::connectivity() const noexcept {
    return index.connectivity();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::size() const noexcept {
    return index.size();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::capacity() const noexcept {
    return index.capacity();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
std::size_t specialized_index<index_type_>::memory_usage() const noexcept {
    return index.memory_usage();
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::health_result_t> specialized_index<index_type_>::health(
    executor_t& executor) noexcept {
    return index.health(executor);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
status_t specialized_index<index_type_>::count(std::span<key_t const> keys, std::span<std::size_t> entries,
                                               executor_t& executor) noexcept {
    return index.count(keys, entries, executor);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
status_t specialized_index<index_type_>::get(std::span<key_t const> keys, rows_span_t vectors, std::size_t wanted,
                                             std::span<std::size_t> entries, executor_t& executor) noexcept {
    return index.get(keys, vectors, wanted, entries, executor);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, dummy_predicate_t) noexcept {
    return index.search(queries, executor, config, keys, distances, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    rows_view_t queries, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t distances,
    counts_span_t counts, key_predicate_t& predicate) noexcept {
    return index.search_if(queries, executor, config, keys, distances, counts, predicate);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    rows_view_t left, executor_t& executor, search_config_t config, keys_span_t keys, scores_span_t products,
    counts_span_t counts) noexcept {
    return index.multiply(left, executor, config, keys, products, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    rows_view_t queries, std::size_t rows_per_query, executor_t& executor, search_config_t config, keys_span_t keys,
    scores_span_t scores, counts_span_t counts) noexcept {
    return index.maxsim(queries, rows_per_query, executor, config, keys, scores, counts);
}
template <typename index_type_>
    requires is_nvme_index<index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    rows_view_t queries, std::size_t head_count, attend_lookup_t value_lookup, f32_t score_scale, rows_span_t outputs,
    std::span<us::dense::softmax_state_t> carry, executor_t& executor, search_config_t config) noexcept {
    return index.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
}

template <typename index_type_>
specialized_index(index_type_&) -> specialized_index<index_type_>;

#pragma endregion Specialized Index
