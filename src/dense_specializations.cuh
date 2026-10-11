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
#include <bit>         // `std::bit_cast` - the C key struct & `us::uuid_t` share sixteen bytes
#include <mutex>       // `std::lock_guard` over the ticks' lock
#include <span>        // `std::span` - the search verbs' landing planes
#include <string_view> // `std::string_view` - kernel tokens & checkpoint paths
#include <type_traits> // `std::is_same_v` - the residency discrimination

#include <forkunion/capabilities.hpp> // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/types.hpp>        // `fu::spin_mutex_t` - the ticks' lock

#include "usearch/dense_index.h"
#include "usearch/dense/ram_indexes.hpp"
#if USEARCH_BUILD_CUDA
#include "usearch/dense/vram_indexes.cuh"
#endif

using namespace unum::usearch;
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
    {usearch_scalar_i4x2_k, nk_i4_k},   {usearch_scalar_u8_k, nk_u8_k},     {usearch_scalar_u4x2_k, nk_u4_k},
    {usearch_scalar_u1x8_k, nk_u1_k},
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
    {nk_cap_haswell_k, "haswell"},
    {nk_cap_skylake_k, "skylake"},
    {nk_cap_icelake_k, "icelake"},
    {nk_cap_genoa_k, "genoa"},
    {nk_cap_sapphire_k, "sapphire"},
    {nk_cap_sapphireamx_k, "sapphireamx"},
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
    case cpu_class_t::intel_diamond_k:
        return numkong_capabilities_of(cpu_class_t::amd_genoa_k) | nk_cap_alder_k | nk_cap_sierra_k | nk_cap_diamond_k;
    case cpu_class_t::aws_g2_k: return nk_cap_serial_k | nk_cap_neon_k;
    case cpu_class_t::apple_m2_k:
    case cpu_class_t::aws_g5_k:
        return numkong_capabilities_of(cpu_class_t::aws_g2_k) | nk_cap_neonsdot_k | nk_cap_neonbfdot_k |
               nk_cap_neonfhm_k;
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
    case cpu_class_t::intel_diamond_k: return cpu_class_t::amd_genoa_k;
    case cpu_class_t::amd_genoa_k: return cpu_class_t::intel_icelake_k;
    case cpu_class_t::intel_icelake_k: return cpu_class_t::intel_skylake_k;
    case cpu_class_t::intel_skylake_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::intel_sierra_k: return cpu_class_t::intel_haswell_k;
    case cpu_class_t::nvidia_vera_k: return cpu_class_t::aws_g5_k;
    case cpu_class_t::aws_g5_k: return cpu_class_t::apple_m2_k;
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
    {gpu_class_t::nvidia_ampere_k, "nvidia_ampere"},
    {gpu_class_t::nvidia_ada_k, "nvidia_ada"},
    {gpu_class_t::nvidia_hopper_k, "nvidia_hopper"},
    {gpu_class_t::nvidia_blackwell_k, "nvidia_blackwell"},
    {gpu_class_t::nvidia_blackwellrtx_k, "nvidia_blackwellrtx"},
    {gpu_class_t::nvidia_rubin_k, "nvidia_rubin"},
};

#if USEARCH_BUILD_CUDA

/** Lowers one GPU class onto the CUDA architecture ceiling its kits compile up to; Rubin rides the
 *  Blackwell ceiling until its own kits land. */
constexpr cuda_arch_t cuda_arch_of(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case gpu_class_t::nvidia_ada_k: return cuda_arch_ada_k;
    case gpu_class_t::nvidia_hopper_k: return cuda_arch_hopper_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_rubin_k: return cuda_arch_blackwell_k;
    case gpu_class_t::nvidia_blackwellrtx_k: return cuda_arch_blackwellrtx_k;
    default: return cuda_arch_ampere_k;
    }
}

/** The class whose unit carries cubins for a device of compute capability @p arch; any generation
 *  without a class of its own runs the Ampere unit, natively on 8.x and through its PTX past that. */
constexpr gpu_class_t gpu_class_of(int arch) noexcept {
    switch (arch / 100) {
    case 8: return arch >= cuda_arch_ada_k ? gpu_class_t::nvidia_ada_k : gpu_class_t::nvidia_ampere_k;
    case 9: return gpu_class_t::nvidia_hopper_k;
    case 10: return gpu_class_t::nvidia_blackwell_k;
    case 12: return gpu_class_t::nvidia_blackwellrtx_k;
    default: return gpu_class_t::nvidia_ampere_k;
    }
}
#endif

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

/** The C key struct and the engine's @c us::uuid_t share sixteen bytes - one bit cast in. */
inline us::uuid_t uuid_of(usearch_key_t key) noexcept { return std::bit_cast<us::uuid_t>(key); }

/** One strided C matrix's rank-2 shape: NumKong counts the last extent in logical dimensions, so
 *  a packed row is as wide as the index; a zero stride means contiguous rows. */
template <typename scalar_type_>
inline nk::shape_storage_<2> rows_shape_of(std::size_t rows_count, std::size_t stride,
                                           std::size_t dimensions) noexcept {
    std::size_t const row_values = divide_round_up<std::size_t>(dimensions, nk::dimensions_per_value<scalar_type_>());
    std::ptrdiff_t const row_bytes = static_cast<std::ptrdiff_t>(stride ? stride : row_values * sizeof(scalar_type_));
    return {{rows_count, dimensions}, {row_bytes, static_cast<std::ptrdiff_t>(sizeof(scalar_type_))}, 2};
}

/** Whether rows reach an arm only through @p staged: a foreign scalar converts, and a zero stride,
 *  which repeats the first row, is materialized into the rows an arm walks. */
template <typename multi_type_>
inline bool stages_rows(nk_dtype_t source_dtype, std::size_t row_bytes) noexcept {
    return source_dtype != multi_type_::vector_dtype_k || !row_bytes;
}

/** A chunk's rows in @p multi_type_ 's stored scalar: the caller's own when they match, else cast
 *  into @p staged from its row @p staged_first on. */
template <typename multi_type_>
inline nk::matrix_view<typename multi_type_::scalar_t> stored_rows_of( //
    char const* rows, std::size_t rows_count, std::size_t row_bytes, nk_dtype_t source_dtype, std::size_t dimensions,
    nk::matrix<typename multi_type_::scalar_t>& staged, std::size_t staged_first = 0) noexcept {
    using scalar_t = typename multi_type_::scalar_t;
    if (!stages_rows<multi_type_>(source_dtype, row_bytes))
        return {rows, rows_shape_of<scalar_t>(rows_count, row_bytes, dimensions)};
    for (std::size_t row = 0; row != rows_count; ++row) {
        [[maybe_unused]] nk_status_t const cast = dense_elementwise_chain<nk_cap_any_k>::kit_t::cast(
            rows + row * row_bytes, source_dtype, staged.span().row(staged_first + row).as_vector().byte_data(),
            multi_type_::vector_dtype_k, dimensions, nullptr);
        assert(cast == nk_success_k && "The serial tier casts every dtype pair");
    }
    return {staged.span().row(staged_first).as_vector().byte_data(),
            rows_shape_of<scalar_t>(rows_count, 0, dimensions)};
}

#pragma endregion Scaffolding

#pragma region Specialized Index

/** Residency discrimination off the alternative's own executor type - no second tag exists to
 *  disagree with the variant. */
template <typename multi_type_>
inline constexpr bool is_ram_index = std::is_same_v<typename multi_type_::executor_t, forkunion_executor_t>;
#if USEARCH_BUILD_CUDA
template <typename multi_type_>
inline constexpr bool is_vram_index = std::is_same_v<typename multi_type_::executor_t, cuda_executor_t>;
#else
template <typename multi_type_>
inline constexpr bool is_vram_index = false;
#endif

/** One RAM alternative: the CPU class alone names the rung, lowering onto its NumKong mask, its waiter,
 *  its atomic reference and its cache hints. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using ram_index = dense_ram_indexes<us::uuid_t, dtype_, numkong_capabilities_of(cpu_class_), waiter_of<cpu_class_>,
                                    atomic_ref_for_cpu_class<cpu_class_>::template type,
                                    typename cache_hints_for_cpu_class<cpu_class_>::type>;
#if USEARCH_BUILD_CUDA

/** One VRAM alternative: the GPU class alone names the rung, lowering onto its architecture ceiling;
 *  its kernels come from that class's unit, compiled for its architecture alone. */
template <nk_dtype_t dtype_, gpu_class_t gpu_class_>
using vram_index = dense_vram_indexes<us::uuid_t, dtype_, cuda_arch_of(gpu_class_)>;
#endif

/** The C predicate lifted onto the index's key vocabulary. */
struct key_predicate_t {
    usearch_predicate_t callback;
    void* state;

    admission_t operator()(us::uuid_t key) const noexcept {
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
    void operator()(std::span<us::uuid_t const> keys, std::span<nk::vector_view<scalar_type_>> rows) const noexcept {
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
    usearch_value_lookup_t callback;
    void* state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

#if USEARCH_BUILD_CUDA
    __device__ void const* operator()(us::uuid_t key) const noexcept {
        usearch_key_t const scored = std::bit_cast<usearch_key_t>(key);
        void const* row = nullptr;
        callback(state, &scored, 1, scalar, dimensions, &row);
        return row;
    }
#endif
};

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

/** One insertion's C rows and its handler: yields a chunk in an arm's stored scalar, and reports
 *  each chunk once it lands. Every row is claimed before the first report and a dispensed row
 *  cannot be withdrawn, so the handler's answer is ignored. */
struct insertion_rows_t {
    char const* data;
    std::size_t row_bytes;
    nk_dtype_t source_dtype;
    usearch_progress_callback_t progress;
    void* progress_state;

    /** Rows @b [first,first+taking) as @p index stores them, staged into @p staged from its row
     *  @p staged_first on when the caller's own differ. */
    template <typename multi_type_>
    nk::matrix_view<typename multi_type_::scalar_t> operator()(multi_type_ const& index, std::size_t first,
                                                               std::size_t taking,
                                                               nk::matrix<typename multi_type_::scalar_t>& staged,
                                                               std::size_t staged_first) const noexcept {
        return stored_rows_of<multi_type_>(data + first * row_bytes, taking, row_bytes, source_dtype,
                                           index.dimensions(), staged, staged_first);
    }

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
    using vectors_view_t = typename index_type_::vectors_view_t;
    using vectors_span_t = typename index_type_::vectors_span_t;
    using reserve_capacity_t = typename index_type_::reserve_capacity_t;
    using insert_config_t = typename index_type_::insert_config_t;
    using insert_result_t = typename index_type_::insert_result_t;
    using search_config_t = typename index_type_::search_config_t;
    using search_result_t = typename index_type_::search_result_t;
    using refine_config_t = typename index_type_::refine_config_t;
    using refine_result_t = typename index_type_::refine_result_t;
    using health_result_t = typename index_type_::health_result_t;
    using keys_span_t = std::span<key_t>;
    using distances_span_t = std::span<typename index_type_::distance_t>;
    using counts_span_t = std::span<std::size_t>;

    /** What an attend resolves value rows through: a host callback per batch on RAM, a device call per key on GPU. */
    using attend_lookup_t = std::conditional_t<is_ram_index<index_type_>, value_lookup_t, device_value_lookup_t>;

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
    std::size_t count(key_t const& key) const noexcept;
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept;
    status_t reserve(reserve_capacity_t config) noexcept;
    status_t clear() noexcept;

    /** Lands @p rows under @p keys through one plan, a step per chunk, and publishes them all at once. */
    expected<insert_result_t> add(std::span<key_t const> keys, insertion_rows_t const& rows, executor_t& executor,
                                  insert_config_t config) noexcept;
    expected<refine_result_t> refine(executor_t& executor, refine_config_t config, shard_ticks_t& ticks) noexcept;
    expected<health_result_t> health(executor_t& executor) noexcept;
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     std::size_t wanted, keys_span_t keys, distances_span_t distances,
                                     counts_span_t counts, dummy_predicate_t) noexcept;
    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     std::size_t wanted, keys_span_t keys, distances_span_t distances,
                                     counts_span_t counts, key_predicate_t& predicate) noexcept
        requires is_ram_index<index_type_>;
    expected<search_result_t> multiply(vectors_view_t left, executor_t& executor, search_config_t config,
                                       std::size_t wanted, keys_span_t keys,
                                       std::span<typename index_type_::dot_result_t> products,
                                       counts_span_t counts) noexcept;
    expected<search_result_t> maxsim(vectors_view_t queries, std::size_t rows_per_query, executor_t& executor,
                                     search_config_t config, std::size_t wanted, keys_span_t keys,
                                     std::span<f64_t> scores, counts_span_t counts) noexcept;
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, attend_lookup_t value_lookup,
                                     f32_t score_scale, std::span<f32_t> outputs,
                                     std::span<us::dense_softmax_state_t> carry, executor_t& executor,
                                     search_config_t config) noexcept;
    status_t save(std::string_view path, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t save(bytes_span_t buffer, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t load(bytes_view_t buffer, executor_t& executor, shard_ticks_t& ticks) noexcept;

    /** The C layer's save config: the path's suffix picks the layout, the residency the contract -
     *  device planes cannot be settlement-probed, so a GPU handle certifies. */
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
std::size_t specialized_index<index_type_>::count(key_t const& key) const noexcept {
    return index.count(key);
}
template <typename index_type_>
std::size_t specialized_index<index_type_>::get(key_t const& key, vectors_span_t vectors,
                                                std::size_t wanted) const noexcept {
    return index.get(key, vectors, wanted);
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
    std::span<key_t const> keys, insertion_rows_t const& rows, executor_t& executor, insert_config_t config) noexcept {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const count = keys.size();
    bool const staging = stages_rows<index_type_>(rows.source_dtype, rows.row_bytes);
    // Only a handler or a staged copy splits the call, and one plan still publishes every chunk at once.
    std::size_t const chunk = rows.progress || staging ? min_of(count, increment_rows_k) : count;
    // A device step reads its rows until `finish`, so device chunks stage side by side rather than over each other.
    std::size_t const staged_rows = is_ram_index<index_type_> ? chunk : count;
    nk::matrix<scalar_t> staged;
    if (staging) {
        auto zeros = nk::matrix<scalar_t>::zeros({staged_rows, index.dimensions()});
        if (!zeros)
            return {{}, zeros.status == nk::status_t::bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k};
        staged = std::move(zeros.value);
    }

    auto [plan, status] = index.plan_insert(count, executor, config);
    if (failed(status)) return {{}, status};
    for (std::size_t landed = 0; landed != count;) {
        std::size_t const taking = min_of(chunk, count - landed);
        std::size_t const staged_first = is_ram_index<index_type_> ? 0 : landed;
        vectors_view_t const chunk_rows = rows(index, landed, taking, staged, staged_first);
        std::span<key_t const> const chunk_keys = keys.subspan(landed, taking);
        if constexpr (is_ram_index<index_type_>) plan.insert(landed, chunk_keys, chunk_rows);
        else if (failed(plan.insert(landed, chunk_keys, chunk_rows))) break;
        landed += taking;
        rows.report(landed, count);
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
    vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted, keys_span_t keys,
    distances_span_t distances, counts_span_t counts, dummy_predicate_t) noexcept {
    return index.search(queries, executor, config, wanted, keys, distances, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::search(
    vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted, keys_span_t keys,
    distances_span_t distances, counts_span_t counts, key_predicate_t& predicate) noexcept
    requires is_ram_index<index_type_>
{
    return index.search_if(queries, executor, config, wanted, keys, distances, counts, predicate);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::multiply(
    vectors_view_t left, executor_t& executor, search_config_t config, std::size_t wanted, keys_span_t keys,
    std::span<typename index_type_::dot_result_t> products, counts_span_t counts) noexcept {
    return index.multiply(left, executor, config, wanted, keys, products, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::maxsim(
    vectors_view_t queries, std::size_t rows_per_query, executor_t& executor, search_config_t config,
    std::size_t wanted, keys_span_t keys, std::span<f64_t> scores, counts_span_t counts) noexcept {
    return index.maxsim(queries, rows_per_query, executor, config, wanted, keys, scores, counts);
}
template <typename index_type_>
expected<typename specialized_index<index_type_>::search_result_t> specialized_index<index_type_>::attend(
    vectors_view_t queries, std::size_t head_count, attend_lookup_t value_lookup, f32_t score_scale,
    std::span<f32_t> outputs, std::span<us::dense_softmax_state_t> carry, executor_t& executor,
    search_config_t config) noexcept {
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

template <typename index_type_>
specialized_index(index_type_&) -> specialized_index<index_type_>;

#pragma endregion Specialized Index
