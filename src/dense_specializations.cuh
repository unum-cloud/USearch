/**
 *  @brief What one dense arm is spelled with and compiled through: the scalar and capability
 *      tables, the CPU and GPU classes with their waiters, the key hash, and the
 *      @c specialized_index view every class unit instantiates.
 *  @author Ash Vardanian
 *  @file src/dense_specializations.cuh
 *  @date August 14, 2026
 *
 *  Every class unit includes this to instantiate its arms; @c dense_index.cuh adds the variant
 *  grid, the handle and the exported C entry points on top of it.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t` - the sliced key hash

#include <atomic>      // `std::atomic` - serializing the progress ticks
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

/** @brief The C scalar vocabulary beside NumKong's - independent enumerations, so both crossings
 *      read one table. */
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

/** @brief A checkpoint's kernel name lifted back onto the C metric vocabulary. */
inline usearch_dense_metric_t usearch_metric_of(std::string_view kernel) noexcept {
    switch (nk_kernel_named(kernel.data(), kernel.size())) {
    case nk_kernel_dot_k: return usearch_dense_metric_inner_k;
    case nk_kernel_angular_k: return usearch_dense_metric_angular_k;
    case nk_kernel_euclidean_k: return usearch_dense_metric_euclidean_k;
    case nk_kernel_hamming_k: return usearch_dense_metric_hamming_k;
    case nk_kernel_jaccard_k: return usearch_dense_metric_jaccard_k;
    default: return usearch_dense_metric_unknown_k;
    }
}

/** @brief Rows one progress tick or one staged conversion spans - a few megabytes at any width in
 *      use, and wide enough to bury a fan-out's per-thread setup on CPU and GPU executors alike. */
inline constexpr std::size_t increment_rows_k = 4096;

/** @brief Codenames of the dispatchable CPU tiers, composed into the capability string once. */
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

/** @brief Lowers one CPU class onto NumKong's compute mask - cumulative within a platform,
 *      so the chains pick their best row per dtype under it. */
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

/** @brief The class a refused rung degrades to - the next one down its own lineage, which is the
 *      class an arm must be distinct from to earn instantiation. Two lineages share the Haswell
 *      floor on x86: the AVX-512 P-cores and the AVX-VNNI E-cores. */
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

/** @brief The wait instruction one CPU class pairs: x86 spins PAUSE, the WAITPKG Xeons and the
 *      monitored Arm & RISC-V classes arm TPAUSE, WFET & WRS, everything else yields - the floor
 *      takes the standard one, as does any class whose @c FU_TARGET_<BIT> is off in the probe list
 *      every runtime-dispatched unit takes. */
template <cpu_class_t cpu_class_>
struct waiter_for_cpu_class {
    using type = fu::standard_yield_t;
};
#if FU_TARGET_X86_PAUSE
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
#if FU_TARGET_X86_TPAUSE
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_sierra_k> {
    using type = fu::x86_tpause_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::intel_diamond_k> {
    using type = fu::x86_tpause_t;
};
#endif
#if FU_TARGET_ARM64_YIELD
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::apple_m2_k> {
    using type = fu::arm64_yield_t;
};
#endif
#if FU_TARGET_ARM64_WFET
template <>
struct waiter_for_cpu_class<cpu_class_t::aws_g5_k> {
    using type = fu::arm64_wfet_t;
};
template <>
struct waiter_for_cpu_class<cpu_class_t::nvidia_vera_k> {
    using type = fu::arm64_wfet_t;
};
#endif
#if FU_TARGET_RISC5_WRS
template <>
struct waiter_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    using type = fu::risc5_wrs_t;
};
#endif

/** @brief The waiter a CPU class pairs, spelled short for the arms' explicit instantiations. */
template <cpu_class_t cpu_class_>
using waiter_of = typename waiter_for_cpu_class<cpu_class_>::type;

/** @brief The atomic reference one CPU class pairs: single-instruction LSE read-modify-writes
 *      under RCpc acquiring loads on the Armv8.3+ classes, conditional adds on the CMPCCXADD
 *      Xeons, `amocas` on Zacas RISC-V - the floors and the older classes take the standard one,
 *      as does every class whose @c FU_TARGET_<BIT> is off in the probe list every runtime-dispatched unit takes. */
template <cpu_class_t cpu_class_>
struct atomic_ref_for_cpu_class {
    template <typename value_type_>
    using type = fu::standard_atomic_ref<value_type_>;
};
#if FU_TARGET_X86_CMPCCXADD
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
#if FU_TARGET_ARM64_RCPC
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
#elif FU_TARGET_ARM64_LSE
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
#if FU_TARGET_RISC5_ZACAS
template <>
struct atomic_ref_for_cpu_class<cpu_class_t::alibaba_c930_k> {
    template <typename value_type_>
    using type = fu::risc5_zacas_atomic_ref<value_type_>;
};
#endif

/** @brief The spin hints every core of an architecture carries - PAUSE, YIELD, Zihintpause: the
 *      runtime confirms nothing about them, and a class differing by one alone earns no rung. */
inline constexpr fu::capabilities_t baseline_hints_k = fu::capability_x86_pause_k | fu::capability_arm64_yield_k |
                                                       fu::capability_risc5_pause_k;

/** @brief Lowers one CPU class onto ForkUnion's mask - the bits the runtime must confirm before
 *      the class's waiter and atomic reference are trusted, read off the two types themselves. */
template <cpu_class_t cpu_class_>
constexpr fu::capabilities_t forkunion_capabilities_of() noexcept {
    constexpr fu::capabilities_t waits_k = waiter_of<cpu_class_>::capability_k;
    constexpr fu::capabilities_t atomics_k = atomic_ref_for_cpu_class<cpu_class_>::template type<u32_t>::capabilities_k;
    return static_cast<fu::capabilities_t>((waits_k | atomics_k) & ~baseline_hints_k);
}

/** @brief Whether every compute bit one CPU class lowers onto is present in @p available. */
inline bool numkong_admits(nk_capability_t available, cpu_class_t cpu_class) noexcept {
    nk_capability_t const wanted = numkong_capabilities_of(cpu_class);
    return (available & wanted) == wanted;
}

/** @brief Whether the runtime confirms every ForkUnion bit one CPU class needs. */
template <cpu_class_t cpu_class_>
inline bool forkunion_admits() noexcept {
    constexpr fu::capabilities_t wanted = forkunion_capabilities_of<cpu_class_>();
    return (fu::runtime_capabilities() & wanted) == wanted;
}

#pragma endregion CPU Classes

/** @brief Codenames of the GPU classes a VRAM alternative dispatches onto. */
struct gpu_class_name_t {
    gpu_class_t gpu_class;
    char const* name;
};
inline constexpr gpu_class_name_t gpu_class_names_k[] = {
    {gpu_class_t::nvidia_hopper_k, "nvidia_hopper"},
    {gpu_class_t::nvidia_blackwell_k, "nvidia_blackwell"},
    {gpu_class_t::nvidia_rubin_k, "nvidia_rubin"},
};

#if USEARCH_BUILD_CUDA
/** @brief Lowers one GPU class onto the CUDA architecture ceiling its kits compile up to;
 *      Rubin rides the newest compiled ceiling until its own kits land. */
constexpr cuda_arch_t cuda_arch_of(gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case gpu_class_t::nvidia_hopper_k: return cuda_arch_hopper_k;
    case gpu_class_t::nvidia_blackwell_k:
    case gpu_class_t::nvidia_rubin_k: return cuda_arch_blackwell_k;
    default: return cuda_arch_ampere_k;
    }
}
#endif

/** @brief The one write path to @p error; the message is null exactly on success. */
inline bool propagate_error(usearch_status_t status, usearch_error_t* error, char const* message) noexcept {
    bool const succeeded = status == usearch_success_k;
    if (error) *error = usearch_error_t {status, succeeded ? nullptr : message};
    return succeeded;
}

/** @brief The engine-side overload: the C++ mirror shares the ABI's values, so crossing is a
 *      cast, and an unnamed @p message falls back to the status's own static text. */
inline bool propagate_error(status_t status, usearch_error_t* error, char const* message = nullptr) noexcept {
    return propagate_error(static_cast<usearch_status_t>(status), error, message ? message : status_to_string(status));
}

/** @brief Success needs no vocabulary crossing and no message. */
inline void propagate_success(usearch_error_t* error) noexcept {
    if (error) *error = usearch_error_t {usearch_success_k, nullptr};
}

/** @brief The C key struct and the engine's @c us::uuid_t share sixteen bytes - one bit cast in. */
inline us::uuid_t uuid_of(usearch_key_t key) noexcept { return std::bit_cast<us::uuid_t>(key); }

/** @brief Slices the low eight key bytes as the hash: keys arrive pre-mixed, UUIDs and content
 *      hashes alike. */
struct uuid_slice_hash_t {
    std::size_t operator()(us::uuid_t const& uuid) const noexcept {
        struct halves_t {
            u64_t low, high;
        };
        return static_cast<std::size_t>(std::bit_cast<halves_t>(uuid).low);
    }
};

#pragma endregion Scaffolding

#pragma region Specialized Index

/** @brief Residency discrimination off the alternative's own executor type - no second tag
 *      exists to disagree with the variant. */
template <typename multi_type_>
inline constexpr bool is_ram_index = std::is_same_v<typename multi_type_::executor_t, forkunion_executor_t>;
#if USEARCH_BUILD_CUDA
template <typename multi_type_>
inline constexpr bool is_vram_index = std::is_same_v<typename multi_type_::executor_t, cuda_executor_t>;
#else
template <typename multi_type_>
inline constexpr bool is_vram_index = false;
#endif

/** One RAM alternative: the CPU class alone names the rung, lowering onto its NumKong mask, its waiter
 *  and its atomic reference. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_>
using ram_index =
    dense_ram_indexes<us::uuid_t, dtype_, kernel_, numkong_capabilities_of(cpu_class_), waiter_of<cpu_class_>,
                      atomic_ref_for_cpu_class<cpu_class_>::template type, uuid_slice_hash_t>;
#if USEARCH_BUILD_CUDA
/** One VRAM alternative: the GPU class is a compilation ceiling, not a target, since the index dispatches
 *  by the actual device's generation at runtime, so the newest class serves the whole fleet. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, gpu_class_t gpu_class_ = gpu_class_t::nvidia_blackwell_k>
using vram_index = dense_vram_indexes<us::uuid_t, dtype_, kernel_, cuda_arch_of(gpu_class_), uuid_slice_hash_t>;
#endif

/** @brief The C predicate lifted onto the index's key vocabulary. */
struct key_predicate_t {
    usearch_predicate_t callback;
    void* state;

    admission_t operator()(us::uuid_t key) const noexcept {
        return static_cast<admission_t>(callback(std::bit_cast<usearch_key_t>(key), state));
    }
};

/** @brief Serializes shard ticks from the executor's threads onto the C handler, which the
 *      contract promises is never concurrent. */
struct shard_ticks_t {
    usearch_progress_callback_t progress;
    void* state;
    std::size_t total;
    std::atomic<std::size_t> done {0};
    fu::spin_mutex_t mutex;

    shard_ticks_t(usearch_progress_callback_t progress, void* state, std::size_t total) noexcept
        : progress(progress), state(state), total(total) {}

    continuation_t operator()(std::size_t) noexcept {
        std::size_t const finished = done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!progress) return continuation_t::continue_k;
        std::lock_guard guard {mutex};
        return progress(finished, total, state) == usearch_cancel_k ? continuation_t::cancel_k
                                                                    : continuation_t::continue_k;
    }
};

/** @brief One arm's verbs, fixed to the C layer's callback types and wrapped as non-template members
 *      of a view over the index: an explicit instantiation compiles exactly these bodies in the arm's
 *      unit, and an @c extern one keeps every arm body out of the dispatch unit. The dispatch reaches
 *      an arm only through this view, so no other instantiation is ever needed. The members are defined
 *      outside the class on purpose: inside it they would be inline, which an explicit instantiation
 *      declaration does not restrain, and the arm bodies would compile into the dispatch unit after all. */
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
    using keys_span_t = std::span<key_t>;
    using distances_span_t = std::span<typename index_type_::distance_t>;
    using counts_span_t = std::span<std::size_t>;

    index_type_& index;

    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept
        requires is_ram_index<index_type_>;
    status_t emplace(int device) noexcept
        requires is_vram_index<index_type_>;
    /** @brief The device the first shard sits on - what the capability string names. */
    int device() const noexcept
        requires is_vram_index<index_type_>;
    std::size_t shards_count() const noexcept;
    std::size_t dimensions() const noexcept;
    std::size_t connectivity() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t memory_usage() const noexcept;
    std::size_t storage_requirements() const noexcept;
    std::size_t count(key_t const& key) const noexcept;
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted) const noexcept;
    status_t reserve(reserve_capacity_t config) noexcept;
    status_t clear() noexcept;
    insert_result_t add(std::span<key_t const> keys, vectors_view_t rows, executor_t& executor,
                        insert_config_t config) noexcept;
    search_result_t search(vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted,
                           keys_span_t keys, distances_span_t distances, counts_span_t counts) noexcept;
    search_result_t search_if(vectors_view_t queries, executor_t& executor, search_config_t config, std::size_t wanted,
                              keys_span_t keys, distances_span_t distances, counts_span_t counts,
                              key_predicate_t& predicate) noexcept
        requires is_ram_index<index_type_>;
    status_t save(std::string_view path, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t save(bytes_span_t buffer, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, shard_ticks_t& ticks) noexcept;
    status_t load(bytes_view_t buffer, executor_t& executor, shard_ticks_t& ticks) noexcept;

    /** @brief The C layer's save config: the path's suffix picks the layout, the residency the
     *      contract - device planes cannot be settlement-probed, so a GPU handle certifies. */
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
int specialized_index<index_type_>::device() const noexcept
    requires is_vram_index<index_type_>
{
    return index.shard(0).device_specs().ordinal;
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
std::size_t specialized_index<index_type_>::storage_requirements() const noexcept {
    return index.storage_requirements(save_config_t {}).value;
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
auto specialized_index<index_type_>::add(std::span<key_t const> keys, vectors_view_t rows, executor_t& executor,
                                         insert_config_t config) noexcept -> insert_result_t {
    return index.add(keys, rows, executor, config);
}
template <typename index_type_>
auto specialized_index<index_type_>::search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                            std::size_t wanted, keys_span_t keys, distances_span_t distances,
                                            counts_span_t counts) noexcept -> search_result_t {
    return index.search(queries, executor, config, wanted, keys, distances, counts);
}
template <typename index_type_>
auto specialized_index<index_type_>::search_if(vectors_view_t queries, executor_t& executor, search_config_t config,
                                               std::size_t wanted, keys_span_t keys, distances_span_t distances,
                                               counts_span_t counts, key_predicate_t& predicate) noexcept
    -> search_result_t
    requires is_ram_index<index_type_>
{
    return index.search(queries, executor, config, wanted, keys, distances, counts, predicate);
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
