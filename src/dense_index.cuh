/**
 *  @brief The dense modality's C surface: one @c std::variant over every compiled backend,
 *      each variant a multi-index of same-typed shards, selected once at construction and
 *      visited once per call.
 *  @author Ash Vardanian
 *  @file src/dense_index.cuh
 *  @date August 14, 2026
 *
 *  This header is the whole type-erasure boundary of the dense modality, compiled twice: as
 *  @c dense.cpp for the portable library and as @c dense.cu for the CUDA one, with
 *  @c USEARCH_BUILD_CUDA deciding whether the VRAM arms join the variant. Residencies share
 *  one variant rather than one per residency, so a handle is a handle whatever traverses it.
 *
 *  Arms are ordered newest-capability-first inside each @c (dtype,kernel) cell, and the
 *  construction cascade tests them in that order: a machine reporting Ice Lake also reports
 *  Haswell and Skylake, so a serial-first walk would shadow every accelerated variant. The
 *  serial variant is unconditional and last, keeping the dispatch non-empty on any platform.
 *
 *  Every variant is a @ref dense_multi_index: many same-typed shards, each under the 32-bit
 *  id ceiling, searched by scatter-gather and filled sequentially. The indexes store no
 *  keys - node ids are their whole vocabulary - so the handle @ref dense_multi_index_t pairs
 *  the chosen variant with the type-free key mapping, one SmashTable of (shard, node)
 *  locators.
 */
#pragma once

#include <cstring> // `std::memcpy`

#include <bit>         // `std::bit_cast` - the C key struct & `uuid_t` share sixteen bytes
#include <memory>      // `std::unique_ptr` - shards hold atomics and never move
#include <new>         // `std::nothrow`
#include <span>        // `std::span` - per-query ranking rows
#include <type_traits> // `std::is_same_v` - the variant-membership audit
#include <variant>     // `std::variant`, `std::visit`

#include <forkunion.h>                     // `fu_runtime_capabilities` - the WFET gate
#include <forkunion/capabilities.hpp>      // `fu::x86_pause_t`, `fu::arm64_wfet_t` & waiter kin
#include <forkunion/types.hpp>             // `fu::dynamic_array` - the shard list
#include <numkong/capabilities.h>          // `nk_capabilities_available`
#include <smashtable/basic_hash_table.hpp> // `st::basic_hash_table` - the reverse key map

#include "usearch/dense_index.h"
#include "usearch/dense/ram_index.hpp"
#include "usearch/dense/nvme_index.hpp"
#if USEARCH_BUILD_CUDA
#include "usearch/dense/vram_index.cuh"
#endif

namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;
namespace fu = ashvardanian::forkunion;
namespace st = ashvardanian::smashtable;

#pragma region Scaffolding

/**
 *  @brief The C scalar vocabulary lowered onto NumKong's - the two enumerations are
 *      independent, so every crossing goes through here rather than through a cast.
 */
inline nk_dtype_t nk_dtype_of(usearch_scalar_t scalar) noexcept {
    switch (scalar) {
    case usearch_scalar_f64_k: return nk_f64_k;
    case usearch_scalar_f32_k: return nk_f32_k;
    case usearch_scalar_bf16_k: return nk_bf16_k;
    case usearch_scalar_f16_k: return nk_f16_k;
    case usearch_scalar_e5m2_k: return nk_e5m2_k;
    case usearch_scalar_e4m3_k: return nk_e4m3_k;
    case usearch_scalar_e3m2_k: return nk_e3m2_k;
    case usearch_scalar_e2m3_k: return nk_e2m3_k;
    case usearch_scalar_i8_k: return nk_i8_k;
    case usearch_scalar_i4x2_k: return nk_i4_k;
    case usearch_scalar_u8_k: return nk_u8_k;
    case usearch_scalar_u4x2_k: return nk_u4_k;
    case usearch_scalar_u1x8_k: return nk_u1_k;
    default: return nk_dtype_unknown_k;
    }
}

/** @brief Codenames of the dispatchable CPU tiers - the vocabulary
 *      @c usearch_dense_capabilities re-composes on demand from the kept mask. */
inline constexpr struct {
    nk_capability_t bit;
    char const* name;
} numkong_capability_names_k[] = {
    {nk_cap_serial_k, "serial"},       {nk_cap_neon_k, "neon"},         {nk_cap_neonsdot_k, "neonsdot"},
    {nk_cap_neonbfdot_k, "neonbfdot"}, {nk_cap_neonfhm_k, "neonfhm"},   {nk_cap_neonfp8_k, "neonfp8"},
    {nk_cap_haswell_k, "haswell"},     {nk_cap_skylake_k, "skylake"},   {nk_cap_icelake_k, "icelake"},
    {nk_cap_genoa_k, "genoa"},         {nk_cap_sapphire_k, "sapphire"}, {nk_cap_sapphireamx_k, "sapphireamx"},
};

/** @brief Whether every bit of @p wanted is present in @p available - the whole-bundle
 *      class match, newest class first, so a stronger core never lands on a weaker rung. */
inline bool has_all(nk_capability_t available, nk_capability_t wanted) noexcept {
    return (available & wanted) == wanted;
}

#pragma region CPU Classes

/**
 *  The dispatch ladders walk @ref us::cpu_class_t rungs - our own CPU families, never a
 *  sister library's feature bits. Each class lowers onto NumKong & ForkUnion masks through
 *  the explicit helpers below; the quads chains pick their best row under the lowered mask
 *  per dtype, and a rung only instantiates where that pick advances - so the ladders stay
 *  sparse per type and the binary carries no duplicate kits.
 */

/** @brief Lowers one CPU class onto NumKong's compute mask - cumulative within a platform,
 *      so the chains pick their best row per dtype under it. */
consteval nk_capability_t numkong_capabilities_of(us::cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case us::cpu_class_t::intel_haswell_k: return nk_cap_serial_k | nk_cap_haswell_k;
    case us::cpu_class_t::intel_skylake_k:
        return numkong_capabilities_of(us::cpu_class_t::intel_haswell_k) | nk_cap_skylake_k;
    case us::cpu_class_t::intel_icelake_k:
        return numkong_capabilities_of(us::cpu_class_t::intel_skylake_k) | nk_cap_icelake_k;
    case us::cpu_class_t::amd_genoa_k:
        return numkong_capabilities_of(us::cpu_class_t::intel_icelake_k) | nk_cap_genoa_k;
    case us::cpu_class_t::aws_g2_k: return nk_cap_serial_k | nk_cap_neon_k;
    case us::cpu_class_t::apple_m2_k:
    case us::cpu_class_t::aws_g5_k:
        return numkong_capabilities_of(us::cpu_class_t::aws_g2_k) | nk_cap_neonsdot_k | nk_cap_neonbfdot_k |
               nk_cap_neonfhm_k;
    case us::cpu_class_t::nvidia_vera_k: return numkong_capabilities_of(us::cpu_class_t::aws_g5_k) | nk_cap_neonfp8_k;
    default: return nk_cap_serial_k;
    }
}

/** @brief Lowers one CPU class onto ForkUnion's schedule mask - the bits the runtime must
 *      confirm before the class's monitored waiter is trusted; empty for spinning classes. */
consteval fu_capabilities_t forkunion_capabilities_of(us::cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case us::cpu_class_t::aws_g5_k:
    case us::cpu_class_t::nvidia_vera_k: return fu_capability_arm64_wfet_k;
    default: return fu_capabilities_unknown_k;
    }
}

/** @brief Whether every compute bit one CPU class lowers onto is present in @p available. */
inline bool has_all(nk_capability_t available, us::cpu_class_t cpu_class) noexcept {
    nk_capability_t const wanted = numkong_capabilities_of(cpu_class);
    return (available & wanted) == wanted;
}

/** @brief Whether the runtime schedule mask confirms one CPU class's wait instruction. */
inline bool schedule_admits(us::cpu_class_t cpu_class) noexcept {
    fu_capabilities_t const wanted = forkunion_capabilities_of(cpu_class);
    return (fu_runtime_capabilities() & wanted) == wanted;
}

/** @brief The wait instruction one CPU class pairs: x86 spins PAUSE, monitored Arm classes
 *      arm WFET, everything else yields - the floor takes the standard one. */
template <us::cpu_class_t cpu_class_>
struct waiter_for_cpu_class {
    using type = fu::standard_yield_t;
};
#if NK_TARGET_X8664_
template <>
struct waiter_for_cpu_class<us::cpu_class_t::intel_haswell_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<us::cpu_class_t::intel_skylake_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<us::cpu_class_t::intel_icelake_k> {
    using type = fu::x86_pause_t;
};
template <>
struct waiter_for_cpu_class<us::cpu_class_t::amd_genoa_k> {
    using type = fu::x86_pause_t;
};
#elif NK_TARGET_ARM64_
template <>
struct waiter_for_cpu_class<us::cpu_class_t::aws_g2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<us::cpu_class_t::apple_m2_k> {
    using type = fu::arm64_yield_t;
};
template <>
struct waiter_for_cpu_class<us::cpu_class_t::aws_g5_k> {
    using type = fu::arm64_wfet_t;
};
template <>
struct waiter_for_cpu_class<us::cpu_class_t::nvidia_vera_k> {
    using type = fu::arm64_wfet_t;
};
#endif

/** @brief The kit one @c (kernel,scalar) chain picks under a class - the compile-time fact
 *      the ladders compare to skip rungs that would duplicate a lower rung's kit. */
template <nk_kernel_kind_t kernel_, typename scalar_type_, us::cpu_class_t cpu_class_>
consteval nk_capability_t picked_kit_for_cpu_class() noexcept {
    return us::dense_quads_chain<kernel_, scalar_type_, numkong_capabilities_of(cpu_class_)>::picked_k;
}

#pragma endregion CPU Classes

/** @brief Codenames of the I/O engine features an NVMe variant dispatched with. */
inline constexpr struct {
    us::io_capability_t bit;
    char const* name;
} io_capability_names_k[] = {
    {us::io_capability_t::fixed_buffers_k, "fixed_buffers"},
    {us::io_capability_t::polled_k, "polled"},
    {us::io_capability_t::lba_addressing_k, "lba_addressing"},
};

/** @brief Codenames of the GPU classes a VRAM variant dispatches onto. */
inline constexpr struct {
    us::gpu_class_t gpu_class;
    char const* name;
} gpu_class_names_k[] = {
    {us::gpu_class_t::nvidia_hopper_k, "nvidia_hopper"},
    {us::gpu_class_t::nvidia_blackwell_k, "nvidia_blackwell"},
    {us::gpu_class_t::nvidia_rubin_k, "nvidia_rubin"},
};

#if USEARCH_BUILD_CUDA
/** @brief Lowers one GPU class onto the CUDA architecture ceiling its kits compile up to;
 *      Rubin rides the newest compiled ceiling until its own kits land. */
consteval us::cuda_arch_t cuda_arch_of(us::gpu_class_t gpu_class) noexcept {
    switch (gpu_class) {
    case us::gpu_class_t::nvidia_hopper_k: return us::cuda_arch_hopper_k;
    case us::gpu_class_t::nvidia_blackwell_k:
    case us::gpu_class_t::nvidia_rubin_k: return us::cuda_arch_blackwell_k;
    default: return us::cuda_arch_ampere_k;
    }
}
#endif

/** @brief The C++ statuses lowered onto the C vocabulary - the C enum is richer, so this
 *      map is total on the engine's side and never called for shim-only failures. */
inline usearch_status_t usearch_status_of(us::status_t status) noexcept {
    switch (status) {
    case us::status_t::success_k: return usearch_success_k;
    case us::status_t::bad_alloc_k: return usearch_error_out_of_memory_k;
    case us::status_t::overflow_k: return usearch_error_overflow_k;
    case us::status_t::uninitialized_k: return usearch_error_uninitialized_k;
    case us::status_t::config_mismatch_k: return usearch_error_config_mismatch_k;
    case us::status_t::device_error_k: return usearch_error_device_k;
    case us::status_t::contended_k: return usearch_error_contended_k;
    case us::status_t::malformed_k: return usearch_error_malformed_input_k;
    default: return usearch_error_internal_k; // A status we don't know is a defect, not a condition.
    }
}

/** @brief The static message every engine status carries; error strings are never allocated. */
inline char const* message_of(us::status_t status) noexcept {
    switch (status) {
    case us::status_t::success_k: return nullptr;
    case us::status_t::bad_alloc_k: return "Allocation failed";
    case us::status_t::overflow_k: return "A hard design ceiling was reached, like the 32-bit id space";
    case us::status_t::uninitialized_k: return "The index needs a successful `reserve` first";
    case us::status_t::config_mismatch_k: return "The requested configuration contradicts the frozen one";
    case us::status_t::device_error_k: return "An accelerator runtime call failed";
    default: return "USearch operation failed";
    }
}

/** @brief The one write path to @p error; the message is null exactly on success. */
inline bool propagate_error(usearch_status_t status, usearch_error_t* error, char const* message) noexcept {
    bool const succeeded = status == usearch_success_k;
    if (error) *error = usearch_error_t {status, succeeded ? nullptr : message};
    return succeeded;
}

/** @brief The engine-side overload: lowers the status, falling back to its static message. */
inline bool propagate_error(us::status_t status, usearch_error_t* error, char const* message = nullptr) noexcept {
    return propagate_error(usearch_status_of(status), error, message ? message : message_of(status));
}

/** @brief Success needs no vocabulary crossing and no message. */
inline void propagate_success(usearch_error_t* error) noexcept {
    if (error) *error = usearch_error_t {usearch_success_k, nullptr};
}

#pragma endregion Scaffolding

#pragma region Multi Index

/** @brief Indexes that accept rows: they expose the reservation & insertion vocabulary.
 *      Read-only designs - checkpoint viewers like the NVMe index - fail this and keep
 *      only the search verbs. */
template <typename index_type_>
concept insertable_index = requires {
    typename index_type_::insert_config_t;
    typename index_type_::reserve_capacity_t;
};

/**
 *  @brief One backend variant: many same-typed shards - each under the 32-bit id ceiling -
 *      viewed as one index. Only what genuinely follows the index type lives here; the key
 *      mappings are the same bytes for every variant and live in @ref dense_multi_index_t.
 *
 *  Shards fill sequentially - the tail is writable, sealing at the id ceiling - and
 *  searches scatter over all of them. Traversal contexts pool under one spin lock: leases
 *  are two array operations per call, cold next to the traversals they scope.
 */
template <typename index_type_>
struct dense_multi_index {
    using index_t = index_type_;
    using context_t = typename index_t::dense_traversal_context_t;
    using scalar_t = typename index_t::scalar_t;

    /** @brief The shards, heap-pinned: an index holds atomics and never moves. */
    fu::dynamic_array<std::unique_ptr<index_t>> shards;

    /** @brief Traversal scratch, one per concurrent walker: an external C caller leases one
     *      for the duration of its call, the internal fan-out one per prong, and a finder
     *      pins one across calls. Sized once from the surface thread count. */
    fu::dynamic_array<std::unique_ptr<context_t>> contexts;
    /** @brief Unleased positions in @c contexts - a plain stack, meaningful only under
     *      @c group_mutex. */
    fu::dynamic_array<std::size_t> free_contexts;
    /** @brief Live depth of the @c free_contexts stack, guarded like the stack itself. */
    std::size_t free_contexts_count = 0;
    /** @brief The one lock over the group's cold bookkeeping - context leases now, shard
     *      sealing & save windows next. Never held across a traversal or a wait: exhaustion
     *      retries outside it, so a pinned finder lease can never deadlock a claimant. */
    fu::spin_mutex<fu::preferred_yield_t> group_mutex;

    /** @brief Leases one context, or null when every one is leased out - the caller
     *      retries at its own cadence, never inside the lock. */
    context_t* claim_context() noexcept {
        group_mutex.lock();
        context_t* claimed = free_contexts_count ? contexts[free_contexts[--free_contexts_count]].get() : nullptr;
        group_mutex.unlock();
        return claimed;
    }

    /** @brief Returns a leased context to the pool. */
    void release_context(context_t& context) noexcept {
        group_mutex.lock();
        for (std::size_t slot = 0; slot != contexts.size(); ++slot)
            if (contexts[slot].get() == &context) {
                free_contexts[free_contexts_count++] = slot;
                break;
            }
        group_mutex.unlock();
    }
};

/**
 *  @brief The one dynamically-typed dense index every binding consumes: a variant over the
 *      compiled multi-index variants, the type-free key mapping beside it, and the geometry the
 *      C surface froze at construction.
 *
 *  Arms exist per wired @c (dtype,kernel,capability) cell - those whose NumKong kernel
 *  kits are adapted - never the full cross product: an unwired cell would not compile, and
 *  advertising it would only move the failure from the build to the caller. The VRAM
 *  siblings join under the CUDA build.
 */
struct dense_multi_index_t {

    /** @brief One RAM alternative: the CPU class lowers onto its NumKong mask and pairs
     *      its waiter through the explicit helpers, so the class alone names the rung. */
    template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, us::cpu_class_t cpu_class_>
    using ram_index = dense_multi_index<us::dense_ram_index<
        dtype_, kernel_, numkong_capabilities_of(cpu_class_), typename waiter_for_cpu_class<cpu_class_>::type,
        std::allocator<std::byte>, std::allocator<std::byte>, std::allocator<std::byte>>>;
#if USEARCH_BUILD_CUDA
    /** @brief The GPU class names a compilation ceiling, never a hardware target: kits for
     *      every generation up to it compile in, and the index dispatches by the actual
     *      device's generation at runtime - so the newest class serves the whole fleet. */
    template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_,
              us::gpu_class_t gpu_class_ = us::gpu_class_t::nvidia_blackwell_k>
    using vram_index = dense_multi_index<us::dense_vram_index<dtype_, kernel_, cuda_arch_of(gpu_class_)>>;
#endif

    /** @brief One NVMe alternative: CPU kits score what the pluggable engine fetches; the
     *      portable POSIX engine is the floor every platform owns. */
    template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_,
              typename engine_type_ = us::dense_posix_sync_engine<fu::preferred_yield_t>>
    using nvme_index = dense_multi_index<us::dense_nvme_index<dtype_, kernel_, capability_, engine_type_>>;

    using variants_t = std::variant<
#if NK_TARGET_X8664_
        // bf16: intel_haswell → intel_skylake → amd_genoa
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_skylake_k>, //
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::amd_genoa_k>,           //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::amd_genoa_k>,     //
        // f16: intel_haswell → intel_skylake
        ram_index<nk_f16_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_f16_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, us::cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_f16_k, nk_kernel_angular_k, us::cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_skylake_k>, //
        // e5m2: intel_haswell → intel_skylake → amd_genoa
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::amd_genoa_k>,           //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::amd_genoa_k>,     //
        // e4m3: intel_haswell → intel_skylake → amd_genoa
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_skylake_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::amd_genoa_k>,           //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::amd_genoa_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::amd_genoa_k>,     //
        // e3m2: intel_haswell → intel_skylake
        ram_index<nk_e3m2_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, us::cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, us::cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_skylake_k>, //
        // e2m3: intel_haswell → intel_skylake
        ram_index<nk_e2m3_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, us::cpu_class_t::intel_skylake_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, us::cpu_class_t::intel_skylake_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_skylake_k>, //
        // i8: intel_haswell → intel_icelake
        ram_index<nk_i8_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, us::cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, us::cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_icelake_k>, //
        // i4: intel_haswell → intel_icelake
        ram_index<nk_i4_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, us::cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, us::cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_icelake_k>, //
        // u8: intel_haswell → intel_icelake
        ram_index<nk_u8_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u8_k, nk_kernel_dot_k, us::cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, us::cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_icelake_k>, //
        // u4: intel_haswell → intel_icelake
        ram_index<nk_u4_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, us::cpu_class_t::intel_haswell_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u4_k, nk_kernel_dot_k, us::cpu_class_t::intel_icelake_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, us::cpu_class_t::intel_icelake_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, us::cpu_class_t::intel_icelake_k>, //
        // u1: intel_haswell → intel_icelake, bit kernels
        ram_index<nk_u1_k, nk_kernel_dot_k, us::cpu_class_t::intel_haswell_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, us::cpu_class_t::intel_haswell_k>, //
        ram_index<nk_u1_k, nk_kernel_dot_k, us::cpu_class_t::intel_icelake_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, us::cpu_class_t::intel_icelake_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, us::cpu_class_t::intel_icelake_k>, //
#elif NK_TARGET_ARM64_
        // bf16 / f16: aws_g2 floor → apple_m2 → aws_g5
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::aws_g2_k>,         //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::aws_g2_k>,     //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g2_k>,   //
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>, //
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,         //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,     //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,   //
        ram_index<nk_f16_k, nk_kernel_dot_k, us::cpu_class_t::aws_g2_k>,          //
        ram_index<nk_f16_k, nk_kernel_angular_k, us::cpu_class_t::aws_g2_k>,      //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g2_k>,    //
        ram_index<nk_f16_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>,  //
        ram_index<nk_f16_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,          //
        ram_index<nk_f16_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,      //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,    //
        // fp8 family: apple_m2 → aws_g5 → nvidia_vera
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, us::cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, us::cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, us::cpu_class_t::nvidia_vera_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,          //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,      //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>,    //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,            //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,        //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,      //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, us::cpu_class_t::nvidia_vera_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, us::cpu_class_t::nvidia_vera_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, us::cpu_class_t::nvidia_vera_k>, //
        // i8 / i4 / u8 / u4: apple_m2 → aws_g5
        ram_index<nk_i8_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,       //
        ram_index<nk_i8_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,   //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,       //
        ram_index<nk_i4_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,   //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>, //
        ram_index<nk_i4_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,       //
        ram_index<nk_u8_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,   //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>, //
        ram_index<nk_u8_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, us::cpu_class_t::apple_m2_k>,       //
        ram_index<nk_u4_k, nk_kernel_angular_k, us::cpu_class_t::apple_m2_k>,   //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, us::cpu_class_t::apple_m2_k>, //
        ram_index<nk_u4_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, us::cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, us::cpu_class_t::aws_g5_k>,   //
        // u1: aws_g2 floor → aws_g5, bit kernels; the wider bundles pick the same NEON kit
        ram_index<nk_u1_k, nk_kernel_dot_k, us::cpu_class_t::aws_g2_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, us::cpu_class_t::aws_g2_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, us::cpu_class_t::aws_g2_k>, //
        ram_index<nk_u1_k, nk_kernel_dot_k, us::cpu_class_t::aws_g5_k>,     //
        ram_index<nk_u1_k, nk_kernel_hamming_k, us::cpu_class_t::aws_g5_k>, //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, us::cpu_class_t::aws_g5_k>, //
#endif
#if USEARCH_BUILD_CUDA
        vram_index<nk_bf16_k, nk_kernel_angular_k>,   //
        vram_index<nk_bf16_k, nk_kernel_euclidean_k>, //
        vram_index<nk_f16_k, nk_kernel_angular_k>,    //
        vram_index<nk_i8_k, nk_kernel_angular_k>,     //
        vram_index<nk_i8_k, nk_kernel_euclidean_k>,   //
        vram_index<nk_i4_k, nk_kernel_angular_k>,     //
        vram_index<nk_u8_k, nk_kernel_angular_k>,     //
        vram_index<nk_u8_k, nk_kernel_euclidean_k>,   //
        vram_index<nk_u1_k, nk_kernel_hamming_k>,     //
        vram_index<nk_u1_k, nk_kernel_jaccard_k>,     //
#endif
        // The serial floor: every wired cell, every platform, the standard yield.
        ram_index<nk_bf16_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,       //
        ram_index<nk_bf16_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_bf16_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>, //
        ram_index<nk_f16_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,        //
        ram_index<nk_f16_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,    //
        ram_index<nk_f16_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>,  //
        ram_index<nk_e5m2_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,       //
        ram_index<nk_e5m2_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_e5m2_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>, //
        ram_index<nk_e4m3_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,       //
        ram_index<nk_e4m3_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_e4m3_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>, //
        ram_index<nk_e3m2_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,       //
        ram_index<nk_e3m2_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_e3m2_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>, //
        ram_index<nk_e2m3_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,       //
        ram_index<nk_e2m3_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_e2m3_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>, //
        ram_index<nk_i8_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,         //
        ram_index<nk_i8_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,     //
        ram_index<nk_i8_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_i4_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,         //
        ram_index<nk_i4_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,     //
        ram_index<nk_i4_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_u8_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,         //
        ram_index<nk_u8_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,     //
        ram_index<nk_u8_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_u4_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,         //
        ram_index<nk_u4_k, nk_kernel_angular_k, us::cpu_class_t::serial_k>,     //
        ram_index<nk_u4_k, nk_kernel_euclidean_k, us::cpu_class_t::serial_k>,   //
        ram_index<nk_u1_k, nk_kernel_dot_k, us::cpu_class_t::serial_k>,         //
        ram_index<nk_u1_k, nk_kernel_hamming_k, us::cpu_class_t::serial_k>,     //
        ram_index<nk_u1_k, nk_kernel_jaccard_k, us::cpu_class_t::serial_k>>;    //

    /** @brief The chosen backend - one alternative of the compiled grid, selected once at
     *      construction and visited once per call. */
    variants_t variants;

    /** @brief The reverse map, key to (shard, node) - the same bytes whatever the variant's
     *      scalar type, so it lives beside the variant rather than inside every arm. */
    dense_key_table_t key_to_locator {};
    /** @brief The frozen dimensionality every stored & queried vector must match. */
    std::size_t dimensions = 0;
    /** @brief The frozen neighbors-per-node bound of every shard's graph. */
    us::connectivity_t connectivity = us::connectivity_32_k;
    /** @brief Where the payload lives and which engine traverses it, frozen at @c init. */
    usearch_residency_t residency = usearch_residency_ram_k;
    /** @brief The compute-tier mask the cascade dispatched with; named on demand by
     *      @c usearch_dense_capabilities, never stored as a string. */
    nk_capability_t numkong_capabilities = nk_cap_serial_k;
    /** @brief The I/O engine feature mask of an NVMe variant; named on demand like the rest. */
    us::io_capability_t io_capabilities = us::io_capability_t::none_k;
    /** @brief The GPU class a VRAM variant compiled kits up to; the actual device's class
     *      replaces it once the first reservation touches a device. */
    us::gpu_class_t gpu_class = us::gpu_class_t::unknown_k;
    /** @brief Where the on-demand composition lands - written under the variant's
     *      @c group_mutex, so concurrent reporters never race the bytes. */
    char capabilities_string[128] = {};

    template <typename variant_type_, typename... arguments_>
    dense_multi_index_t(std::in_place_type_t<variant_type_> tag, arguments_&&... args) noexcept
        : variants(tag, std::forward<arguments_>(args)...) {}
};

/** @brief Whether @p alternative_type_ is listed as an arm of @p variant_type_ - the
 *      membership fact the ladder asserts in both directions against the explicit list. */
template <typename alternative_type_, typename variant_type_>
inline constexpr bool variant_carries_k = false;
template <typename alternative_type_, typename... alternative_types_>
inline constexpr bool variant_carries_k<alternative_type_, std::variant<alternative_types_...>> =
    (std::is_same_v<alternative_type_, alternative_types_> || ...);

#pragma endregion Multi Index

#pragma region Construction

/** @brief Allocates a handle holding the @p variant_type_ backend and publishes the opaque
 *      pointer, freezing the surface geometry from the options. */
template <typename variant_type_>
static usearch_dense_t make_dense_handle(usearch_dense_options_t const* options, nk_capability_t numkong_capabilities,
                                         usearch_error_t* error) noexcept {
    static_assert(variant_carries_k<variant_type_, dense_multi_index_t::variants_t>,
                  "Every constructed backend must be a listed variant alternative");
    auto* multi_index = new (std::nothrow) dense_multi_index_t(std::in_place_type_t<variant_type_>());
    if (!multi_index)
        return propagate_error(usearch_error_out_of_memory_k, error, "Failed to allocate the index"), nullptr;
    multi_index->dimensions = options->dimensions;
    multi_index->connectivity = options->connectivity ? static_cast<us::connectivity_t>(options->connectivity)
                                                      : us::connectivity_32_k;
    multi_index->residency = options->residency;
    multi_index->numkong_capabilities = numkong_capabilities;
    return propagate_success(error), reinterpret_cast<usearch_dense_t>(multi_index);
}

/** @brief The highest class of a pack, or the serial floor when the pack is empty. */
template <us::cpu_class_t... cpu_classes_>
inline constexpr us::cpu_class_t highest_class_k = us::cpu_class_t::serial_k;
template <us::cpu_class_t highest_, us::cpu_class_t... lower_>
inline constexpr us::cpu_class_t highest_class_k<highest_, lower_...> = highest_;

/** @brief Whether one rung is genuinely distinct from the one below: a different picked
 *      kit or a different schedule requirement - the one fact justifying its instantiation. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, us::cpu_class_t upper_class_, us::cpu_class_t lower_class_>
consteval bool rung_advances() noexcept {
    using scalar_t = typename nk::type_for<dtype_>::type;
    return picked_kit_for_cpu_class<kernel_, scalar_t, upper_class_>() !=
               picked_kit_for_cpu_class<kernel_, scalar_t, lower_class_>() ||
           forkunion_capabilities_of(upper_class_) != forkunion_capabilities_of(lower_class_);
}

/** @brief The unconditional serial floor - every ladder's last rung. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_t descend_ram_ladder(usearch_dense_options_t const* options, nk_capability_t,
                                          usearch_error_t* error) noexcept {
    return make_dense_handle<dense_multi_index_t::ram_index<dtype_, kernel_, us::cpu_class_t::serial_k>>(
        options, nk_cap_serial_k, error);
}

/** @brief One distinct rung: taken when the machine reports the class, else falls through. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, us::cpu_class_t cpu_class_, us::cpu_class_t... lower_classes_>
    requires(rung_advances<dtype_, kernel_, cpu_class_, highest_class_k<lower_classes_...>>())
static usearch_dense_t descend_ram_ladder(usearch_dense_options_t const* options, nk_capability_t available,
                                          usearch_error_t* error) noexcept {
    using scalar_t = typename nk::type_for<dtype_>::type;
    if (has_all(available, cpu_class_) && schedule_admits(cpu_class_))
        return make_dense_handle<dense_multi_index_t::ram_index<dtype_, kernel_, cpu_class_>>(
            options, picked_kit_for_cpu_class<kernel_, scalar_t, cpu_class_>(), error);
    return descend_ram_ladder<dtype_, kernel_, lower_classes_...>(options, available, error);
}

/** @brief A rung whose class picks nothing new: skipped without instantiating its index. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, us::cpu_class_t cpu_class_, us::cpu_class_t... lower_classes_>
    requires(!rung_advances<dtype_, kernel_, cpu_class_, highest_class_k<lower_classes_...>>())
static usearch_dense_t descend_ram_ladder(usearch_dense_options_t const* options, nk_capability_t available,
                                          usearch_error_t* error) noexcept {
    static_assert(!variant_carries_k<dense_multi_index_t::ram_index<dtype_, kernel_, cpu_class_>,
                                     dense_multi_index_t::variants_t>,
                  "This rung no longer advances past the one below - drop its arm from the variant list");
    return descend_ram_ladder<dtype_, kernel_, lower_classes_...>(options, available, error);
}

/** @brief Picks the RAM variant for one @c (dtype,kernel) cell: CPU classes newest-first,
 *      each rung compiled only where it advances past the rung below, so the ladders stay
 *      sparse per type and every alternative is genuinely distinct. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_t make_ram_variant(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    nk_capability_t const available = nk_capabilities_available();
#if NK_TARGET_X8664_
    return descend_ram_ladder<dtype_, kernel_, us::cpu_class_t::amd_genoa_k, us::cpu_class_t::intel_icelake_k,
                              us::cpu_class_t::intel_skylake_k, us::cpu_class_t::intel_haswell_k>( //
        options, available, error);
#elif NK_TARGET_ARM64_
    return descend_ram_ladder<dtype_, kernel_, us::cpu_class_t::nvidia_vera_k, us::cpu_class_t::aws_g5_k,
                              us::cpu_class_t::apple_m2_k, us::cpu_class_t::aws_g2_k>( //
        options, available, error);
#else
    return descend_ram_ladder<dtype_, kernel_>(options, available, error);
#endif
}

#if USEARCH_BUILD_CUDA
/** @brief The VRAM cells with adapted CUDA kits - the device half of the variant list. */
constexpr bool vram_cell_wired(nk_dtype_t dtype, nk_kernel_kind_t kernel) noexcept {
    switch (dtype) {
    case nk_i8_k: return kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_u8_k: return kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_f16_k: return kernel == nk_kernel_angular_k;
    case nk_bf16_k: return kernel == nk_kernel_angular_k || kernel == nk_kernel_euclidean_k;
    case nk_i4_k: return kernel == nk_kernel_angular_k;
    case nk_u1_k: return kernel == nk_kernel_hamming_k || kernel == nk_kernel_jaccard_k;
    default: return false;
    }
}

/** @brief The VRAM sibling of @ref make_ram_variant; the device tier is a template argument
 *      of the index itself, so the cell alone selects the variant. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_t make_vram_variant(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    if constexpr (!vram_cell_wired(dtype_, kernel_))
        return propagate_error(usearch_error_unknown_kind_k, error,
                               "No CUDA kit is wired for this (quantization,metric) pair"),
               nullptr;
    else {
        usearch_dense_t handle = make_dense_handle<dense_multi_index_t::vram_index<dtype_, kernel_>>(
            options, nk_capability_t {0}, error);
        if (handle) reinterpret_cast<dense_multi_index_t*>(handle)->gpu_class = us::gpu_class_t::nvidia_blackwell_k;
        return handle;
    }
}
#else
/** @brief The portable build's stand-in: every VRAM request lands on one static message. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_t make_vram_variant(usearch_dense_options_t const*, usearch_error_t* error) noexcept {
    return propagate_error(usearch_error_device_k, error, "This build carries no CUDA backends"), nullptr;
}
#endif

/** @brief Routes one lowered cell onto its residency's constructor. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
static usearch_dense_t make_resident_variant(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->residency) {
    case usearch_residency_ram_k: return make_ram_variant<dtype_, kernel_>(options, error);
    case usearch_residency_vram_k: return make_vram_variant<dtype_, kernel_>(options, error);
    case usearch_residency_nvme_k:
        return propagate_error(usearch_error_config_mismatch_k, error,
                               "The NVMe variants join this dispatch with the normalized batched search"),
               nullptr;
    default: return propagate_error(usearch_error_unknown_kind_k, error, "Unknown residency"), nullptr;
    }
}

/** @brief Routes a numeric dtype's metric onto its three wired kernels. */
template <nk_dtype_t dtype_>
    requires(dtype_ != nk_u1_k)
static usearch_dense_t with_metric(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->metric) {
    case usearch_dense_metric_inner_k: return make_resident_variant<dtype_, nk_kernel_dot_k>(options, error);
    case usearch_dense_metric_angular_k: return make_resident_variant<dtype_, nk_kernel_angular_k>(options, error);
    case usearch_dense_metric_euclidean_k: return make_resident_variant<dtype_, nk_kernel_euclidean_k>(options, error);
    default:
        return propagate_error(usearch_error_unknown_kind_k, error, "This metric needs a bit-packed quantization"),
               nullptr;
    }
}

/** @brief Routes the bit-packed dtype's metric onto the bit kernels. */
template <nk_dtype_t dtype_>
    requires(dtype_ == nk_u1_k)
static usearch_dense_t with_metric(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->metric) {
    case usearch_dense_metric_inner_k: return make_resident_variant<dtype_, nk_kernel_dot_k>(options, error);
    case usearch_dense_metric_hamming_k: return make_resident_variant<dtype_, nk_kernel_hamming_k>(options, error);
    case usearch_dense_metric_jaccard_k: return make_resident_variant<dtype_, nk_kernel_jaccard_k>(options, error);
    default:
        return propagate_error(usearch_error_unknown_kind_k, error,
                               "No kernel kit is wired for this (quantization,metric) pair"),
               nullptr;
    }
}

/** @brief Lowers the runtime @c (quantization,metric) pair onto the wired cells - every
 *      numeric dtype speaks inner, angular & euclidean; the bit dtype speaks inner,
 *      hamming & jaccard. */
static usearch_dense_t with_cell(usearch_dense_options_t const* options, usearch_error_t* error) noexcept {
    switch (options->quantization) {
    case usearch_scalar_bf16_k: return with_metric<nk_bf16_k>(options, error);
    case usearch_scalar_f16_k: return with_metric<nk_f16_k>(options, error);
    case usearch_scalar_e5m2_k: return with_metric<nk_e5m2_k>(options, error);
    case usearch_scalar_e4m3_k: return with_metric<nk_e4m3_k>(options, error);
    case usearch_scalar_e3m2_k: return with_metric<nk_e3m2_k>(options, error);
    case usearch_scalar_e2m3_k: return with_metric<nk_e2m3_k>(options, error);
    case usearch_scalar_i8_k: return with_metric<nk_i8_k>(options, error);
    case usearch_scalar_i4x2_k: return with_metric<nk_i4_k>(options, error);
    case usearch_scalar_u8_k: return with_metric<nk_u8_k>(options, error);
    case usearch_scalar_u4x2_k: return with_metric<nk_u4_k>(options, error);
    case usearch_scalar_u1x8_k: return with_metric<nk_u1_k>(options, error);
    default: return propagate_error(usearch_error_unknown_kind_k, error, "Unsupported quantization"), nullptr;
    }
}

#pragma endregion Construction

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT usearch_dense_t usearch_dense_init(usearch_dense_options_t const* options, usearch_error_t* error) {
    if (!options) return propagate_error(usearch_error_malformed_input_k, error, "Options must not be null"), nullptr;
    return with_cell(options, error);
}

USEARCH_EXPORT void usearch_dense_free(usearch_dense_t index, usearch_error_t* error) {
    delete reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
}

USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_t index, size_t capacity, usearch_error_t* error) {}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT size_t usearch_dense_size(usearch_dense_t index, usearch_error_t* error) {
    auto* multi_index = reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
    return std::visit(
        [](auto& variant) {
            std::size_t total = 0;
            for (std::size_t slot = 0; slot != variant.shards.size(); ++slot) total += variant.shards[slot]->size();
            return total;
        },
        multi_index->variants);
}

USEARCH_EXPORT size_t usearch_dense_connectivity(usearch_dense_t index, usearch_error_t* error) {
    auto* multi_index = reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
    return static_cast<size_t>(multi_index->connectivity);
}

USEARCH_EXPORT size_t usearch_dense_dimensions(usearch_dense_t index, usearch_error_t* error) {
    auto* multi_index = reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
    return multi_index->dimensions;
}

USEARCH_EXPORT char const* usearch_dense_capabilities(usearch_dense_t index, usearch_error_t* error) {
    auto* multi_index = reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
    std::visit(
        [&](auto& variant) {
            variant.group_mutex.lock();
            char* cursor = multi_index->capabilities_string;
            char* const sentinel = cursor + sizeof(multi_index->capabilities_string) - 1;
            auto append = [&](char const* name) {
                if (cursor != multi_index->capabilities_string && cursor != sentinel) *cursor++ = ',';
                while (*name && cursor != sentinel) *cursor++ = *name++;
            };
            for (auto const& named : gpu_class_names_k)
                if (multi_index->gpu_class == named.gpu_class) append(named.name);
            for (auto const& named : numkong_capability_names_k)
                if (multi_index->numkong_capabilities & named.bit) append(named.name);
            for (auto const& named : io_capability_names_k)
                if (multi_index->io_capabilities & named.bit) append(named.name);
            *cursor = '\0';
            variant.group_mutex.unlock();
        },
        multi_index->variants);
    return multi_index->capabilities_string;
}

USEARCH_EXPORT bool usearch_dense_contains(usearch_dense_t index, usearch_key_t key, usearch_error_t* error) {
    auto* multi_index = reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
    return multi_index->key_to_locator.contains(uuid_of(key));
}

USEARCH_EXPORT size_t usearch_dense_count(usearch_dense_t index, usearch_key_t key, usearch_error_t* error) {
    auto* multi_index = reinterpret_cast<dense_multi_index_t*>(index);
    propagate_success(error);
    size_t matches = 0;
    multi_index->key_to_locator.find_atomic(uuid_of(key), [&](auto const&) { ++matches; });
    return matches;
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT size_t usearch_dense_add(              //
    usearch_dense_t index, usearch_key_t const* keys, //
    void const* vectors, size_t count, size_t stride, //
    usearch_scalar_t kind,                            //
    usearch_dense_add_config_t const* config, usearch_dense_add_stats_t* stats, usearch_error_t* error) {}

USEARCH_EXPORT void usearch_dense_search(                             //
    usearch_dense_t index,                                            //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t queries_kind, size_t wanted,                     //
    usearch_dense_search_config_t const* config,                      //
    usearch_dense_search_stats_t* stats,                              //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    size_t* counts, usearch_error_t* error) {}

#pragma endregion Insertion and Search

} // extern "C"
