/**
 *  @file include/usearch/dense/vram_amd_metric.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Device metric kits for the dense VRAM indexes on AMD GPUs - the HIP twin of
 *      @c vram_nvidia_metric.cuh.
 *
 *  A kit binds one storage dtype to one AMD generation, named @b dtype_generation_quads_kit_t, and
 *  the generation slot names the oldest tier whose instructions the kit needs: @c rocm for what
 *  every ROCm device runs, @c cdna3 for @c v_dot4_i32_i8 and @c v_dot2_f32_f16, which MI400
 *  dropped, @c cdna4 for @c v_dot2_f32_bf16 and the OCP fp8 conversions, and @c cdna5 for
 *  @c v_dot4_i32_iu8. Each class unit compiles for its own processor with that tier's
 *  @c NUMKONG_TARGET_* alone, so a tier's kits compile only where its instructions exist, and the
 *  chains pick among them under the class's ceiling. MI300's fp8 conversions read the FNUZ
 *  encodings rather than the OCP ones the vectors matrix stores, so CDNA3 widens fp8 in software, through
 *  NumKong's converters.
 *
 *  Fragments are 16 bytes, rows and the metric are runtime state of the flat view, and the
 *  cross-lane reductions live in @c vram_index.cuh over @ref hip_warp_t, exactly as on NVIDIA; the
 *  arithmetic lives in named device primitives, and a kit binds them. Only HIP-Clang's device pass
 *  sees past the guard: the host classes take their scalar layouts from the dtype alone.
 *
 *  How a wavefront's candidate rows reach shared memory is a separate @b schedule, picked under the
 *  same ceiling: LDS DMA on CDNA3 and CDNA4, register copies elsewhere.
 */
#pragma once

#include <array>       // `std::array` - a fragment's weighted sums
#include <bit>         // `std::bit_cast`, `std::popcount`, `std::countr_zero` - pairs and lane masks
#include <type_traits> // `std::conditional_t` - a mask as wide as the wavefront

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the epilogues switch on it
#include <numkong/types.hpp>      // `nk::bf16_t` & the other scalars the kits load

#include "usearch/utilities/types.hpp"
#include "usearch/dense/types.hpp"

#if NUMKONG_ARCH_ROCM_

#include <hip/hip_runtime.h>     // `__shfl`, `__ballot`, `__lane_id` & the agent-scope atomics
#include <hip/hip_fp16.h>        // `__half2` packed math
#include <numkong/cast/simt.cuh> // `nk_e4m3_to_f32_simt_` & the other software widenings

namespace unum::usearch::dense {

/*  A wavefront spans 32 lanes on the GFX12 parts, MI400, and 64 on the GFX9 ones, MI300 and MI350.
 *  HIP-Clang learns the wavefront size only after code generation, so the width comes from the
 *  processor family each code object is compiled for. */
#if defined(__GFX12__)
inline constexpr u32_t hip_device_wave_lanes_k = 32;
#else
inline constexpr u32_t hip_device_wave_lanes_k = 64;
#endif // defined(__GFX12__)

/**
 *  @brief The AMD warp policy the shared walk code is written against, the twin of
 *      @c cuda_warp_t: wavefronts as wide as the processor's and agent-scope atomics.
 *
 *  A wavefront moves in lockstep, so the votes and shuffles take no participation mask, and
 *  @ref sync is a wave-scoped fence pair around the scheduling barrier. Masks are as wide as the
 *  wavefront, @c u64_t on CDNA3 and CDNA4.
 */
struct hip_warp_t {
    inline static constexpr u32_t width_k = hip_device_wave_lanes_k;
    using mask_t = std::conditional_t<width_k == 64, u64_t, u32_t>;
    inline static constexpr mask_t all_k = ~mask_t(0);

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle(value_type_ value, u32_t source_lane) noexcept {
        return __shfl(value, source_lane, width_k);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_xor(value_type_ value, u32_t lane_mask) noexcept {
        return __shfl_xor(value, lane_mask, width_k);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_up(value_type_ value, u32_t delta) noexcept {
        return __shfl_up(value, delta, width_k);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_down(value_type_ value, u32_t delta,
                                                               u32_t width = width_k) noexcept {
        return __shfl_down(value, delta, width);
    }
    // The predicates are `int`, as the vote intrinsics take them, so any nonzero lane votes.
    static __device__ __forceinline__ mask_t ballot(int predicate) noexcept { return __ballot(predicate); }
    static __device__ __forceinline__ bool any(int predicate) noexcept { return __any(predicate) != 0; }
    static __device__ __forceinline__ bool all(int predicate) noexcept { return __all(predicate) != 0; }
    static __device__ __forceinline__ void sync() noexcept { __syncwarp(); }

    /** The lanes holding the same @p value as this one, this lane included: one ballot per distinct
     *  value, since the ISA has no matching vote. */
    static __device__ __forceinline__ mask_t match_any(u32_t value) noexcept {
        mask_t unmatched = all_k, matched = 0;
        while (unmatched) {
            mask_t const same = ballot(shuffle(value, first(unmatched)) == value);
            if ((same >> __lane_id()) & 1u) matched = same;
            unmatched &= ~same;
        }
        return matched;
    }

    static __device__ __forceinline__ u32_t count(mask_t mask) noexcept { return std::popcount(mask); }

    /** The lowest lane of a non-empty @p mask. */
    static __device__ __forceinline__ u32_t first(mask_t mask) noexcept { return std::countr_zero(mask); }

    /** The lanes strictly below @p lane. */
    static __device__ __forceinline__ mask_t below(u32_t lane) noexcept { return (mask_t(1) << lane) - 1; }

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_relaxed(value_type_& word) noexcept {
        return __hip_atomic_load(&word, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_acquire(value_type_& word) noexcept {
        return __hip_atomic_load(&word, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_relaxed(value_type_& word, value_type_ value) noexcept {
        __hip_atomic_store(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_release(value_type_& word, value_type_ value) noexcept {
        __hip_atomic_store(&word, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ exchange_acquire(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_exchange(&word, value, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_add_relaxed(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_fetch_add(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    /** Orders every store before it ahead of any store after it, for readers across the device. */
    static __device__ __forceinline__ void fence_release() noexcept { __threadfence(); }
};

/** Two halves or two brain floats, as the dot-of-pairs instructions take them. */
using hip_f16x2_t = _Float16 __attribute__((ext_vector_type(2)));
using hip_bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));

#pragma region Metrics

/** Angular distance from a dot and both sides' squared norms, finalized exactly like the CPU's
 *  @c nk_angular_through_f32_from_dot: clamped at zero, and a zero-norm side scores zero against a
 *  vanishing dot and one otherwise. */
__device__ __forceinline__ distance_t hip_angular_through_dot(f32_t dot, f32_t query_sumsq,
                                                              f32_t target_sumsq) noexcept {
    if (query_sumsq > 0.0f && target_sumsq > 0.0f)
        return fmaxf(1.0f - dot * (rsqrtf(query_sumsq) * rsqrtf(target_sumsq)), 0.0f);
    return dot == 0.0f ? 0.0f : 1.0f;
}

/** A numeric kit's distance for @p kernel "through the dot": @b 1-dot for the inner product, the
 *  clamped angular, and the squared Euclidean the walk ranks by. */
template <typename dot_type_>
__device__ __forceinline__ distance_t hip_distance(nk_kernel_kind_t kernel, dot_type_ dot,
                                                   vector_norms<dot_type_> query,
                                                   vector_norms<dot_type_> target) noexcept {
    f32_t const dot_value = static_cast<f32_t>(dot);
    switch (kernel) {
    case nk_kernel_dot_k: return 1.0f - dot_value;
    case nk_kernel_angular_k:
        return hip_angular_through_dot(dot_value, static_cast<f32_t>(query.sumsq), static_cast<f32_t>(target.sumsq));
    default: return static_cast<f32_t>(query.sumsq) + static_cast<f32_t>(target.sumsq) - 2.0f * dot_value;
    }
}

/** Output-boundary finalize: identity unless a metric owes a monotone transform, like Euclidean's
 *  square root. */
__device__ __forceinline__ distance_t hip_convert_to_distance(nk_kernel_kind_t kernel, distance_t ranked) noexcept {
    return kernel == nk_kernel_euclidean_k ? sqrtf(fmaxf(ranked, 0.0f)) : ranked;
}

#pragma endregion Metrics

/** The kits every chain names, declared whether or not their tier is compiled in. */
struct f16_rocm_quads_kit_t;
struct f16_cdna3_quads_kit_t;
struct bf16_rocm_quads_kit_t;
struct bf16_cdna4_quads_kit_t;
struct i8_cdna3_quads_kit_t;
struct i8_cdna5_quads_kit_t;
struct u8_rocm_quads_kit_t;
struct e5m2_rocm_quads_kit_t;
struct e5m2_cdna4_quads_kit_t;
struct e4m3_rocm_quads_kit_t;
struct e4m3_cdna4_quads_kit_t;
struct e3m2_rocm_quads_kit_t;
struct e2m3_rocm_quads_kit_t;
struct i4x2_cdna3_quads_kit_t;
struct i4x2_cdna5_quads_kit_t;
struct u4x2_rocm_quads_kit_t;
struct u1x8_rocm_quads_kit_t;

#pragma region Half Precision Floats

/** A packed @c f16 pair widened to two floats, exactly. */
__device__ __forceinline__ float2 hip_f16x2_to_f32x2(u32_t pair) noexcept {
    return __half22float2(std::bit_cast<__half2>(pair));
}

/** A packed @c bf16 pair widened to two floats, exactly: each is the high half of its float. */
__device__ __forceinline__ float2 hip_bf16x2_to_f32x2(u32_t pair) noexcept {
    return {__uint_as_float(pair << 16), __uint_as_float(pair & 0xFFFF0000u)};
}

/** Eight 16-bit float products of a fragment pair, each pair widened by @p widen_ and fused into
 *  the f32 sum - exact products, since an 11-bit significand squared fits in a float's 24. */
template <float2 (*widen_)(u32_t)>
__device__ __forceinline__ f32_t hip_dot_halves_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values_a = widen_(words_a[word]), values_b = widen_(words_b[word]);
        sum = fmaf(values_a.x, values_b.x, sum);
        sum = fmaf(values_a.y, values_b.y, sum);
    }
    return sum;
}

/** A 16-bit float fragment's sum & sum of squares, each pair widened by @p widen_. */
template <float2 (*widen_)(u32_t)>
__device__ __forceinline__ float2 hip_moments_halves_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = widen_(words[word]);
        moments.x += values.x + values.y;
        moments.y += values.x * values.x + values.y * values.y;
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of a 16-bit float fragment's eight values. */
template <float2 (*widen_)(u32_t)>
__device__ __forceinline__ std::array<f32_t, 8> hip_weighted_halves_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 8> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = widen_(words[word]);
        sums[2 * word] += weight * values.x;
        sums[2 * word + 1] += weight * values.y;
    }
    return sums;
}

/** f16 kit - every ROCm device, widening into f32 multiply-adds. */
struct f16_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_halves_rocm<hip_f16x2_to_f32x2>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_halves_rocm<hip_f16x2_to_f32x2>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_halves_rocm<hip_f16x2_to_f32x2>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

/** bf16 kit - every ROCm device, widening into f32 multiply-adds. */
struct bf16_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_halves_rocm<hip_bf16x2_to_f32x2>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_halves_rocm<hip_bf16x2_to_f32x2>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_halves_rocm<hip_bf16x2_to_f32x2>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#if NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

/** Eight @c f16 products of a fragment pair, two per @c v_dot2_f32_f16 into the f32 sum. */
__device__ __forceinline__ f32_t hip_dot_f16x8_cdna3(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
        sum = __builtin_amdgcn_fdot2(std::bit_cast<hip_f16x2_t>(words_a[word]),
                                     std::bit_cast<hip_f16x2_t>(words_b[word]), sum, false);
    return sum;
}

/** f16 kit - CDNA3 and CDNA4, @c v_dot2_f32_f16. */
struct f16_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_f16x8_cdna3(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_halves_rocm<hip_f16x2_to_f32x2>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_halves_rocm<hip_f16x2_to_f32x2>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA4

/** Eight @c bf16 products of a fragment pair, two per @c v_dot2_f32_bf16 into the f32 sum. */
__device__ __forceinline__ f32_t hip_dot_bf16x8_cdna4(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
        sum = __builtin_amdgcn_fdot2_f32_bf16(std::bit_cast<hip_bf16x2_t>(words_a[word]),
                                              std::bit_cast<hip_bf16x2_t>(words_b[word]), sum, false);
    return sum;
}

/** bf16 kit - CDNA4, @c v_dot2_f32_bf16. */
struct bf16_cdna4_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_bf16x8_cdna4(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_halves_rocm<hip_bf16x2_to_f32x2>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_halves_rocm<hip_bf16x2_to_f32x2>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_CDNA4

#pragma endregion Half Precision Floats

#pragma region Byte Sized Integers

/*  The signed byte dots differ by tier: CDNA3 and CDNA4 multiply signed bytes with
 *  @c v_dot4_i32_i8, and MI400 marks each operand's signedness on @c v_dot4_i32_iu8 instead. A kit
 *  names its dot as a policy, so the moments and the nibble unpacking are written once. */

/** Sixteen unsigned byte products of a fragment pair, one @c v_dot4_u32_u8 per word. */
__device__ __forceinline__ u32_t hip_dot_u8x16_rocm(uint4 a, uint4 b, u32_t sum) noexcept {
    sum = __builtin_amdgcn_udot4(a.x, b.x, sum, false);
    sum = __builtin_amdgcn_udot4(a.y, b.y, sum, false);
    sum = __builtin_amdgcn_udot4(a.z, b.z, sum, false);
    return __builtin_amdgcn_udot4(a.w, b.w, sum, false);
}

/** A @c u8 fragment's sum & sum of squares, through the same dot against splat-one and self. */
__device__ __forceinline__ uint2 hip_moments_u8x16_rocm(uint4 a, uint2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        moments.x = __builtin_amdgcn_udot4(words[word], 0x01010101u, moments.x, false);
        moments.y = __builtin_amdgcn_udot4(words[word], words[word], moments.y, false);
    }
    return moments;
}

/** Sixteen signed byte products of a fragment pair, four per call of @p dot_. */
template <int (*dot_)(u32_t, u32_t, int)>
__device__ __forceinline__ int hip_dot_i8x16(uint4 a, uint4 b, int sum) noexcept {
    sum = dot_(a.x, b.x, sum);
    sum = dot_(a.y, b.y, sum);
    sum = dot_(a.z, b.z, sum);
    return dot_(a.w, b.w, sum);
}

/** An @c i8 fragment's sum & sum of squares, through @p dot_ against splat-one and self. */
template <int (*dot_)(u32_t, u32_t, int)>
__device__ __forceinline__ int2 hip_moments_i8x16(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        moments.x = dot_(words[word], 0x01010101u, moments.x);
        moments.y = dot_(words[word], words[word], moments.y);
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c i8 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_i8x16_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t byte = 0; byte != 16; ++byte)
        sums[byte] += weight * static_cast<f32_t>(static_cast<i8_t>(words[byte / 4] >> (8 * (byte % 4))));
    return sums;
}

/** Returns @p sums plus @p weight times each of a @c u8 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_u8x16_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t byte = 0; byte != 16; ++byte)
        sums[byte] += weight * static_cast<f32_t>(static_cast<u8_t>(words[byte / 4] >> (8 * (byte % 4))));
    return sums;
}

#if NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

/** Four signed byte products added to @p sum by @c v_dot4_i32_i8. */
__device__ __forceinline__ int hip_dot_i8x4_cdna3(u32_t a, u32_t b, int sum) noexcept {
    return __builtin_amdgcn_sdot4(static_cast<int>(a), static_cast<int>(b), sum, false);
}

/** i8 kit - CDNA3 and CDNA4, @c v_dot4_i32_i8. */
struct i8_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i8x16<hip_dot_i8x4_cdna3>(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i8x16<hip_dot_i8x4_cdna3>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_i8x16_rocm(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5

/** Four signed byte products added to @p sum by @c v_dot4_i32_iu8, both operands marked signed. */
__device__ __forceinline__ int hip_dot_i8x4_cdna5(u32_t a, u32_t b, int sum) noexcept {
    return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), sum, false);
}

/** i8 kit - CDNA5, @c v_dot4_i32_iu8. */
struct i8_cdna5_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i8x16<hip_dot_i8x4_cdna5>(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i8x16<hip_dot_i8x4_cdna5>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_i8x16_rocm(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_CDNA5

/** u8 kit - every ROCm device, @c v_dot4_u32_u8. */
struct u8_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::u8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_u8x16_rocm(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return hip_moments_u8x16_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_u8x16_rocm(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Byte Sized Integers

#pragma region Minifloats

/*  No generation has fp8 or fp6 arithmetic outside the matrix cores, so every minifloat kit widens
 *  byte pairs and accumulates in f32, and the tiers' kits differ only in the widening. */

/** Sixteen minifloat products of a fragment pair, each byte pair widened by @p widening_type_. */
template <typename widening_type_>
__device__ __forceinline__ f32_t hip_dot_minifloats(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = widening_type_::pair_to_floats(words_a[word] >> (16 * pair));
            float2 const values_b = widening_type_::pair_to_floats(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** A minifloat fragment's sum & sum of squares, each byte pair widened by @p widening_type_. */
template <typename widening_type_>
__device__ __forceinline__ float2 hip_moments_minifloats(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = widening_type_::pair_to_floats(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** Returns @p sums plus @p weight times each of a minifloat fragment's sixteen values. */
template <typename widening_type_>
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_minifloats( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = widening_type_::pair_to_floats(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** Every ROCm device widens a minifloat pair in software, one byte at a time through @p convert_, a
 *  NumKong converter, so CDNA3's FNUZ conversions never misread an OCP code. */
template <void (*convert_)(unsigned char const*, nk_f32_t*)>
struct hip_software_widening {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        unsigned char const codes[2] = {static_cast<unsigned char>(packed_pair),
                                        static_cast<unsigned char>(packed_pair >> 8)};
        float2 values;
        convert_(&codes[0], &values.x);
        convert_(&codes[1], &values.y);
        return values;
    }
};

using e5m2_rocm_widening_t = hip_software_widening<nk_e5m2_to_f32_simt_>;
using e4m3_rocm_widening_t = hip_software_widening<nk_e4m3_to_f32_simt_>;
using e3m2_rocm_widening_t = hip_software_widening<nk_e3m2_to_f32_simt_>;
using e2m3_rocm_widening_t = hip_software_widening<nk_e2m3_to_f32_simt_>;

/** The quads kit of a minifloat scalar @p scalar_type_ widened by @p widening_type_ at
 *  @p capability_. */
template <typename scalar_type_, typename widening_type_, nk_capability_t capability_>
struct hip_minifloats_quads_kit {
    inline static constexpr nk_capability_t capability_k = capability_;
    using warp_t = hip_warp_t;
    using scalar_t = scalar_type_;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_minifloats<widening_type_>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_minifloats<widening_type_>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_minifloats<widening_type_>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

/** e5m2 kit - every ROCm device, widening in software. */
struct e5m2_rocm_quads_kit_t : hip_minifloats_quads_kit<nk::e5m2_t, e5m2_rocm_widening_t, nk_cap_rocm_k> {};

/** e4m3 kit - every ROCm device, widening in software. */
struct e4m3_rocm_quads_kit_t : hip_minifloats_quads_kit<nk::e4m3_t, e4m3_rocm_widening_t, nk_cap_rocm_k> {};

/** e3m2 kit - every ROCm device, widening in software. */
struct e3m2_rocm_quads_kit_t : hip_minifloats_quads_kit<nk::e3m2_t, e3m2_rocm_widening_t, nk_cap_rocm_k> {};

/** e2m3 kit - every ROCm device, widening in software. */
struct e2m3_rocm_quads_kit_t : hip_minifloats_quads_kit<nk::e2m3_t, e2m3_rocm_widening_t, nk_cap_rocm_k> {};

#if NUMKONG_TARGET_CDNA4 || NUMKONG_TARGET_CDNA5

/** CDNA4 and every tier since widen an OCP @c e4m3 pair in one @c v_cvt_pk_f32_fp8. */
struct e4m3_cdna4_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        auto const values = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(packed_pair), false);
        return {values[0], values[1]};
    }
};

/** The @c e5m2 twin of @ref e4m3_cdna4_widening_t, through @c v_cvt_pk_f32_bf8. */
struct e5m2_cdna4_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        auto const values = __builtin_amdgcn_cvt_pk_f32_bf8(static_cast<int>(packed_pair), false);
        return {values[0], values[1]};
    }
};

/** e5m2 kit - CDNA4 and CDNA5, widening through the conversion. */
struct e5m2_cdna4_quads_kit_t : hip_minifloats_quads_kit<nk::e5m2_t, e5m2_cdna4_widening_t, nk_cap_cdna4_k> {};

/** e4m3 kit - CDNA4 and CDNA5, widening through the conversion. */
struct e4m3_cdna4_quads_kit_t : hip_minifloats_quads_kit<nk::e4m3_t, e4m3_cdna4_widening_t, nk_cap_cdna4_k> {};

#endif // NUMKONG_TARGET_CDNA4 || NUMKONG_TARGET_CDNA5

#pragma endregion Minifloats

#pragma region Packed Nibbles

/** Unpacks a word of eight signed nibbles into two byte-lane words, sign-extended through an
 *  xor-add-xor bias, since a word-wide subtraction would borrow across lanes. */
__device__ __forceinline__ uint2 hip_unpack_i4x8_rocm(u32_t word) noexcept {
    u32_t const low = (((word & 0x0F0F0F0Fu) ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
    u32_t const high = ((((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
    return {low, high};
}

/** Unpacks a word of eight unsigned nibbles into two byte-lane words. */
__device__ __forceinline__ uint2 hip_unpack_u4x8_rocm(u32_t word) noexcept {
    return {word & 0x0F0F0F0Fu, (word >> 4) & 0x0F0F0F0Fu};
}

/** Thirty-two signed @c i4 products of a fragment pair, two calls of @p dot_ per source word. */
template <int (*dot_)(u32_t, u32_t, int)>
__device__ __forceinline__ int hip_dot_i4x32(uint4 a, uint4 b, int sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = hip_unpack_i4x8_rocm(words_a[word]);
        uint2 const unpacked_b = hip_unpack_i4x8_rocm(words_b[word]);
        sum = dot_(unpacked_a.x, unpacked_b.x, sum);
        sum = dot_(unpacked_a.y, unpacked_b.y, sum);
    }
    return sum;
}

/** An @c i4 fragment's sum & sum of squares. */
template <int (*dot_)(u32_t, u32_t, int)>
__device__ __forceinline__ int2 hip_moments_i4x32(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_i4x8_rocm(words[word]);
        moments.x = dot_(unpacked.x, 0x01010101u, moments.x);
        moments.x = dot_(unpacked.y, 0x01010101u, moments.x);
        moments.y = dot_(unpacked.x, unpacked.x, moments.y);
        moments.y = dot_(unpacked.y, unpacked.y, moments.y);
    }
    return moments;
}

/** Thirty-two unsigned @c u4 products of a fragment pair, two @c v_dot4_u32_u8 per source word. */
__device__ __forceinline__ u32_t hip_dot_u4x32_rocm(uint4 a, uint4 b, u32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = hip_unpack_u4x8_rocm(words_a[word]);
        uint2 const unpacked_b = hip_unpack_u4x8_rocm(words_b[word]);
        sum = __builtin_amdgcn_udot4(unpacked_a.x, unpacked_b.x, sum, false);
        sum = __builtin_amdgcn_udot4(unpacked_a.y, unpacked_b.y, sum, false);
    }
    return sum;
}

/** A @c u4 fragment's sum & sum of squares. */
__device__ __forceinline__ uint2 hip_moments_u4x32_rocm(uint4 a, uint2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_u4x8_rocm(words[word]);
        moments.x = __builtin_amdgcn_udot4(unpacked.x, 0x01010101u, moments.x, false);
        moments.x = __builtin_amdgcn_udot4(unpacked.y, 0x01010101u, moments.x, false);
        moments.y = __builtin_amdgcn_udot4(unpacked.x, unpacked.x, moments.y, false);
        moments.y = __builtin_amdgcn_udot4(unpacked.y, unpacked.y, moments.y, false);
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c i4 fragment's thirty-two values, through the
 *  same unpacking; the even column of a byte is its high nibble. */
__device__ __forceinline__ std::array<f32_t, 32> hip_weighted_i4x32_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 32> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_i4x8_rocm(words[word]);
#pragma unroll
        for (u32_t byte = 0; byte != 4; ++byte) {
            sums[8 * word + 2 * byte] += weight * static_cast<f32_t>(static_cast<i8_t>(unpacked.y >> (8 * byte)));
            sums[8 * word + 2 * byte + 1] += weight * static_cast<f32_t>(static_cast<i8_t>(unpacked.x >> (8 * byte)));
        }
    }
    return sums;
}

/** The @c u4 twin of @ref hip_weighted_i4x32_rocm. */
__device__ __forceinline__ std::array<f32_t, 32> hip_weighted_u4x32_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 32> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_u4x8_rocm(words[word]);
#pragma unroll
        for (u32_t byte = 0; byte != 4; ++byte) {
            sums[8 * word + 2 * byte] += weight * static_cast<f32_t>(static_cast<u8_t>(unpacked.y >> (8 * byte)));
            sums[8 * word + 2 * byte + 1] += weight * static_cast<f32_t>(static_cast<u8_t>(unpacked.x >> (8 * byte)));
        }
    }
    return sums;
}

/** The quads kit of signed nibbles over the signed byte dot @p dot_, at @p capability_. */
template <int (*dot_)(u32_t, u32_t, int), nk_capability_t capability_>
struct hip_i4x2_quads_kit {
    inline static constexpr nk_capability_t capability_k = capability_;
    using warp_t = hip_warp_t;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i4x32<dot_>(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i4x32<dot_>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_i4x32_rocm(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#if NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

/** i4x2 kit - CDNA3 and CDNA4, the byte engine over unpacked nibbles. */
struct i4x2_cdna3_quads_kit_t : hip_i4x2_quads_kit<hip_dot_i8x4_cdna3, nk_cap_cdna3_k> {};

#endif // NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5

/** i4x2 kit - CDNA5, the byte engine over unpacked nibbles. */
struct i4x2_cdna5_quads_kit_t : hip_i4x2_quads_kit<hip_dot_i8x4_cdna5, nk_cap_cdna5_k> {};

#endif // NUMKONG_TARGET_CDNA5

/** u4x2 kit - every ROCm device, the byte engine over unpacked nibbles. */
struct u4x2_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::u4x2_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_u4x32_rocm(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return hip_moments_u4x32_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_u4x32_rocm(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return hip_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return hip_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Packed Nibbles

#pragma region Packed Bits

/** The intersection popcount of a fragment pair via two 64-bit popcounts. */
__device__ __forceinline__ u32_t hip_intersect_u1x128_rocm(uint4 a, uint4 b) noexcept {
    u64_t const low = (static_cast<u64_t>(a.y) << 32 | a.x) & (static_cast<u64_t>(b.y) << 32 | b.x);
    u64_t const high = (static_cast<u64_t>(a.w) << 32 | a.z) & (static_cast<u64_t>(b.w) << 32 | b.z);
    return static_cast<u32_t>(std::popcount(low) + std::popcount(high));
}

/** Returns @p sums plus @p weight times each of a fragment's 128 bits, most significant bit of each
 *  byte first, as NumKong's casts order them. */
__device__ __forceinline__ std::array<f32_t, 128> hip_weighted_u1x128_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 128> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t bit = 0; bit != 128; ++bit)
        sums[bit] += weight * static_cast<f32_t>((words[bit / 32] >> (8 * (bit / 8 % 4) + 7 - bit % 8)) & 1u);
    return sums;
}

/** u1x8 kit - every ROCm device: the "dot" is the intersection popcount, and the norms store each
 *  row's set-bit count, the currency of Jaccard's union, mirrored into both fields. */
struct u1x8_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    using warp_t = hip_warp_t;
    using scalar_t = nk::u1x8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 128>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return sum + hip_intersect_u1x128_rocm(a, b);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        moments.x += hip_intersect_u1x128_rocm(a, a);
        return moments;
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_u1x128_rocm(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.x};
    }

    /** Hamming is |a| + |b| - 2·intersection; Jaccard is 1 - intersection / union, and two empty
     *  sets are identical, distance zero. */
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        switch (kernel) {
        case nk_kernel_dot_k: return 1.0f - static_cast<f32_t>(dot);
        case nk_kernel_hamming_k: return static_cast<f32_t>(query.sum + target.sum - 2 * dot);
        default: {
            u32_t const unions = query.sum + target.sum - dot;
            return unions ? 1.0f - static_cast<f32_t>(dot) / static_cast<f32_t>(unions) : 0.0f;
        }
        }
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t, distance_t ranked) noexcept {
        return ranked;
    }
};

#pragma endregion Packed Bits

#pragma region Schedules

/** The staging every ROCm device runs: each lane copies the fragments it strides over through its
 *  registers, and one wavefront barrier lands them before the staging returns. */
struct rocm_registers_schedule_t {
    inline static constexpr u32_t stages_k = 1;

    /** Nothing shared: the copies are plain stores. */
    struct shared_t {};

    __device__ __forceinline__ rocm_registers_schedule_t(shared_t&, u32_t) noexcept {}

    template <typename storage_, typename cache_>
    static __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids,
                                                      u32_t rows, u32_t lane) noexcept {
        for (u32_t word = lane; word < rows * storage.row_fragments; word += hip_warp_t::width_k) {
            u32_t const row = storage.row_of_word(word), fragment = storage.fragment_of_word(word, row);
            cache.at(row, fragment) = storage.vector_for(ids[row], fragment);
        }
        hip_warp_t::sync();
    }

    __device__ __forceinline__ void await_rows(u32_t) noexcept {}
};

/** @p pointer as the global address space the LDS DMA reads, the same 64-bit value. */
__device__ __forceinline__ __attribute__((address_space(1))) void* hip_global_pointer(void* pointer) noexcept {
    return (__attribute__((address_space(1))) void*)pointer;
}

/** @p pointer as the shared address space the LDS DMA writes, its 32-bit offset. */
__device__ __forceinline__ __attribute__((address_space(3))) void* hip_shared_pointer(void* pointer) noexcept {
    return (__attribute__((address_space(3))) void*)pointer;
}

/**
 *  @brief The CDNA3 and CDNA4 staging: LDS DMA moves @p lane_bytes_ per lane from global memory
 *      straight into the shared cache, past the registers - a 32-bit word on CDNA3, a whole
 *      fragment on CDNA4.
 *
 *  One instruction lands a wavefront's units side by side above one wavefront-uniform base, and the
 *  cache rows lie back to back, so a unit's index in the staged rows is its offset in the cache. The
 *  copies count against @c vmcnt like loads, which one wait drains. MI400 keeps the register copies
 *  until its asynchronous LDS loads are wired.
 */
template <u32_t lane_bytes_>
struct hip_lds_schedule {
    inline static constexpr u32_t stages_k = 1;

    /** Nothing shared: the copies count on the wavefront's own @c vmcnt. */
    struct shared_t {};

    __device__ __forceinline__ hip_lds_schedule(shared_t&, u32_t) noexcept {}

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        if (!__builtin_amdgcn_is_shared(cache.row(0)))
            return rocm_registers_schedule_t::stage_rows(storage, cache, ids, rows, lane);
        constexpr u32_t units_per_fragment_k = sizeof(typename storage_::fragment_t) / lane_bytes_;
        u32_t const row_units = storage.row_fragments * units_per_fragment_k, units = rows * row_units;
        byte_t* const cache_bytes = reinterpret_cast<byte_t*>(cache.row(0));
        for (u32_t base = 0; base < units; base += hip_warp_t::width_k) {
            u32_t const unit = base + lane;
            if (unit >= units) break;
            u32_t const row = storage.row_of_word(unit / units_per_fragment_k);
            byte_t* const source = reinterpret_cast<byte_t*>(storage.vector_span_for(ids[row])) +
                                   (unit - row * row_units) * lane_bytes_;
            // A size of dependent type defers the builtin's constant check to instantiation, where it is one.
            __builtin_amdgcn_global_load_lds(hip_global_pointer(source),
                                             hip_shared_pointer(cache_bytes + base * lane_bytes_),
                                             static_cast<decltype(storage.row_fragments)>(lane_bytes_), 0, 0);
        }
    }

    __device__ __forceinline__ void await_rows(u32_t) noexcept { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
};

using cdna3_lds_schedule_t = hip_lds_schedule<sizeof(u32_t)>;
using cdna4_lds_schedule_t = hip_lds_schedule<sizeof(uint4)>;

#pragma endregion Schedules

#pragma region Chains

/*  A class's ceiling carries every older tier's bit, so each chain lists the newest tier first:
 *  MI400 keeps the CDNA4 conversions but drops CDNA3's signed byte and half dots, and its unit
 *  compiles none of CDNA3's kits. */

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                                    //
        f16_rocm_quads_kit_t,                                                                      //
        chain_layer<NUMKONG_TARGET_CDNA5 && (caps_ & nk_cap_cdna5_k) != 0, f16_rocm_quads_kit_t>,  //
        chain_layer<NUMKONG_TARGET_CDNA4 && (caps_ & nk_cap_cdna4_k) != 0, f16_cdna3_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_CDNA3 && (caps_ & nk_cap_cdna3_k) != 0, f16_cdna3_quads_kit_t>  //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                    //
        bf16_rocm_quads_kit_t,                                                                     //
        chain_layer<NUMKONG_TARGET_CDNA4 && (caps_ & nk_cap_cdna4_k) != 0, bf16_cdna4_quads_kit_t> //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::i8_t, caps_> {
    using kit_t = typename first_available_kit<                                                  //
        i8_cdna3_quads_kit_t,                                                                    //
        chain_layer<NUMKONG_TARGET_CDNA5 && (caps_ & nk_cap_cdna5_k) != 0, i8_cdna5_quads_kit_t> //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::u8_t, caps_> {
    using kit_t = u8_rocm_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                     //
        e5m2_rocm_quads_kit_t,                                                                      //
        chain_layer<NUMKONG_TARGET_CDNA5 && (caps_ & nk_cap_cdna5_k) != 0, e5m2_cdna4_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_CDNA4 && (caps_ & nk_cap_cdna4_k) != 0, e5m2_cdna4_quads_kit_t>  //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                     //
        e4m3_rocm_quads_kit_t,                                                                      //
        chain_layer<NUMKONG_TARGET_CDNA5 && (caps_ & nk_cap_cdna5_k) != 0, e4m3_cdna4_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_CDNA4 && (caps_ & nk_cap_cdna4_k) != 0, e4m3_cdna4_quads_kit_t>  //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::e3m2_t, caps_> {
    using kit_t = e3m2_rocm_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::e2m3_t, caps_> {
    using kit_t = e2m3_rocm_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::i4x2_t, caps_> {
    using kit_t = typename first_available_kit<                                                    //
        i4x2_cdna3_quads_kit_t,                                                                    //
        chain_layer<NUMKONG_TARGET_CDNA5 && (caps_ & nk_cap_cdna5_k) != 0, i4x2_cdna5_quads_kit_t> //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::u4x2_t, caps_> {
    using kit_t = u4x2_rocm_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_chain<nk::u1x8_t, caps_> {
    using kit_t = u1x8_rocm_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

/** The staging schedule under @p caps_: the LDS DMA copies on CDNA3 and CDNA4, a fragment per lane
 *  where CDNA4 allows it, and the register copies on MI400 and every other ROCm device. */
template <nk_capability_t caps_>
using hip_schedule_chain = typename first_available_kit<                   //
    rocm_registers_schedule_t,                                             //
    chain_layer<(caps_ & nk_cap_cdna5_k) != 0, rocm_registers_schedule_t>, //
    chain_layer<(caps_ & nk_cap_cdna4_k) != 0, cdna4_lds_schedule_t>,      //
    chain_layer<(caps_ & nk_cap_cdna3_k) != 0, cdna3_lds_schedule_t>       //
    >::type;

#pragma endregion Chains

} // namespace unum::usearch::dense

#endif // NUMKONG_ARCH_ROCM_
