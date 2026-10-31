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
 *  @c NUMKONG_TARGET_* alone, so a tier's kits compile only where its instructions exist and are
 *  stubs elsewhere. Every kit states its @c capability_k and @c compiled_k, and a ladder picks the
 *  first both admit through @c highest_kit. MI300's fp8 conversions read the FNUZ encodings rather
 *  than the OCP ones the vectors matrix stores, so CDNA3 widens fp8 in software, through NumKong's
 *  converters.
 *
 *  Fragments are 16 bytes, rows and the metric are runtime state of the flat view, and the
 *  cross-lane reductions live in @c vram_index.cuh over the schedule's @ref hip_warp, exactly as on
 *  NVIDIA. The arithmetic lives in named device primitives, and a kit binds them. Only HIP units
 *  include it: the host classes take their scalar layouts from the dtype alone.
 *
 *  How a wavefront's candidate rows reach shared memory is a separate @b schedule, picked by
 *  @c schedule_ladder the same way: LDS DMA on CDNA3 and CDNA4, register copies elsewhere. Each
 *  schedule names the warp policy of its capability, @c hip_wave_lanes, so the same kit runs
 *  64-lane wavefronts on CDNA3 and CDNA4 and 32-lane ones on CDNA5.
 */
#pragma once

#include <array>       // `std::array` - a fragment's weighted sums
#include <bit>         // `std::bit_cast`, `std::popcount`, `std::countr_zero` - pairs and lane masks
#include <type_traits> // `std::conditional_t` - a mask as wide as the wavefront

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the epilogues switch on it
#include <numkong/types.hpp>      // `nk::bf16_t` & the other scalars the kits load

#include "usearch/utilities/types.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_index.cuh"

#include <hip/hip_fp16.h>        // `__half2` packed math
#include <numkong/cast/simt.cuh> // `nk_e4m3_to_f32_simt_` & the other software widenings
#include <numkong/dots/rocm.cuh> // `nk_dot_i8x4_rocm_` - the portable signed byte dot

#include "usearch/utilities/vram_amd_device.hip" // `hip_warp`, each schedule's warp policy

namespace unum::usearch::dense {

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

#pragma region Half Precision Floats

/** A packed @c f16 pair widened to two floats, exactly. */
__device__ __forceinline__ float2 hip_f16x2_to_f32x2_rocm(u32_t pair) noexcept {
    return __half22float2(std::bit_cast<__half2>(pair));
}

/** A packed @c bf16 pair widened to two floats, exactly: each is the high half of its float. */
__device__ __forceinline__ float2 hip_bf16x2_to_f32x2_rocm(u32_t pair) noexcept {
    return {__uint_as_float(pair << 16), __uint_as_float(pair & 0xFFFF0000u)};
}

/** Eight @c f16 products of a fragment pair, each pair widened and fused into the f32 sum - exact
 *  products, since an 11-bit significand squared fits in a float's 24. */
__device__ __forceinline__ f32_t hip_dot_f16x8_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values_a = hip_f16x2_to_f32x2_rocm(words_a[word]);
        float2 const values_b = hip_f16x2_to_f32x2_rocm(words_b[word]);
        sum = fmaf(values_a.x, values_b.x, sum);
        sum = fmaf(values_a.y, values_b.y, sum);
    }
    return sum;
}

/** An @c f16 fragment's sum & sum of squares, widened to f32. */
__device__ __forceinline__ float2 hip_moments_f16x8_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = hip_f16x2_to_f32x2_rocm(words[word]);
        moments.x += values.x + values.y;
        moments.y += values.x * values.x + values.y * values.y;
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c f16 fragment's eight values. */
__device__ __forceinline__ std::array<f32_t, 8> hip_weighted_f16x8_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 8> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = hip_f16x2_to_f32x2_rocm(words[word]);
        sums[2 * word] += weight * values.x;
        sums[2 * word + 1] += weight * values.y;
    }
    return sums;
}

/** The @c bf16 twin of @ref hip_dot_f16x8_rocm. */
__device__ __forceinline__ f32_t hip_dot_bf16x8_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values_a = hip_bf16x2_to_f32x2_rocm(words_a[word]);
        float2 const values_b = hip_bf16x2_to_f32x2_rocm(words_b[word]);
        sum = fmaf(values_a.x, values_b.x, sum);
        sum = fmaf(values_a.y, values_b.y, sum);
    }
    return sum;
}

/** The @c bf16 twin of @ref hip_moments_f16x8_rocm. */
__device__ __forceinline__ float2 hip_moments_bf16x8_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = hip_bf16x2_to_f32x2_rocm(words[word]);
        moments.x += values.x + values.y;
        moments.y += values.x * values.x + values.y * values.y;
    }
    return moments;
}

/** The @c bf16 twin of @ref hip_weighted_f16x8_rocm. */
__device__ __forceinline__ std::array<f32_t, 8> hip_weighted_bf16x8_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 8> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = hip_bf16x2_to_f32x2_rocm(words[word]);
        sums[2 * word] += weight * values.x;
        sums[2 * word + 1] += weight * values.y;
    }
    return sums;
}

/** f16 kit - every ROCm device, widening into f32 multiply-adds. */
struct f16_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_f16x8_rocm(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_f16x8_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_f16x8_rocm(a, weight, sums);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_bf16x8_rocm(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_bf16x8_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_bf16x8_rocm(a, weight, sums);
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

/** f16 kit - CDNA3 and CDNA4, @c v_dot2_f32_f16, which MI400 dropped. */
struct f16_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = true;
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
        return hip_moments_f16x8_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_f16x8_rocm(a, weight, sums);
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

#else

/** The kit this unit cannot carry - CDNA3's half dot, absent before CDNA3 and on MI400 - kept so
 *  its ladder names it whatever the unit compiles. */
struct f16_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = false;
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
    inline static constexpr bool compiled_k = true;
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
        return hip_moments_bf16x8_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_bf16x8_rocm(a, weight, sums);
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

#else

/** The kit this unit cannot carry - CDNA4's brain-float dot, which only CDNA4 units compile - kept
 *  so its ladder names it whatever the unit compiles. */
struct bf16_cdna4_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    inline static constexpr bool compiled_k = false;
};

#endif // NUMKONG_TARGET_CDNA4

#pragma endregion Half Precision Floats

#pragma region Byte Sized Integers

/*  The signed byte dots differ by tier: every ROCm device has NumKong's portable byte dot, CDNA3
 *  and CDNA4 multiply signed bytes with @c v_dot4_i32_i8, and MI400 marks each operand's signedness
 *  on @c v_dot4_i32_iu8 instead. Each tier's kit calls its own dot, moments and nibble loops. */

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

/** Four signed byte products added to @p sum by NumKong's portable byte dot. */
__device__ __forceinline__ int hip_dot_i8x4_rocm(u32_t a, u32_t b, int sum) noexcept {
    return nk_dot_i8x4_rocm_(a, b, sum);
}

/** Sixteen signed @c i8 products of a fragment pair, four per @ref hip_dot_i8x4_rocm. */
__device__ __forceinline__ int hip_dot_i8x16_rocm(uint4 a, uint4 b, int sum) noexcept {
    sum = hip_dot_i8x4_rocm(a.x, b.x, sum);
    sum = hip_dot_i8x4_rocm(a.y, b.y, sum);
    sum = hip_dot_i8x4_rocm(a.z, b.z, sum);
    return hip_dot_i8x4_rocm(a.w, b.w, sum);
}

/** An @c i8 fragment's sum & sum of squares, through @ref hip_dot_i8x4_rocm against splat-one and
 *  self. */
__device__ __forceinline__ int2 hip_moments_i8x16_rocm(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        moments.x = hip_dot_i8x4_rocm(words[word], 0x01010101u, moments.x);
        moments.y = hip_dot_i8x4_rocm(words[word], words[word], moments.y);
    }
    return moments;
}

/** i8 kit - every ROCm device, NumKong's portable byte dot. */
struct i8_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i8x16_rocm(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i8x16_rocm(a, moments);
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

#if NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

/** Four signed byte products added to @p sum by @c v_dot4_i32_i8. */
__device__ __forceinline__ int hip_dot_i8x4_cdna3(u32_t a, u32_t b, int sum) noexcept {
    return __builtin_amdgcn_sdot4(static_cast<int>(a), static_cast<int>(b), sum, false);
}

/** Sixteen signed @c i8 products of a fragment pair, four per @c v_dot4_i32_i8. */
__device__ __forceinline__ int hip_dot_i8x16_cdna3(uint4 a, uint4 b, int sum) noexcept {
    sum = hip_dot_i8x4_cdna3(a.x, b.x, sum);
    sum = hip_dot_i8x4_cdna3(a.y, b.y, sum);
    sum = hip_dot_i8x4_cdna3(a.z, b.z, sum);
    return hip_dot_i8x4_cdna3(a.w, b.w, sum);
}

/** An @c i8 fragment's sum & sum of squares, through @c v_dot4_i32_i8 against splat-one and
 *  self. */
__device__ __forceinline__ int2 hip_moments_i8x16_cdna3(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        moments.x = hip_dot_i8x4_cdna3(words[word], 0x01010101u, moments.x);
        moments.y = hip_dot_i8x4_cdna3(words[word], words[word], moments.y);
    }
    return moments;
}

/** i8 kit - CDNA3 and CDNA4, @c v_dot4_i32_i8. */
struct i8_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = true;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i8x16_cdna3(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i8x16_cdna3(a, moments);
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

#else

/** The kit this unit cannot carry - CDNA3's signed byte dot, absent before CDNA3 and on MI400 -
 *  kept so its ladder names it whatever the unit compiles. */
struct i8_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = false;
};

#endif // NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5

/** Four signed byte products added to @p sum by @c v_dot4_i32_iu8, both operands flagged signed. */
__device__ __forceinline__ int hip_dot_i8x4_cdna5(u32_t a, u32_t b, int sum) noexcept {
    return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), sum, false);
}

/** Sixteen signed @c i8 products of a fragment pair, four per @c v_dot4_i32_iu8. */
__device__ __forceinline__ int hip_dot_i8x16_cdna5(uint4 a, uint4 b, int sum) noexcept {
    sum = hip_dot_i8x4_cdna5(a.x, b.x, sum);
    sum = hip_dot_i8x4_cdna5(a.y, b.y, sum);
    sum = hip_dot_i8x4_cdna5(a.z, b.z, sum);
    return hip_dot_i8x4_cdna5(a.w, b.w, sum);
}

/** An @c i8 fragment's sum & sum of squares, through @c v_dot4_i32_iu8 against splat-one and
 *  self. */
__device__ __forceinline__ int2 hip_moments_i8x16_cdna5(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        moments.x = hip_dot_i8x4_cdna5(words[word], 0x01010101u, moments.x);
        moments.y = hip_dot_i8x4_cdna5(words[word], words[word], moments.y);
    }
    return moments;
}

/** i8 kit - CDNA5, @c v_dot4_i32_iu8. */
struct i8_cdna5_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    inline static constexpr bool compiled_k = true;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i8x16_cdna5(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i8x16_cdna5(a, moments);
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

#else

/** The kit this unit cannot carry - MI400's signedness-flagged byte dot - kept so its ladder names
 *  it whatever the unit compiles. */
struct i8_cdna5_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    inline static constexpr bool compiled_k = false;
};

#endif // NUMKONG_TARGET_CDNA5

/** u8 kit - every ROCm device, @c v_dot4_u32_u8. */
struct u8_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
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
 *  byte pairs and accumulates in f32, and the tiers' kits differ only in the conversion their loops
 *  call. Every ROCm device widens in software, one byte at a time through NumKong's converters, so
 *  CDNA3's FNUZ conversions never misread an OCP code. */

/** A packed @c e5m2 pair widened to two floats in software. */
__device__ __forceinline__ float2 hip_e5m2x2_to_f32x2_rocm(u32_t packed_pair) noexcept {
    unsigned char const codes[2] = {static_cast<unsigned char>(packed_pair),
                                    static_cast<unsigned char>(packed_pair >> 8)};
    float2 values;
    nk_e5m2_to_f32_simt_(&codes[0], &values.x);
    nk_e5m2_to_f32_simt_(&codes[1], &values.y);
    return values;
}

/** A packed @c e4m3 pair widened to two floats in software. */
__device__ __forceinline__ float2 hip_e4m3x2_to_f32x2_rocm(u32_t packed_pair) noexcept {
    unsigned char const codes[2] = {static_cast<unsigned char>(packed_pair),
                                    static_cast<unsigned char>(packed_pair >> 8)};
    float2 values;
    nk_e4m3_to_f32_simt_(&codes[0], &values.x);
    nk_e4m3_to_f32_simt_(&codes[1], &values.y);
    return values;
}

/** A packed @c e3m2 pair widened to two floats in software. */
__device__ __forceinline__ float2 hip_e3m2x2_to_f32x2_rocm(u32_t packed_pair) noexcept {
    unsigned char const codes[2] = {static_cast<unsigned char>(packed_pair),
                                    static_cast<unsigned char>(packed_pair >> 8)};
    float2 values;
    nk_e3m2_to_f32_simt_(&codes[0], &values.x);
    nk_e3m2_to_f32_simt_(&codes[1], &values.y);
    return values;
}

/** A packed @c e2m3 pair widened to two floats in software. */
__device__ __forceinline__ float2 hip_e2m3x2_to_f32x2_rocm(u32_t packed_pair) noexcept {
    unsigned char const codes[2] = {static_cast<unsigned char>(packed_pair),
                                    static_cast<unsigned char>(packed_pair >> 8)};
    float2 values;
    nk_e2m3_to_f32_simt_(&codes[0], &values.x);
    nk_e2m3_to_f32_simt_(&codes[1], &values.y);
    return values;
}

/** Sixteen @c e5m2 products of a fragment pair, each byte pair widened in software. */
__device__ __forceinline__ f32_t hip_dot_e5m2x16_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = hip_e5m2x2_to_f32x2_rocm(words_a[word] >> (16 * pair));
            float2 const values_b = hip_e5m2x2_to_f32x2_rocm(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** An @c e5m2 fragment's sum & sum of squares, widened in software. */
__device__ __forceinline__ float2 hip_moments_e5m2x16_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e5m2x2_to_f32x2_rocm(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c e5m2 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_e5m2x16_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e5m2x2_to_f32x2_rocm(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e4m3 twin of @ref hip_dot_e5m2x16_rocm. */
__device__ __forceinline__ f32_t hip_dot_e4m3x16_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = hip_e4m3x2_to_f32x2_rocm(words_a[word] >> (16 * pair));
            float2 const values_b = hip_e4m3x2_to_f32x2_rocm(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** The @c e4m3 twin of @ref hip_moments_e5m2x16_rocm. */
__device__ __forceinline__ float2 hip_moments_e4m3x16_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e4m3x2_to_f32x2_rocm(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e4m3 twin of @ref hip_weighted_e5m2x16_rocm. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_e4m3x16_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e4m3x2_to_f32x2_rocm(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e3m2 twin of @ref hip_dot_e5m2x16_rocm. */
__device__ __forceinline__ f32_t hip_dot_e3m2x16_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = hip_e3m2x2_to_f32x2_rocm(words_a[word] >> (16 * pair));
            float2 const values_b = hip_e3m2x2_to_f32x2_rocm(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** The @c e3m2 twin of @ref hip_moments_e5m2x16_rocm. */
__device__ __forceinline__ float2 hip_moments_e3m2x16_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e3m2x2_to_f32x2_rocm(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e3m2 twin of @ref hip_weighted_e5m2x16_rocm. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_e3m2x16_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e3m2x2_to_f32x2_rocm(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e2m3 twin of @ref hip_dot_e5m2x16_rocm. */
__device__ __forceinline__ f32_t hip_dot_e2m3x16_rocm(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = hip_e2m3x2_to_f32x2_rocm(words_a[word] >> (16 * pair));
            float2 const values_b = hip_e2m3x2_to_f32x2_rocm(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** The @c e2m3 twin of @ref hip_moments_e5m2x16_rocm. */
__device__ __forceinline__ float2 hip_moments_e2m3x16_rocm(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e2m3x2_to_f32x2_rocm(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e2m3 twin of @ref hip_weighted_e5m2x16_rocm. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_e2m3x16_rocm( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e2m3x2_to_f32x2_rocm(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** e5m2 kit - every ROCm device, widening in software. */
struct e5m2_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_e5m2x16_rocm(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_e5m2x16_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_e5m2x16_rocm(a, weight, sums);
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

/** e4m3 kit - every ROCm device, widening in software. */
struct e4m3_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_e4m3x16_rocm(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_e4m3x16_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_e4m3x16_rocm(a, weight, sums);
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

/** e3m2 kit - every ROCm device, widening in software. */
struct e3m2_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_e3m2x16_rocm(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_e3m2x16_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_e3m2x16_rocm(a, weight, sums);
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

/** e2m3 kit - every ROCm device, widening in software. */
struct e2m3_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_e2m3x16_rocm(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_e2m3x16_rocm(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_e2m3x16_rocm(a, weight, sums);
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

#if NUMKONG_ARCH_ROCM_CDNA4_

/** A packed OCP @c e4m3 pair widened to two floats in one @c v_cvt_pk_f32_fp8, on CDNA4 and every
 *  tier since. */
__device__ __forceinline__ float2 hip_e4m3x2_to_f32x2_cdna4(u32_t packed_pair) noexcept {
    auto const values = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(packed_pair), false);
    return {values[0], values[1]};
}

/** The @c e5m2 twin of @ref hip_e4m3x2_to_f32x2_cdna4, through @c v_cvt_pk_f32_bf8. */
__device__ __forceinline__ float2 hip_e5m2x2_to_f32x2_cdna4(u32_t packed_pair) noexcept {
    auto const values = __builtin_amdgcn_cvt_pk_f32_bf8(static_cast<int>(packed_pair), false);
    return {values[0], values[1]};
}

/** Sixteen @c e5m2 products of a fragment pair, each byte pair widened through the conversion. */
__device__ __forceinline__ f32_t hip_dot_e5m2x16_cdna4(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = hip_e5m2x2_to_f32x2_cdna4(words_a[word] >> (16 * pair));
            float2 const values_b = hip_e5m2x2_to_f32x2_cdna4(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** An @c e5m2 fragment's sum & sum of squares, widened through the conversion. */
__device__ __forceinline__ float2 hip_moments_e5m2x16_cdna4(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e5m2x2_to_f32x2_cdna4(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c e5m2 fragment's sixteen values, widened
 *  through the conversion. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_e5m2x16_cdna4( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e5m2x2_to_f32x2_cdna4(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e4m3 twin of @ref hip_dot_e5m2x16_cdna4. */
__device__ __forceinline__ f32_t hip_dot_e4m3x16_cdna4(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = hip_e4m3x2_to_f32x2_cdna4(words_a[word] >> (16 * pair));
            float2 const values_b = hip_e4m3x2_to_f32x2_cdna4(words_b[word] >> (16 * pair));
            sum = fmaf(values_a.x, values_b.x, sum);
            sum = fmaf(values_a.y, values_b.y, sum);
        }
    return sum;
}

/** The @c e4m3 twin of @ref hip_moments_e5m2x16_cdna4. */
__device__ __forceinline__ float2 hip_moments_e4m3x16_cdna4(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e4m3x2_to_f32x2_cdna4(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e4m3 twin of @ref hip_weighted_e5m2x16_cdna4. */
__device__ __forceinline__ std::array<f32_t, 16> hip_weighted_e4m3x16_cdna4( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = hip_e4m3x2_to_f32x2_cdna4(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** e5m2 kit - CDNA4 and CDNA5, widening through the conversion. */
struct e5m2_cdna4_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    inline static constexpr bool compiled_k = true;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_e5m2x16_cdna4(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_e5m2x16_cdna4(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_e5m2x16_cdna4(a, weight, sums);
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

/** e4m3 kit - CDNA4 and CDNA5, widening through the conversion. */
struct e4m3_cdna4_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    inline static constexpr bool compiled_k = true;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_e4m3x16_cdna4(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return hip_moments_e4m3x16_cdna4(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return hip_weighted_e4m3x16_cdna4(a, weight, sums);
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

#else

/** The kit this unit cannot carry - the OCP fp8 conversions, from CDNA4 on - kept so its ladder
 *  names it whatever the unit compiles. */
struct e5m2_cdna4_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    inline static constexpr bool compiled_k = false;
};

/** The @c e4m3 twin of the @c e5m2_cdna4_quads_kit_t stub. */
struct e4m3_cdna4_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    inline static constexpr bool compiled_k = false;
};

#endif // NUMKONG_ARCH_ROCM_CDNA4_

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

/** Thirty-two signed @c i4 products of a fragment pair, two @ref hip_dot_i8x4_rocm per source
 *  word. */
__device__ __forceinline__ int hip_dot_i4x32_rocm(uint4 a, uint4 b, int sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = hip_unpack_i4x8_rocm(words_a[word]);
        uint2 const unpacked_b = hip_unpack_i4x8_rocm(words_b[word]);
        sum = hip_dot_i8x4_rocm(unpacked_a.x, unpacked_b.x, sum);
        sum = hip_dot_i8x4_rocm(unpacked_a.y, unpacked_b.y, sum);
    }
    return sum;
}

/** An @c i4 fragment's sum & sum of squares, through @ref hip_dot_i8x4_rocm. */
__device__ __forceinline__ int2 hip_moments_i4x32_rocm(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_i4x8_rocm(words[word]);
        moments.x = hip_dot_i8x4_rocm(unpacked.x, 0x01010101u, moments.x);
        moments.x = hip_dot_i8x4_rocm(unpacked.y, 0x01010101u, moments.x);
        moments.y = hip_dot_i8x4_rocm(unpacked.x, unpacked.x, moments.y);
        moments.y = hip_dot_i8x4_rocm(unpacked.y, unpacked.y, moments.y);
    }
    return moments;
}

/** i4x2 kit - every ROCm device, NumKong's portable byte dot over unpacked nibbles. */
struct i4x2_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i4x32_rocm(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i4x32_rocm(a, moments);
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

/** Thirty-two signed @c i4 products of a fragment pair, two @c v_dot4_i32_i8 per source word. */
__device__ __forceinline__ int hip_dot_i4x32_cdna3(uint4 a, uint4 b, int sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = hip_unpack_i4x8_rocm(words_a[word]);
        uint2 const unpacked_b = hip_unpack_i4x8_rocm(words_b[word]);
        sum = hip_dot_i8x4_cdna3(unpacked_a.x, unpacked_b.x, sum);
        sum = hip_dot_i8x4_cdna3(unpacked_a.y, unpacked_b.y, sum);
    }
    return sum;
}

/** An @c i4 fragment's sum & sum of squares, through @c v_dot4_i32_i8. */
__device__ __forceinline__ int2 hip_moments_i4x32_cdna3(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_i4x8_rocm(words[word]);
        moments.x = hip_dot_i8x4_cdna3(unpacked.x, 0x01010101u, moments.x);
        moments.x = hip_dot_i8x4_cdna3(unpacked.y, 0x01010101u, moments.x);
        moments.y = hip_dot_i8x4_cdna3(unpacked.x, unpacked.x, moments.y);
        moments.y = hip_dot_i8x4_cdna3(unpacked.y, unpacked.y, moments.y);
    }
    return moments;
}

/** i4x2 kit - CDNA3 and CDNA4, the byte engine over unpacked nibbles. */
struct i4x2_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = true;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i4x32_cdna3(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i4x32_cdna3(a, moments);
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

#else

/** The kit this unit cannot carry - the nibbles over CDNA3's signed byte dot - kept so its ladder
 *  names it whatever the unit compiles. */
struct i4x2_cdna3_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = false;
};

#endif // NUMKONG_TARGET_CDNA3 || NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5

/** Thirty-two signed @c i4 products of a fragment pair, two @c v_dot4_i32_iu8 per source word. */
__device__ __forceinline__ int hip_dot_i4x32_cdna5(uint4 a, uint4 b, int sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = hip_unpack_i4x8_rocm(words_a[word]);
        uint2 const unpacked_b = hip_unpack_i4x8_rocm(words_b[word]);
        sum = hip_dot_i8x4_cdna5(unpacked_a.x, unpacked_b.x, sum);
        sum = hip_dot_i8x4_cdna5(unpacked_a.y, unpacked_b.y, sum);
    }
    return sum;
}

/** An @c i4 fragment's sum & sum of squares, through @c v_dot4_i32_iu8. */
__device__ __forceinline__ int2 hip_moments_i4x32_cdna5(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = hip_unpack_i4x8_rocm(words[word]);
        moments.x = hip_dot_i8x4_cdna5(unpacked.x, 0x01010101u, moments.x);
        moments.x = hip_dot_i8x4_cdna5(unpacked.y, 0x01010101u, moments.x);
        moments.y = hip_dot_i8x4_cdna5(unpacked.x, unpacked.x, moments.y);
        moments.y = hip_dot_i8x4_cdna5(unpacked.y, unpacked.y, moments.y);
    }
    return moments;
}

/** i4x2 kit - CDNA5, the byte engine over unpacked nibbles. */
struct i4x2_cdna5_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    inline static constexpr bool compiled_k = true;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return hip_dot_i4x32_cdna5(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return hip_moments_i4x32_cdna5(a, moments);
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

#else

/** The kit this unit cannot carry - the nibbles over MI400's byte dot - kept so its ladder names it
 *  whatever the unit compiles. */
struct i4x2_cdna5_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    inline static constexpr bool compiled_k = false;
};

#endif // NUMKONG_TARGET_CDNA5

/** u4x2 kit - every ROCm device, the byte engine over unpacked nibbles. */
struct u4x2_rocm_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
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

/** Copies the fragments a lane strides over, @p lanes apart, from the rows an ID span names into
 *  the cache's leading rows through the lane's registers. */
template <typename storage_, typename cache_>
__device__ __forceinline__ void hip_copy_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                              u32_t lane, u32_t lanes) noexcept {
    for (u32_t word = lane; word < rows * storage.geometry.row_fragments; word += lanes) {
        u32_t const row = storage.geometry.row_of_word(word), fragment = storage.geometry.fragment_of_word(word, row);
        cache.at(row, fragment) = storage.vector_for(ids[row], fragment);
    }
}

/** The staging every ROCm device runs: each lane copies the fragments it strides over through its
 *  registers, and one wavefront barrier publishes them before the staging returns. */
struct rocm_registers_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_rocm_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_;
    using warp_t = hip_warp<hip_wave_lanes(capability_k)>;
    inline static constexpr u32_t stages_k = 1;

    /** Nothing shared: the copies are plain stores. */
    struct shared_t {};

    __device__ __forceinline__ rocm_registers_schedule_t(shared_t&, u32_t) noexcept {}

    /** Scores a walk's batch through the kit's scalar dots, one fragment a load. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return vram_score_batch<kit_, warp_t, resident_blocks_>( //
            storage, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
    }

    /** The greedy diversity prefix as @ref vram_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return vram_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    static __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids,
                                                      u32_t rows, u32_t lane) noexcept {
        if (cache.ids_) return;
        hip_copy_rows(storage, cache, ids, rows, lane, warp_t::width_k);
        warp_t::sync();
    }

    __device__ __forceinline__ void await_rows(u32_t) noexcept {}
};

/** The MI400 staging: the register copies over 32-lane wavefronts, until its asynchronous LDS loads
 *  are wired. */
struct cdna5_registers_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna5_k;
    inline static constexpr bool compiled_k = NUMKONG_TARGET_CDNA5;
    using warp_t = hip_warp<hip_wave_lanes(capability_k)>;
    inline static constexpr u32_t stages_k = 1;

    /** Nothing shared: the copies are plain stores. */
    struct shared_t {};

    __device__ __forceinline__ cdna5_registers_schedule_t(shared_t&, u32_t) noexcept {}

    /** Scores a walk's batch through the kit's scalar dots, one fragment a load. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return vram_score_batch<kit_, warp_t, resident_blocks_>( //
            storage, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
    }

    /** The greedy diversity prefix as @ref vram_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return vram_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    static __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids,
                                                      u32_t rows, u32_t lane) noexcept {
        if (cache.ids_) return;
        hip_copy_rows(storage, cache, ids, rows, lane, warp_t::width_k);
        warp_t::sync();
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

/*  The CDNA3 and CDNA4 staging: LDS DMA moves a unit per lane from global memory straight into the
 *  shared cache, past the registers - a 32-bit word on CDNA3, a whole fragment on CDNA4. One
 *  instruction stores a wavefront's units side by side above one wavefront-uniform base, and the
 *  cache rows lie back to back, so a unit's index in the staged rows is its offset in the cache.
 *  The copies count against @c vmcnt like loads, which one wait drains. A size of dependent type
 *  defers the builtin's constant check to instantiation, where it is one. */

/** The CDNA3 staging: LDS DMA of one 32-bit word per lane. */
struct cdna3_lds_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna3_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_CDNA3_;
    using warp_t = hip_warp<hip_wave_lanes(capability_k)>;
    inline static constexpr u32_t stages_k = 1;

    /** Nothing shared: the copies count on the wavefront's own @c vmcnt. */
    struct shared_t {};

    __device__ __forceinline__ cdna3_lds_schedule_t(shared_t&, u32_t) noexcept {}

    /** Scores a walk's batch as @ref rocm_registers_schedule_t does. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return vram_score_batch<kit_, warp_t, resident_blocks_>( //
            storage, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
    }

    /** The greedy diversity prefix as @ref vram_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return vram_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        if (cache.ids_) return;
        constexpr u32_t lane_bytes_k = sizeof(u32_t);
        constexpr u32_t units_per_fragment_k = sizeof(typename storage_::fragment_t) / lane_bytes_k;
        u32_t const row_units = storage.geometry.row_fragments * units_per_fragment_k, units = rows * row_units;
        byte_t* const cache_bytes = reinterpret_cast<byte_t*>(cache.row(0));
        for (u32_t base = 0; base < units; base += warp_t::width_k) {
            u32_t const unit = base + lane;
            if (unit >= units) break;
            u32_t const row = storage.geometry.row_of_word(unit / units_per_fragment_k);
            byte_t* const source = reinterpret_cast<byte_t*>(storage.vector_span_for(ids[row])) +
                                   (unit - row * row_units) * lane_bytes_k;
            __builtin_amdgcn_global_load_lds(hip_global_pointer(source),
                                             hip_shared_pointer(cache_bytes + base * lane_bytes_k),
                                             static_cast<u32_t>(lane_bytes_k), 0, 0);
        }
    }

    __device__ __forceinline__ void await_rows(u32_t) noexcept { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
};

/** The CDNA4 staging: LDS DMA of a whole fragment per lane. */
struct cdna4_lds_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_cdna4_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_ROCM_CDNA4_;
    using warp_t = hip_warp<hip_wave_lanes(capability_k)>;
    inline static constexpr u32_t stages_k = 1;

    /** Nothing shared: the copies count on the wavefront's own @c vmcnt. */
    struct shared_t {};

    __device__ __forceinline__ cdna4_lds_schedule_t(shared_t&, u32_t) noexcept {}

    /** Scores a walk's batch as @ref rocm_registers_schedule_t does. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return vram_score_batch<kit_, warp_t, resident_blocks_>( //
            storage, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
    }

    /** The greedy diversity prefix as @ref vram_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return vram_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        if (cache.ids_) return;
        constexpr u32_t lane_bytes_k = sizeof(uint4);
        constexpr u32_t units_per_fragment_k = sizeof(typename storage_::fragment_t) / lane_bytes_k;
        u32_t const row_units = storage.geometry.row_fragments * units_per_fragment_k, units = rows * row_units;
        byte_t* const cache_bytes = reinterpret_cast<byte_t*>(cache.row(0));
        for (u32_t base = 0; base < units; base += warp_t::width_k) {
            u32_t const unit = base + lane;
            if (unit >= units) break;
            u32_t const row = storage.geometry.row_of_word(unit / units_per_fragment_k);
            byte_t* const source = reinterpret_cast<byte_t*>(storage.vector_span_for(ids[row])) +
                                   (unit - row * row_units) * lane_bytes_k;
            __builtin_amdgcn_global_load_lds(hip_global_pointer(source),
                                             hip_shared_pointer(cache_bytes + base * lane_bytes_k),
                                             static_cast<u32_t>(lane_bytes_k), 0, 0);
        }
    }

    __device__ __forceinline__ void await_rows(u32_t) noexcept { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
};

#pragma endregion Schedules

#pragma region Ladders

/*  A class's ceiling carries every older tier's bit, so each ladder lists the newest tier first.
 *  MI400 keeps the CDNA4 conversions but drops CDNA3's signed byte and half dots, and its unit
 *  compiles neither those nor CDNA4's brain-float dot, so its ladders fall past them. */

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::f16_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        f16_rocm_quads_kit_t,           //
        f16_cdna3_quads_kit_t           //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::bf16_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        bf16_rocm_quads_kit_t,          //
        bf16_cdna4_quads_kit_t          //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::i8_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        i8_rocm_quads_kit_t,            //
        i8_cdna5_quads_kit_t,           //
        i8_cdna3_quads_kit_t            //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::u8_t, caps_> {
    using kit_t = typename highest_kit<caps_, u8_rocm_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::e5m2_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        e5m2_rocm_quads_kit_t,          //
        e5m2_cdna4_quads_kit_t          //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::e4m3_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        e4m3_rocm_quads_kit_t,          //
        e4m3_cdna4_quads_kit_t          //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::e3m2_t, caps_> {
    using kit_t = typename highest_kit<caps_, e3m2_rocm_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::e2m3_t, caps_> {
    using kit_t = typename highest_kit<caps_, e2m3_rocm_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::i4x2_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        i4x2_rocm_quads_kit_t,          //
        i4x2_cdna5_quads_kit_t,         //
        i4x2_cdna3_quads_kit_t          //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::u4x2_t, caps_> {
    using kit_t = typename highest_kit<caps_, u4x2_rocm_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct quads_ladder<nk::u1x8_t, caps_> {
    using kit_t = typename highest_kit<caps_, u1x8_rocm_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

/** The staging schedule under @p caps_: the register copies on MI400 and every ROCm device below
 *  CDNA3, and the LDS DMA copies on CDNA3 and CDNA4, a fragment per lane where CDNA4 allows it. */
template <nk_capability_t caps_>
    requires((caps_ & nk_cap_rocm_k) != 0)
struct schedule_ladder<caps_> {
    using schedule_t = typename highest_kit< //
        caps_,                               //
        rocm_registers_schedule_t,           //
        cdna5_registers_schedule_t,          //
        cdna4_lds_schedule_t,                //
        cdna3_lds_schedule_t                 //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & schedule_t::capability_k;
};

#pragma endregion Ladders

} // namespace unum::usearch::dense
