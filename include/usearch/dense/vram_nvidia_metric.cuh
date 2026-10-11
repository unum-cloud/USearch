/**
 *  @file include/usearch/dense/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Device metric kits for the dense VRAM indexes - the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one storage dtype to one GPU generation, named like the CPU kits -
 *  @b dtype_generation_quads_kit_t. The generation slot names the oldest tier whose instructions
 *  the kit needs: @c ampere for the sm_80 floor, @c ada for the fp8 conversions, @c blackwell for
 *  the mixed-precision multiply-adds and the fp6 conversions. Each tier's kits sit behind its
 *  @c NUMKONG_TARGET_* gate and a chain picks among them under the caller's ceiling, exactly like
 *  the CPU chains, so a unit compiled for one architecture compiles the kits that architecture
 *  runs. The arithmetic lives in named device primitives; a kit binds them.
 *
 *  Fragments are 16 bytes of scalars held in registers, and metrics with precomputable norms
 *  finalize "through the dot". Row geometry - how many fragments span one vector, how many lanes
 *  cooperate on a row - and the metric are runtime state of the planes view, not the kit:
 *  dimensions are frozen by the first @c reserve, never by a type, and the metric only steers the
 *  scalar epilogue, so one kernel serves every metric of a dtype. The cross-lane reduction is
 *  deliberately not a kit member - one shuffle tree in @c vram_index.cuh serves every kit, over the
 *  lanes policy each kit names. Distances take both sides' norms, as the CPU finalizers do, so a
 *  symmetric candidate-to-candidate distance is the same call.
 *
 *  Families sort from the widest numeric types to the smallest & most exotic: 16-bit floats, 8-bit
 *  integers, minifloats, packed nibbles, packed bits. Norms are the CPU's @c vector_norms over each
 *  kit's dot type, so the unsigned families accumulate in unsigned words with defined wrapping and
 *  no cast chatter, while @c i8 keeps the CPU planes' signed @b {sum,sumsq} layout.
 */
#pragma once

#include <array> // `std::array` - a fragment's weighted sums

#include <cuda/atomic>     // `cuda::atomic_ref` - the device twin of `std::atomic_ref`
#include <cuda_bf16.h>     // `__nv_bfloat162` packed math, sm_80+
#include <cuda_fp16.h>     // `__half2` packed math
#include <cuda_pipeline.h> // `__pipeline_memcpy_async` - register-free global-to-shared copies

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the epilogues switch on it
#include <numkong/types.hpp>      // `nk::bf16_t` & the other scalars the kits load

#include "usearch/utilities/types.hpp"
#include "usearch/dense/types.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief The NVIDIA lanes policy the shared walk code is written against: 32-lane warps with
 *      32-bit lane masks, device-scope atomics, and the @c cp.async staging copies.
 *
 *  Every collective takes the full-warp mask, since the walks keep their warps converged.
 */
struct cuda_lanes_t {
    using mask_t = u32_t;
    inline static constexpr u32_t width_k = 32;
    inline static constexpr mask_t all_k = 0xFFFFFFFFu;

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle(value_type_ value, int source_lane) noexcept {
        return __shfl_sync(all_k, value, source_lane);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_xor(value_type_ value, int lane_mask) noexcept {
        return __shfl_xor_sync(all_k, value, lane_mask);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_up(value_type_ value, u32_t delta) noexcept {
        return __shfl_up_sync(all_k, value, delta);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_down(value_type_ value, u32_t delta,
                                                               int width = width_k) noexcept {
        return __shfl_down_sync(all_k, value, delta, width);
    }
    // The predicates are `int`, as the warp intrinsics take them, so any nonzero lane votes.
    static __device__ __forceinline__ mask_t ballot(int predicate) noexcept { return __ballot_sync(all_k, predicate); }
    static __device__ __forceinline__ bool any(int predicate) noexcept { return __any_sync(all_k, predicate); }
    static __device__ __forceinline__ bool all(int predicate) noexcept { return __all_sync(all_k, predicate); }
    static __device__ __forceinline__ void sync() noexcept { __syncwarp(); }

    /** The lanes holding the same @p value as this one, this lane included. */
    static __device__ __forceinline__ mask_t match_any(u32_t value) noexcept { return __match_any_sync(all_k, value); }

    static __device__ __forceinline__ u32_t count(mask_t mask) noexcept { return __popc(mask); }

    /** The lowest lane of a non-empty @p mask. */
    static __device__ __forceinline__ int first(mask_t mask) noexcept { return __ffs(mask) - 1; }

    /** The lanes strictly below @p lane. */
    static __device__ __forceinline__ mask_t below(int lane) noexcept { return (1u << lane) - 1; }

    static __device__ __forceinline__ void sleep(u32_t nanoseconds) noexcept { __nanosleep(nanoseconds); }

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_relaxed(value_type_& word) noexcept {
        return cuda::atomic_ref<value_type_, cuda::thread_scope_device>(word).load(cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_acquire(value_type_& word) noexcept {
        return cuda::atomic_ref<value_type_, cuda::thread_scope_device>(word).load(cuda::memory_order_acquire);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_relaxed(value_type_& word, value_type_ value) noexcept {
        cuda::atomic_ref<value_type_, cuda::thread_scope_device>(word).store(value, cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_release(value_type_& word, value_type_ value) noexcept {
        cuda::atomic_ref<value_type_, cuda::thread_scope_device>(word).store(value, cuda::memory_order_release);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ exchange_acquire(value_type_& word, value_type_ value) noexcept {
        return cuda::atomic_ref<value_type_, cuda::thread_scope_device>(word).exchange(value,
                                                                                       cuda::memory_order_acquire);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_add_relaxed(value_type_& word, value_type_ value) noexcept {
        return cuda::atomic_ref<value_type_, cuda::thread_scope_device>(word).fetch_add(value,
                                                                                        cuda::memory_order_relaxed);
    }

    /** Orders every store before it ahead of any store after it, for readers anywhere on the
     *  device. */
    static __device__ __forceinline__ void fence_release() noexcept {
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
    }

    /** Queues one register-free copy of @p source into @p destination; @ref copy_wait lands
     *  them. */
    template <typename value_type_>
    static __device__ __forceinline__ void copy_async(value_type_* destination, value_type_ const* source) noexcept {
        __pipeline_memcpy_async(destination, source, sizeof(value_type_));
    }
    static __device__ __forceinline__ void copy_wait() noexcept {
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
};

/** The device-scope atomic reference the NVIDIA-only kernels reach a shared word through, so each
 *  access names its memory order - relaxed wherever the launch boundary is what publishes the
 *  result. */
template <typename value_type_>
using cuda_atomic_ref = cuda::atomic_ref<value_type_, cuda::thread_scope_device>;

#pragma region Metrics

/** Angular distance from a dot and both sides' squared norms, finalized exactly like the CPU's
 *  @c nk_angular_through_f32_from_dot: clamped at zero, and a zero-norm side scores zero against a
 *  vanishing dot and one otherwise.
 *  The query's root is loop-invariant, so the compiler hoists it. */
__device__ __forceinline__ dense_distance_t dense_cuda_angular_through_dot(f32_t dot, f32_t query_sumsq,
                                                                           f32_t target_sumsq) noexcept {
    if (query_sumsq > 0.0f && target_sumsq > 0.0f)
        return fmaxf(1.0f - dot * (rsqrtf(query_sumsq) * rsqrtf(target_sumsq)), 0.0f);
    return dot == 0.0f ? 0.0f : 1.0f;
}

/** A numeric kit's distance for @p kernel "through the dot": @b 1-dot for the inner product,
 *  monotone decreasing and negative-safe, the clamped angular, and the squared Euclidean the walk
 *  ranks by. The index validates @p kernel at compile time, so the fallback only ever sees the
 *  Euclidean pair. */
template <typename dot_type_>
__device__ __forceinline__ dense_distance_t dense_cuda_distance(nk_kernel_kind_t kernel, dot_type_ dot,
                                                                vector_norms<dot_type_> query,
                                                                vector_norms<dot_type_> target) noexcept {
    f32_t const dot_value = static_cast<f32_t>(dot);
    switch (kernel) {
    case nk_kernel_dot_k: return 1.0f - dot_value;
    case nk_kernel_angular_k:
        return dense_cuda_angular_through_dot(dot_value, static_cast<f32_t>(query.sumsq),
                                              static_cast<f32_t>(target.sumsq));
    default: return static_cast<f32_t>(query.sumsq) + static_cast<f32_t>(target.sumsq) - 2.0f * dot_value;
    }
}

/** Output-boundary finalize, the CPU kits' @c convert_to_distances in miniature: the traversal
 *  ranks on raw @c distance values end to end, and only the matches a caller receives pass through
 *  here - identity unless a metric owes a monotone transform, like Euclidean's square root. */
__device__ __forceinline__ dense_distance_t dense_cuda_convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
    return kernel == nk_kernel_euclidean_k ? sqrtf(fmaxf(ranked, 0.0f)) : ranked;
}

#pragma endregion Metrics

/** The kits every chain names, declared whether or not their tier is compiled in. */
struct f16_ampere_quads_kit_t;
struct f16_blackwell_quads_kit_t;
struct bf16_ampere_quads_kit_t;
struct bf16_blackwell_quads_kit_t;
struct i8_ampere_quads_kit_t;
struct u8_ampere_quads_kit_t;
struct e5m2_ampere_quads_kit_t;
struct e5m2_ada_quads_kit_t;
struct e5m2_blackwell_quads_kit_t;
struct e4m3_ampere_quads_kit_t;
struct e4m3_ada_quads_kit_t;
struct e4m3_blackwell_quads_kit_t;
struct e3m2_ampere_quads_kit_t;
struct e3m2_blackwell_quads_kit_t;
struct e2m3_ampere_quads_kit_t;
struct e2m3_blackwell_quads_kit_t;
struct i4x2_ampere_quads_kit_t;
struct u4x2_ampere_quads_kit_t;
struct u1x8_ampere_quads_kit_t;

#pragma region Half Precision Floats

/** Eight @c f16 products of a fragment pair, each rounded to half by @c hmul2, summed in f32 per
 *  fragment so no half-range overflow can accumulate. */
__device__ __forceinline__ f32_t dense_cuda_dot_f16x8_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const products = __half22float2(__hmul2(*reinterpret_cast<__half2 const*>(&words_a[word]),
                                                       *reinterpret_cast<__half2 const*>(&words_b[word])));
        sum += products.x + products.y;
    }
    return sum;
}

/** An @c f16 fragment's sum & sum of squares, widened to f32. */
__device__ __forceinline__ float2 dense_cuda_moments_f16x8_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = __half22float2(*reinterpret_cast<__half2 const*>(&words[word]));
        moments.x += values.x + values.y;
        moments.y += values.x * values.x + values.y * values.y;
    }
    return moments;
}

/** The @c bf16 twin of @ref dense_cuda_dot_f16x8_ampere, through the @c bf16x2 instructions that
 *  land exactly at the sm_80 floor. */
__device__ __forceinline__ f32_t dense_cuda_dot_bf16x8_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const products = __bfloat1622float2(__hmul2(*reinterpret_cast<__nv_bfloat162 const*>(&words_a[word]),
                                                           *reinterpret_cast<__nv_bfloat162 const*>(&words_b[word])));
        sum += products.x + products.y;
    }
    return sum;
}

/** A @c bf16 fragment's sum & sum of squares, widened to f32. */
__device__ __forceinline__ float2 dense_cuda_moments_bf16x8_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&words[word]));
        moments.x += values.x + values.y;
        moments.y += values.x * values.x + values.y * values.y;
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c f16 fragment's eight values, widened to
 *  f32. */
__device__ __forceinline__ std::array<f32_t, 8> dense_cuda_weighted_f16x8_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 8> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = __half22float2(*reinterpret_cast<__half2 const*>(&words[word]));
        sums[2 * word] += weight * values.x;
        sums[2 * word + 1] += weight * values.y;
    }
    return sums;
}

/** The @c bf16 twin of @ref dense_cuda_weighted_f16x8_ampere. */
__device__ __forceinline__ std::array<f32_t, 8> dense_cuda_weighted_bf16x8_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 8> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&words[word]));
        sums[2 * word] += weight * values.x;
        sums[2 * word + 1] += weight * values.y;
    }
    return sums;
}

/** f16 kit - the sm_80 floor. */
struct f16_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_f16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_f16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_f16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** bf16 kit - the sm_80 floor. */
struct bf16_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_bf16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_bf16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_bf16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#if NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

/** Both halves of two @c f16x2 words multiplied straight into an f32 sum by @c fma.rn.f32.f16 -
 *  exact products, so none rounds to half or overflows its range. */
__device__ __forceinline__ f32_t dense_cuda_fma_f16x2_blackwell(u32_t first_pair, u32_t second_pair,
                                                                f32_t sum) noexcept {
    asm("{.reg .b16 first_low, first_high, second_low, second_high;\n\t" //
        "mov.b32 {first_low, first_high}, %1;\n\t"                       //
        "mov.b32 {second_low, second_high}, %2;\n\t"                     //
        "fma.rn.f32.f16 %0, first_low, second_low, %0;\n\t"              //
        "fma.rn.f32.f16 %0, first_high, second_high, %0;}"
        : "+f"(sum)
        : "r"(first_pair), "r"(second_pair));
    return sum;
}

/** The @c bf16x2 twin of @ref dense_cuda_fma_f16x2_blackwell, through @c fma.rn.f32.bf16. */
__device__ __forceinline__ f32_t dense_cuda_fma_bf16x2_blackwell(u32_t first_pair, u32_t second_pair,
                                                                 f32_t sum) noexcept {
    asm("{.reg .b16 first_low, first_high, second_low, second_high;\n\t" //
        "mov.b32 {first_low, first_high}, %1;\n\t"                       //
        "mov.b32 {second_low, second_high}, %2;\n\t"                     //
        "fma.rn.f32.bf16 %0, first_low, second_low, %0;\n\t"             //
        "fma.rn.f32.bf16 %0, first_high, second_high, %0;}"
        : "+f"(sum)
        : "r"(first_pair), "r"(second_pair));
    return sum;
}

/** Eight @c f16 products of a fragment pair, each exact, fused straight into the f32 sum. */
__device__ __forceinline__ f32_t dense_cuda_dot_f16x8_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    sum = dense_cuda_fma_f16x2_blackwell(a.x, b.x, sum);
    sum = dense_cuda_fma_f16x2_blackwell(a.y, b.y, sum);
    sum = dense_cuda_fma_f16x2_blackwell(a.z, b.z, sum);
    return dense_cuda_fma_f16x2_blackwell(a.w, b.w, sum);
}

/** Eight @c bf16 products of a fragment pair, each exact, fused straight into the f32 sum. */
__device__ __forceinline__ f32_t dense_cuda_dot_bf16x8_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    sum = dense_cuda_fma_bf16x2_blackwell(a.x, b.x, sum);
    sum = dense_cuda_fma_bf16x2_blackwell(a.y, b.y, sum);
    sum = dense_cuda_fma_bf16x2_blackwell(a.z, b.z, sum);
    return dense_cuda_fma_bf16x2_blackwell(a.w, b.w, sum);
}

/** f16 kit - Blackwell, @c fma.rn.f32.f16. */
struct f16_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_f16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_f16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_f16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** bf16 kit - Blackwell, @c fma.rn.f32.bf16. */
struct bf16_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_bf16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_bf16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_bf16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

#pragma endregion Half Precision Floats

#pragma region Byte Sized Integers

/** Sixteen signed @c i8 products of a fragment pair, one @c dp4a per word. */
__device__ __forceinline__ int dense_cuda_dot_i8x16_ampere(uint4 a, uint4 b, int sum) noexcept {
    sum = __dp4a(static_cast<int>(a.x), static_cast<int>(b.x), sum);
    sum = __dp4a(static_cast<int>(a.y), static_cast<int>(b.y), sum);
    sum = __dp4a(static_cast<int>(a.z), static_cast<int>(b.z), sum);
    return __dp4a(static_cast<int>(a.w), static_cast<int>(b.w), sum);
}

/** An @c i8 fragment's sum & sum of squares, through the same instruction against splat-one and
 *  self. */
__device__ __forceinline__ int2 dense_cuda_moments_i8x16_ampere(uint4 a, int2 moments) noexcept {
    moments.x = __dp4a(static_cast<int>(a.x), 0x01010101, moments.x);
    moments.x = __dp4a(static_cast<int>(a.y), 0x01010101, moments.x);
    moments.x = __dp4a(static_cast<int>(a.z), 0x01010101, moments.x);
    moments.x = __dp4a(static_cast<int>(a.w), 0x01010101, moments.x);
    moments.y = __dp4a(static_cast<int>(a.x), static_cast<int>(a.x), moments.y);
    moments.y = __dp4a(static_cast<int>(a.y), static_cast<int>(a.y), moments.y);
    moments.y = __dp4a(static_cast<int>(a.z), static_cast<int>(a.z), moments.y);
    moments.y = __dp4a(static_cast<int>(a.w), static_cast<int>(a.w), moments.y);
    return moments;
}

/** Sixteen unsigned @c u8 products of a fragment pair - the unsigned @c dp4a form, whose native
 *  operand types need no cast chatter; the dot alone can pass the signed range at wide rows. */
__device__ __forceinline__ u32_t dense_cuda_dot_u8x16_ampere(uint4 a, uint4 b, u32_t sum) noexcept {
    sum = __dp4a(a.x, b.x, sum);
    sum = __dp4a(a.y, b.y, sum);
    sum = __dp4a(a.z, b.z, sum);
    return __dp4a(a.w, b.w, sum);
}

/** A @c u8 fragment's sum & sum of squares. */
__device__ __forceinline__ uint2 dense_cuda_moments_u8x16_ampere(uint4 a, uint2 moments) noexcept {
    moments.x = __dp4a(a.x, 0x01010101u, moments.x);
    moments.x = __dp4a(a.y, 0x01010101u, moments.x);
    moments.x = __dp4a(a.z, 0x01010101u, moments.x);
    moments.x = __dp4a(a.w, 0x01010101u, moments.x);
    moments.y = __dp4a(a.x, a.x, moments.y);
    moments.y = __dp4a(a.y, a.y, moments.y);
    moments.y = __dp4a(a.z, a.z, moments.y);
    moments.y = __dp4a(a.w, a.w, moments.y);
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c i8 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> dense_cuda_weighted_i8x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t byte = 0; byte != 16; ++byte)
        sums[byte] += weight * static_cast<f32_t>(static_cast<i8_t>(words[byte / 4] >> (8 * (byte % 4))));
    return sums;
}

/** Returns @p sums plus @p weight times each of a @c u8 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> dense_cuda_weighted_u8x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t byte = 0; byte != 16; ++byte)
        sums[byte] += weight * static_cast<f32_t>(static_cast<u8_t>(words[byte / 4] >> (8 * (byte % 4))));
    return sums;
}

/** i8 kit - the sm_80 floor. */
struct i8_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_i8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return dense_cuda_moments_i8x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_i8x16_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** u8 kit - the sm_80 floor. */
struct u8_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::u8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_u8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return dense_cuda_moments_u8x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_u8x16_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Byte Sized Integers

#pragma region Minifloats

/*  No generation has fp8 or fp6 arithmetic outside the tensor cores - conversion is all the ISA
 *  offers - so every minifloat kit widens byte pairs and accumulates like the 16-bit float kits,
 *  and one generation's kit differs from another's only in its widening. */

/** Sixteen minifloat products of a fragment pair, each byte pair widened by @p widening_type_ into
 *  f32. */
template <typename widening_type_>
__device__ __forceinline__ f32_t dense_cuda_dot_minifloats_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = widening_type_::pair_to_floats(words_a[word] >> (16 * pair));
            float2 const values_b = widening_type_::pair_to_floats(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** A minifloat fragment's sum & sum of squares, each byte pair widened by @p widening_type_ into
 *  f32. */
template <typename widening_type_>
__device__ __forceinline__ float2 dense_cuda_moments_minifloats_ampere(uint4 a, float2 moments) noexcept {
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

/** Returns @p sums plus @p weight times each of a minifloat fragment's sixteen values, each byte
 *  pair widened by @p widening_type_ into f32. */
template <typename widening_type_>
__device__ __forceinline__ std::array<f32_t, 16> dense_cuda_weighted_minifloats_ampere( //
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

/** The Ampere floor has no fp8 conversion, so each byte's fields are placed into an f32 by hand:
 *  subnormals scale by 2⁻⁹, and the all-ones magnitude is the format's only NaN. */
struct e4m3_ampere_widening_t {
    static __device__ __forceinline__ f32_t byte_to_float(u32_t byte) noexcept {
        u32_t const exponent = (byte >> 3) & 0xF, mantissa = byte & 0x7;
        f32_t const magnitude = (byte & 0x7F) == 0x7F ? __int_as_float(0x7FC00000)
                                : exponent
                                    ? __int_as_float(static_cast<int>(((exponent + 120u) << 23) | (mantissa << 20)))
                                    : static_cast<f32_t>(mantissa) * 0x1p-9f;
        return byte & 0x80 ? -magnitude : magnitude;
    }
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        return {byte_to_float(packed_pair & 0xFF), byte_to_float((packed_pair >> 8) & 0xFF)};
    }
};

/** The Ampere floor for @c e5m2, which is exactly the high byte of a half: shifting each byte up
 *  is the whole conversion, infinities and NaNs included. */
struct e5m2_ampere_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t const half_pair = ((packed_pair & 0xFF) << 8) | ((packed_pair & 0xFF00) << 16);
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }
};

/** The OCP fp6 formats sit low in a byte - sign at bit 5, no infinities or NaN - so the floor
 *  widens a pair by hand: the five magnitude bits drop into a half's exponent and mantissa fields
 *  as they stand, the sign moves to bit 15, and one exact power-of-two product rebiases, subnormals
 *  too. */
template <u32_t mantissa_bits_, u32_t rebias_log2_>
__device__ __forceinline__ float2 dense_cuda_fp6x2_to_f32x2_ampere(u32_t packed_pair) noexcept {
    u32_t const magnitudes = ((packed_pair & 0x001Fu) | ((packed_pair & 0x1F00u) << 8)) << (10 - mantissa_bits_);
    u32_t const signs = ((packed_pair & 0x0020u) | ((packed_pair & 0x2000u) << 8)) << 10;
    u32_t const unscaled = magnitudes | signs;
    __half2 const scale = __half2half2(__ushort_as_half(static_cast<unsigned short>((15u + rebias_log2_) << 10)));
    return __half22float2(__hmul2(*reinterpret_cast<__half2 const*>(&unscaled), scale));
}

/** @c e3m2 by hand: two mantissa bits, bias 3 against the half's 15. */
struct e3m2_ampere_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        return dense_cuda_fp6x2_to_f32x2_ampere<2, 12>(packed_pair);
    }
};

/** @c e2m3 by hand: three mantissa bits, bias 1 against the half's 15. */
struct e2m3_ampere_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        return dense_cuda_fp6x2_to_f32x2_ampere<3, 14>(packed_pair);
    }
};

/** e5m2 kit - the sm_80 floor, widening by shifts. */
struct e5m2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e5m2_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e5m2_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e5m2_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e4m3 kit - the sm_80 floor, widening by hand. */
struct e4m3_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e4m3_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e4m3_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e4m3_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e3m2 kit - the sm_80 floor, widening by hand. */
struct e3m2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e3m2_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e3m2_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e3m2_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e2m3 kit - the sm_80 floor, widening by hand. */
struct e2m3_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e2m3_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e2m3_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e2m3_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#if NUMKONG_TARGET_ADA

/** Ada and every tier since widens an fp8 pair in one @c cvt.rn.f16x2.e4m3x2. */
struct e4m3_ada_widening_t {
    static __device__ __forceinline__ u32_t pair_to_halves(u32_t packed_pair) noexcept {
        u32_t half_pair;
        asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
        return half_pair;
    }
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t const half_pair = pair_to_halves(packed_pair);
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }
};

/** The @c e5m2 twin of @ref e4m3_ada_widening_t, through @c cvt.rn.f16x2.e5m2x2. */
struct e5m2_ada_widening_t {
    static __device__ __forceinline__ u32_t pair_to_halves(u32_t packed_pair) noexcept {
        u32_t half_pair;
        asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
        return half_pair;
    }
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t const half_pair = pair_to_halves(packed_pair);
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }
};

/** e5m2 kit - Ada, widening through the conversion. */
struct e5m2_ada_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ada_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e5m2_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e5m2_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e5m2_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e4m3 kit - Ada, widening through the conversion. */
struct e4m3_ada_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ada_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e4m3_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e4m3_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e4m3_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_ADA

#if NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

/** Sixteen minifloat products of a fragment pair, each byte pair widened by @p widening_type_ into
 *  halves and multiplied straight into the f32 sum, so the dot never widens to f32 - the norms
 *  still do. */
template <typename widening_type_>
__device__ __forceinline__ f32_t dense_cuda_dot_minifloats_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair)
            sum = dense_cuda_fma_f16x2_blackwell(widening_type_::pair_to_halves(words_a[word] >> (16 * pair)),
                                                 widening_type_::pair_to_halves(words_b[word] >> (16 * pair)), sum);
    return sum;
}

/** Blackwell widens an fp6 pair in one @c cvt.rn.f16x2.e3m2x2, which only family-specific code
 *  encodes. */
struct e3m2_blackwell_widening_t {
    static __device__ __forceinline__ u32_t pair_to_halves(u32_t packed_pair) noexcept {
        u32_t half_pair;
        asm("{cvt.rn.f16x2.e3m2x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
        return half_pair;
    }
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t const half_pair = pair_to_halves(packed_pair);
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }
};

/** The @c e2m3 twin of @ref e3m2_blackwell_widening_t, through @c cvt.rn.f16x2.e2m3x2. */
struct e2m3_blackwell_widening_t {
    static __device__ __forceinline__ u32_t pair_to_halves(u32_t packed_pair) noexcept {
        u32_t half_pair;
        asm("{cvt.rn.f16x2.e2m3x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
        return half_pair;
    }
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t const half_pair = pair_to_halves(packed_pair);
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }
};

/** e5m2 kit - Blackwell, converted halves through @c fma.rn.f32.f16. */
struct e5m2_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e5m2_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e5m2_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e5m2_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e4m3 kit - Blackwell, converted halves through @c fma.rn.f32.f16. */
struct e4m3_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e4m3_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e4m3_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e4m3_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e3m2 kit - Blackwell, the one-instruction conversion into @c fma.rn.f32.f16. */
struct e3m2_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e3m2_blackwell_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e3m2_blackwell_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e3m2_blackwell_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** e2m3 kit - Blackwell, the one-instruction conversion into @c fma.rn.f32.f16. */
struct e2m3_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e2m3_blackwell_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e2m3_blackwell_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_minifloats_ampere<e2m3_blackwell_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#endif // NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

#pragma endregion Minifloats

#pragma region Packed Nibbles

/** Unpacks a word of eight signed nibbles into two byte-lane words, sign-extended through an
 *  xor-add-xor bias, since a word-wide subtraction would borrow across lanes. */
__device__ __forceinline__ uint2 dense_cuda_unpack_i4x8_ampere(u32_t word) noexcept {
    u32_t const low = (((word & 0x0F0F0F0Fu) ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
    u32_t const high = ((((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
    return {low, high};
}

/** Thirty-two signed @c i4 products of a fragment pair, two @c dp4a per source word. */
__device__ __forceinline__ int dense_cuda_dot_i4x32_ampere(uint4 a, uint4 b, int sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = dense_cuda_unpack_i4x8_ampere(words_a[word]);
        uint2 const unpacked_b = dense_cuda_unpack_i4x8_ampere(words_b[word]);
        sum = __dp4a(static_cast<int>(unpacked_a.x), static_cast<int>(unpacked_b.x), sum);
        sum = __dp4a(static_cast<int>(unpacked_a.y), static_cast<int>(unpacked_b.y), sum);
    }
    return sum;
}

/** An @c i4 fragment's sum & sum of squares. */
__device__ __forceinline__ int2 dense_cuda_moments_i4x32_ampere(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = dense_cuda_unpack_i4x8_ampere(words[word]);
        moments.x = __dp4a(static_cast<int>(unpacked.x), 0x01010101, moments.x);
        moments.x = __dp4a(static_cast<int>(unpacked.y), 0x01010101, moments.x);
        moments.y = __dp4a(static_cast<int>(unpacked.x), static_cast<int>(unpacked.x), moments.y);
        moments.y = __dp4a(static_cast<int>(unpacked.y), static_cast<int>(unpacked.y), moments.y);
    }
    return moments;
}

/** Unpacks a word of eight unsigned nibbles into two byte-lane words. */
__device__ __forceinline__ uint2 dense_cuda_unpack_u4x8_ampere(u32_t word) noexcept {
    return {word & 0x0F0F0F0Fu, (word >> 4) & 0x0F0F0F0Fu};
}

/** Thirty-two unsigned @c u4 products of a fragment pair, two @c dp4a per source word. */
__device__ __forceinline__ u32_t dense_cuda_dot_u4x32_ampere(uint4 a, uint4 b, u32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = dense_cuda_unpack_u4x8_ampere(words_a[word]);
        uint2 const unpacked_b = dense_cuda_unpack_u4x8_ampere(words_b[word]);
        sum = __dp4a(unpacked_a.x, unpacked_b.x, sum);
        sum = __dp4a(unpacked_a.y, unpacked_b.y, sum);
    }
    return sum;
}

/** A @c u4 fragment's sum & sum of squares. */
__device__ __forceinline__ uint2 dense_cuda_moments_u4x32_ampere(uint4 a, uint2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = dense_cuda_unpack_u4x8_ampere(words[word]);
        moments.x = __dp4a(unpacked.x, 0x01010101u, moments.x);
        moments.x = __dp4a(unpacked.y, 0x01010101u, moments.x);
        moments.y = __dp4a(unpacked.x, unpacked.x, moments.y);
        moments.y = __dp4a(unpacked.y, unpacked.y, moments.y);
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c i4 fragment's thirty-two values, through the
 *  same unpacking; the even column of a byte is its high nibble. */
__device__ __forceinline__ std::array<f32_t, 32> dense_cuda_weighted_i4x32_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 32> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = dense_cuda_unpack_i4x8_ampere(words[word]);
#pragma unroll
        for (u32_t byte = 0; byte != 4; ++byte) {
            sums[8 * word + 2 * byte] += weight * static_cast<f32_t>(static_cast<i8_t>(unpacked.y >> (8 * byte)));
            sums[8 * word + 2 * byte + 1] += weight * static_cast<f32_t>(static_cast<i8_t>(unpacked.x >> (8 * byte)));
        }
    }
    return sums;
}

/** The @c u4 twin of @ref dense_cuda_weighted_i4x32_ampere. */
__device__ __forceinline__ std::array<f32_t, 32> dense_cuda_weighted_u4x32_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 32> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = dense_cuda_unpack_u4x8_ampere(words[word]);
#pragma unroll
        for (u32_t byte = 0; byte != 4; ++byte) {
            sums[8 * word + 2 * byte] += weight * static_cast<f32_t>(static_cast<u8_t>(unpacked.y >> (8 * byte)));
            sums[8 * word + 2 * byte + 1] += weight * static_cast<f32_t>(static_cast<u8_t>(unpacked.x >> (8 * byte)));
        }
    }
    return sums;
}

/** i4x2 kit - the sm_80 floor, the byte engine over unpacked nibbles. */
struct i4x2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_i4x32_ampere(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return dense_cuda_moments_i4x32_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_i4x32_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

/** u4x2 kit - the sm_80 floor, the byte engine over unpacked nibbles. */
struct u4x2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::u4x2_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_u4x32_ampere(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return dense_cuda_moments_u4x32_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_u4x32_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                                norms_t target) noexcept {
        return dense_cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                           dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Packed Nibbles

#pragma region Packed Bits

/** The intersection popcount of a fragment pair via two 64-bit @c popc - the counter, not the
 *  logic, limits throughput, so the AND folds in front of as few @c popc as possible. */
__device__ __forceinline__ u32_t dense_cuda_intersect_u1x128_ampere(uint4 a, uint4 b) noexcept {
    u64_t const low = (static_cast<u64_t>(a.y) << 32 | a.x) & (static_cast<u64_t>(b.y) << 32 | b.x);
    u64_t const high = (static_cast<u64_t>(a.w) << 32 | a.z) & (static_cast<u64_t>(b.w) << 32 | b.z);
    return static_cast<u32_t>(__popcll(low) + __popcll(high));
}

/** Returns @p sums plus @p weight times each of a fragment's 128 bits, most significant bit of each
 *  byte first, as NumKong's casts order them. */
__device__ __forceinline__ std::array<f32_t, 128> dense_cuda_weighted_u1x128_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 128> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t bit = 0; bit != 128; ++bit)
        sums[bit] += weight * static_cast<f32_t>((words[bit / 32] >> (8 * (bit / 8 % 4) + 7 - bit % 8)) & 1u);
    return sums;
}

/** u1x8 kit - the sm_80 floor: the "dot" is the intersection popcount, and the norms store each
 *  row's set-bit count, the currency of Jaccard's union, mirrored into both fields. */
struct u1x8_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using lanes_t = cuda_lanes_t;
    using scalar_t = nk::u1x8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 128>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return sum + dense_cuda_intersect_u1x128_ampere(a, b);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        moments.x += dense_cuda_intersect_u1x128_ampere(a, a);
        return moments;
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return dense_cuda_weighted_u1x128_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.x};
    }

    /** Hamming is |a| + |b| - 2·intersection, never negative since the intersection is bounded by
     *  either count; Jaccard is 1 - intersection / union, and two empty sets are identical,
     *  distance zero. */
    static __device__ __forceinline__ dense_distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
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
    static __device__ __forceinline__ dense_distance_t convert_to_distance(nk_kernel_kind_t,
                                                                           dense_distance_t ranked) noexcept {
        return ranked;
    }
};

#pragma endregion Packed Bits

#pragma region Chains

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                      //
        f16_ampere_quads_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && (caps_ & nk_cap_blackwellrtx_k) != 0, f16_blackwell_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && (caps_ & nk_cap_blackwell_k) != 0, f16_blackwell_quads_kit_t>        //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                                                       //
        bf16_ampere_quads_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && (caps_ & nk_cap_blackwellrtx_k) != 0, bf16_blackwell_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && (caps_ & nk_cap_blackwell_k) != 0, bf16_blackwell_quads_kit_t>        //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::i8_t, caps_> {
    using kit_t = i8_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::u8_t, caps_> {
    using kit_t = u8_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                       //
        e5m2_ampere_quads_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && (caps_ & nk_cap_blackwellrtx_k) != 0, e5m2_blackwell_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && (caps_ & nk_cap_blackwell_k) != 0, e5m2_blackwell_quads_kit_t>,       //
        chain_layer<NUMKONG_TARGET_ADA && (caps_ & nk_cap_ada_k) != 0, e5m2_ada_quads_kit_t>                          //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                                       //
        e4m3_ampere_quads_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && (caps_ & nk_cap_blackwellrtx_k) != 0, e4m3_blackwell_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && (caps_ & nk_cap_blackwell_k) != 0, e4m3_blackwell_quads_kit_t>,       //
        chain_layer<NUMKONG_TARGET_ADA && (caps_ & nk_cap_ada_k) != 0, e4m3_ada_quads_kit_t>                          //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::e3m2_t, caps_> {
    using kit_t = typename first_available_kit<                                                                       //
        e3m2_ampere_quads_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && (caps_ & nk_cap_blackwellrtx_k) != 0, e3m2_blackwell_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && (caps_ & nk_cap_blackwell_k) != 0, e3m2_blackwell_quads_kit_t>        //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::e2m3_t, caps_> {
    using kit_t = typename first_available_kit<                                                                       //
        e2m3_ampere_quads_kit_t,                                                                                      //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && (caps_ & nk_cap_blackwellrtx_k) != 0, e2m3_blackwell_quads_kit_t>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && (caps_ & nk_cap_blackwell_k) != 0, e2m3_blackwell_quads_kit_t>        //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::i4x2_t, caps_> {
    using kit_t = i4x2_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::u4x2_t, caps_> {
    using kit_t = u4x2_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct dense_quads_chain<nk::u1x8_t, caps_> {
    using kit_t = u1x8_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Chains

} // namespace usearch
} // namespace unum
