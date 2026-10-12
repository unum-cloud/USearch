/**
 *  @file include/usearch/dense/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Device metric kits for the dense VRAM indexes - the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one storage dtype to one GPU generation, named like the CPU kits -
 *  @b dtype_generation_quads_kit_t. The generation slot names the oldest tier whose instructions
 *  the kit needs: @c ampere for the sm_80 floor, @c ada for the fp8 conversions, @c blackwell for
 *  the mixed-precision multiply-adds and the fp6 conversions. A chain picks among them under the
 *  class's ceiling alone, never under a @c NUMKONG_TARGET_* switch, and only the picked kit's
 *  instructions are emitted, so a unit compiled for one architecture carries no newer one's. The
 *  arithmetic lives in named device primitives; a kit binds them.
 *
 *  Kits stay pure arithmetic. How a warp's candidate rows reach shared memory is a separate
 *  @b schedule, picked under the same ceiling: @c cp.async commit groups through Ada, whole-row
 *  Tensor Memory Accelerator copies from Hopper on.
 *
 *  Fragments are 16 bytes of scalars held in registers, and metrics with precomputable norms
 *  finalize "through the dot". Row geometry - how many fragments span one vector, how many lanes
 *  cooperate on a row - and the metric are runtime state of the flat view, not the kit:
 *  dimensions are frozen by the first @c reserve, never by a type, and the metric only steers the
 *  scalar epilogue, so one kernel serves every metric of a dtype. The cross-lane reduction is
 *  deliberately not a kit member - one shuffle tree in @c vram_index.cuh serves every kit, over the
 *  warp policy each kit names. Distances take both sides' norms, as the CPU finalizers do, so a
 *  symmetric candidate-to-candidate distance is the same call.
 *
 *  Families sort from the widest numeric types to the smallest & most exotic: 16-bit floats, 8-bit
 *  integers, minifloats, packed nibbles, packed bits. Norms are the CPU's @c vector_norms over each
 *  kit's dot type, so the unsigned families accumulate in unsigned words with defined wrapping and
 *  no cast chatter, while @c i8 keeps the CPU norms' signed @b {sum,sumsq} layout.
 */
#pragma once

#include <array> // `std::array` - a fragment's weighted sums

#include <cuda/atomic>     // `cuda::atomic_ref` - the device twin of `std::atomic_ref`
#include <cuda_bf16.h>     // `__nv_bfloat162` packed math, sm_80+
#include <cuda_fp16.h>     // `__half2` packed math
#include <cuda_pipeline.h> // `__pipeline_memcpy_async` - the Ampere schedule's register-free copies

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the epilogues switch on it
#include <numkong/types.hpp>      // `nk::bf16_t` & the other scalars the kits load

#include "usearch/utilities/types.hpp"
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

/** The device-scope atomic reference the NVIDIA kernels reach a shared word through, so each
 *  access names its memory order - relaxed wherever the launch boundary is what publishes the
 *  result. */
template <typename value_type_>
using cuda_atomic_ref = cuda::atomic_ref<value_type_, cuda::thread_scope_device>;

/** The shared-window address of @p pointer, the one the asynchronous copies and barriers take. */
__device__ __forceinline__ u32_t cuda_shared_address(void const* pointer) noexcept {
    return static_cast<u32_t>(__cvta_generic_to_shared(pointer));
}

/**
 *  @brief The NVIDIA warp policy the shared walk code is written against: 32-lane warps with
 *      32-bit lane masks and device-scope atomics.
 *
 *  Every collective takes the full-warp mask, since the walks keep their warps converged.
 */
struct cuda_warp_t {
    using mask_t = u32_t;
    inline static constexpr u32_t width_k = 32;
    inline static constexpr mask_t all_k = 0xFFFFFFFFu;

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle(value_type_ value, u32_t source_lane) noexcept {
        return __shfl_sync(all_k, value, source_lane);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_xor(value_type_ value, u32_t lane_mask) noexcept {
        return __shfl_xor_sync(all_k, value, lane_mask);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_up(value_type_ value, u32_t delta) noexcept {
        return __shfl_up_sync(all_k, value, delta);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_down(value_type_ value, u32_t delta,
                                                               u32_t width = width_k) noexcept {
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
    static __device__ __forceinline__ u32_t first(mask_t mask) noexcept { return __ffs(mask) - 1; }

    /** The lanes strictly below @p lane. */
    static __device__ __forceinline__ mask_t below(u32_t lane) noexcept { return (1u << lane) - 1; }

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_relaxed(value_type_& word) noexcept {
        return cuda_atomic_ref<value_type_>(word).load(cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_acquire(value_type_& word) noexcept {
        return cuda_atomic_ref<value_type_>(word).load(cuda::memory_order_acquire);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_relaxed(value_type_& word, value_type_ value) noexcept {
        cuda_atomic_ref<value_type_>(word).store(value, cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_release(value_type_& word, value_type_ value) noexcept {
        cuda_atomic_ref<value_type_>(word).store(value, cuda::memory_order_release);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ exchange_acquire(value_type_& word, value_type_ value) noexcept {
        return cuda_atomic_ref<value_type_>(word).exchange(value, cuda::memory_order_acquire);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_add_relaxed(value_type_& word, value_type_ value) noexcept {
        return cuda_atomic_ref<value_type_>(word).fetch_add(value, cuda::memory_order_relaxed);
    }

    /** Orders every store before it ahead of any store after it, for readers anywhere on the
     *  device. */
    static __device__ __forceinline__ void fence_release() noexcept {
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
    }
};

#pragma region Metrics

/** Angular distance from a dot and both sides' squared norms, finalized exactly like the CPU's
 *  @c nk_angular_through_f32_from_dot: clamped at zero, and a zero-norm side scores zero against a
 *  vanishing dot and one otherwise.
 *  The query's root is loop-invariant, so the compiler hoists it. */
__device__ __forceinline__ distance_t cuda_angular_through_dot(f32_t dot, f32_t query_sumsq,
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
__device__ __forceinline__ distance_t cuda_distance(nk_kernel_kind_t kernel, dot_type_ dot,
                                                    vector_norms<dot_type_> query,
                                                    vector_norms<dot_type_> target) noexcept {
    f32_t const dot_value = static_cast<f32_t>(dot);
    switch (kernel) {
    case nk_kernel_dot_k: return 1.0f - dot_value;
    case nk_kernel_angular_k:
        return cuda_angular_through_dot(dot_value, static_cast<f32_t>(query.sumsq), static_cast<f32_t>(target.sumsq));
    default: return static_cast<f32_t>(query.sumsq) + static_cast<f32_t>(target.sumsq) - 2.0f * dot_value;
    }
}

/** Output-boundary finalize, the CPU kits' @c convert_to_distances in miniature: the traversal
 *  ranks on raw @c distance values end to end, and only the matches a caller receives pass through
 *  here - identity unless a metric owes a monotone transform, like Euclidean's square root. */
__device__ __forceinline__ distance_t cuda_convert_to_distance(nk_kernel_kind_t kernel, distance_t ranked) noexcept {
    return kernel == nk_kernel_euclidean_k ? sqrtf(fmaxf(ranked, 0.0f)) : ranked;
}

#pragma endregion Metrics

#pragma region Half Precision Floats

/** Eight @c f16 products of a fragment pair, each rounded to half by @c hmul2, summed in f32 per
 *  fragment so no half-range overflow can accumulate. */
__device__ __forceinline__ f32_t cuda_dot_f16x8_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
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
__device__ __forceinline__ float2 cuda_moments_f16x8_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        float2 const values = __half22float2(*reinterpret_cast<__half2 const*>(&words[word]));
        moments.x += values.x + values.y;
        moments.y += values.x * values.x + values.y * values.y;
    }
    return moments;
}

/** The @c bf16 twin of @ref cuda_dot_f16x8_ampere, through the @c bf16x2 instructions that
 *  land exactly at the sm_80 floor. */
__device__ __forceinline__ f32_t cuda_dot_bf16x8_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
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
__device__ __forceinline__ float2 cuda_moments_bf16x8_ampere(uint4 a, float2 moments) noexcept {
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
__device__ __forceinline__ std::array<f32_t, 8> cuda_weighted_f16x8_ampere( //
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

/** The @c bf16 twin of @ref cuda_weighted_f16x8_ampere. */
__device__ __forceinline__ std::array<f32_t, 8> cuda_weighted_bf16x8_ampere( //
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
    using warp_t = cuda_warp_t;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_f16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_f16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_f16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** bf16 kit - the sm_80 floor. */
struct bf16_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_bf16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_bf16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_bf16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** Both halves of two @c f16x2 words multiplied straight into an f32 sum by @c fma.rn.f32.f16 -
 *  exact products, so none rounds to half or overflows its range. */
__device__ __forceinline__ f32_t cuda_fma_f16x2_blackwell(u32_t first_pair, u32_t second_pair, f32_t sum) noexcept {
    asm("{.reg .b16 first_low, first_high, second_low, second_high;\n\t" //
        "mov.b32 {first_low, first_high}, %1;\n\t"                       //
        "mov.b32 {second_low, second_high}, %2;\n\t"                     //
        "fma.rn.f32.f16 %0, first_low, second_low, %0;\n\t"              //
        "fma.rn.f32.f16 %0, first_high, second_high, %0;}"
        : "+f"(sum)
        : "r"(first_pair), "r"(second_pair));
    return sum;
}

/** The @c bf16x2 twin of @ref cuda_fma_f16x2_blackwell, through @c fma.rn.f32.bf16. */
__device__ __forceinline__ f32_t cuda_fma_bf16x2_blackwell(u32_t first_pair, u32_t second_pair, f32_t sum) noexcept {
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
__device__ __forceinline__ f32_t cuda_dot_f16x8_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    sum = cuda_fma_f16x2_blackwell(a.x, b.x, sum);
    sum = cuda_fma_f16x2_blackwell(a.y, b.y, sum);
    sum = cuda_fma_f16x2_blackwell(a.z, b.z, sum);
    return cuda_fma_f16x2_blackwell(a.w, b.w, sum);
}

/** Eight @c bf16 products of a fragment pair, each exact, fused straight into the f32 sum. */
__device__ __forceinline__ f32_t cuda_dot_bf16x8_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    sum = cuda_fma_bf16x2_blackwell(a.x, b.x, sum);
    sum = cuda_fma_bf16x2_blackwell(a.y, b.y, sum);
    sum = cuda_fma_bf16x2_blackwell(a.z, b.z, sum);
    return cuda_fma_bf16x2_blackwell(a.w, b.w, sum);
}

/** f16 kit - Blackwell, @c fma.rn.f32.f16. */
struct f16_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_f16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_f16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_f16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** bf16 kit - Blackwell, @c fma.rn.f32.bf16. */
struct bf16_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_bf16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_bf16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_bf16x8_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Half Precision Floats

#pragma region Byte Sized Integers

/** Sixteen signed @c i8 products of a fragment pair, one @c dp4a per word. */
__device__ __forceinline__ int cuda_dot_i8x16_ampere(uint4 a, uint4 b, int sum) noexcept {
    sum = __dp4a(static_cast<int>(a.x), static_cast<int>(b.x), sum);
    sum = __dp4a(static_cast<int>(a.y), static_cast<int>(b.y), sum);
    sum = __dp4a(static_cast<int>(a.z), static_cast<int>(b.z), sum);
    return __dp4a(static_cast<int>(a.w), static_cast<int>(b.w), sum);
}

/** An @c i8 fragment's sum & sum of squares, through the same instruction against splat-one and
 *  self. */
__device__ __forceinline__ int2 cuda_moments_i8x16_ampere(uint4 a, int2 moments) noexcept {
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
__device__ __forceinline__ u32_t cuda_dot_u8x16_ampere(uint4 a, uint4 b, u32_t sum) noexcept {
    sum = __dp4a(a.x, b.x, sum);
    sum = __dp4a(a.y, b.y, sum);
    sum = __dp4a(a.z, b.z, sum);
    return __dp4a(a.w, b.w, sum);
}

/** A @c u8 fragment's sum & sum of squares. */
__device__ __forceinline__ uint2 cuda_moments_u8x16_ampere(uint4 a, uint2 moments) noexcept {
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
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_i8x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t byte = 0; byte != 16; ++byte)
        sums[byte] += weight * static_cast<f32_t>(static_cast<i8_t>(words[byte / 4] >> (8 * (byte % 4))));
    return sums;
}

/** Returns @p sums plus @p weight times each of a @c u8 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_u8x16_ampere( //
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
    using warp_t = cuda_warp_t;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_i8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return cuda_moments_i8x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_i8x16_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** u8 kit - the sm_80 floor. */
struct u8_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::u8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_u8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return cuda_moments_u8x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_u8x16_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
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
__device__ __forceinline__ f32_t cuda_dot_minifloats_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
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
__device__ __forceinline__ float2 cuda_moments_minifloats_ampere(uint4 a, float2 moments) noexcept {
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
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_minifloats_ampere( //
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
__device__ __forceinline__ float2 cuda_fp6x2_to_f32x2_ampere(u32_t packed_pair) noexcept {
    u32_t const magnitudes = ((packed_pair & 0x001Fu) | ((packed_pair & 0x1F00u) << 8)) << (10 - mantissa_bits_);
    u32_t const signs = ((packed_pair & 0x0020u) | ((packed_pair & 0x2000u) << 8)) << 10;
    u32_t const unscaled = magnitudes | signs;
    __half2 const scale = __half2half2(__ushort_as_half(static_cast<unsigned short>((15u + rebias_log2_) << 10)));
    return __half22float2(__hmul2(*reinterpret_cast<__half2 const*>(&unscaled), scale));
}

/** @c e3m2 by hand: two mantissa bits, bias 3 against the half's 15. */
struct e3m2_ampere_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        return cuda_fp6x2_to_f32x2_ampere<2, 12>(packed_pair);
    }
};

/** @c e2m3 by hand: three mantissa bits, bias 1 against the half's 15. */
struct e2m3_ampere_widening_t {
    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        return cuda_fp6x2_to_f32x2_ampere<3, 14>(packed_pair);
    }
};

/** e5m2 kit - the sm_80 floor, widening by shifts. */
struct e5m2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_ampere<e5m2_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e5m2_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e5m2_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e4m3 kit - the sm_80 floor, widening by hand. */
struct e4m3_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_ampere<e4m3_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e4m3_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e4m3_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e3m2 kit - the sm_80 floor, widening by hand. */
struct e3m2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_ampere<e3m2_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e3m2_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e3m2_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e2m3 kit - the sm_80 floor, widening by hand. */
struct e2m3_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_ampere<e2m3_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e2m3_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e2m3_ampere_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

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
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_ampere<e5m2_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e5m2_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e5m2_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e4m3 kit - Ada, widening through the conversion. */
struct e4m3_ada_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ada_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_ampere<e4m3_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e4m3_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e4m3_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** Sixteen minifloat products of a fragment pair, each byte pair widened by @p widening_type_ into
 *  halves and multiplied straight into the f32 sum, so the dot never widens to f32 - the norms
 *  still do. */
template <typename widening_type_>
__device__ __forceinline__ f32_t cuda_dot_minifloats_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair)
            sum = cuda_fma_f16x2_blackwell(widening_type_::pair_to_halves(words_a[word] >> (16 * pair)),
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
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_blackwell<e5m2_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e5m2_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e5m2_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e4m3 kit - Blackwell, converted halves through @c fma.rn.f32.f16. */
struct e4m3_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_blackwell<e4m3_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e4m3_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e4m3_ada_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e3m2 kit - Blackwell, the one-instruction conversion into @c fma.rn.f32.f16. */
struct e3m2_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_blackwell<e3m2_blackwell_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e3m2_blackwell_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e3m2_blackwell_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** e2m3 kit - Blackwell, the one-instruction conversion into @c fma.rn.f32.f16. */
struct e2m3_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_minifloats_blackwell<e2m3_blackwell_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_minifloats_ampere<e2m3_blackwell_widening_t>(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_minifloats_ampere<e2m3_blackwell_widening_t>(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Minifloats

#pragma region Packed Nibbles

/** Unpacks a word of eight signed nibbles into two byte-lane words, sign-extended through an
 *  xor-add-xor bias, since a word-wide subtraction would borrow across lanes. */
__device__ __forceinline__ uint2 cuda_unpack_i4x8_ampere(u32_t word) noexcept {
    u32_t const low = (((word & 0x0F0F0F0Fu) ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
    u32_t const high = ((((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
    return {low, high};
}

/** Thirty-two signed @c i4 products of a fragment pair, two @c dp4a per source word. */
__device__ __forceinline__ int cuda_dot_i4x32_ampere(uint4 a, uint4 b, int sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = cuda_unpack_i4x8_ampere(words_a[word]);
        uint2 const unpacked_b = cuda_unpack_i4x8_ampere(words_b[word]);
        sum = __dp4a(static_cast<int>(unpacked_a.x), static_cast<int>(unpacked_b.x), sum);
        sum = __dp4a(static_cast<int>(unpacked_a.y), static_cast<int>(unpacked_b.y), sum);
    }
    return sum;
}

/** An @c i4 fragment's sum & sum of squares. */
__device__ __forceinline__ int2 cuda_moments_i4x32_ampere(uint4 a, int2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = cuda_unpack_i4x8_ampere(words[word]);
        moments.x = __dp4a(static_cast<int>(unpacked.x), 0x01010101, moments.x);
        moments.x = __dp4a(static_cast<int>(unpacked.y), 0x01010101, moments.x);
        moments.y = __dp4a(static_cast<int>(unpacked.x), static_cast<int>(unpacked.x), moments.y);
        moments.y = __dp4a(static_cast<int>(unpacked.y), static_cast<int>(unpacked.y), moments.y);
    }
    return moments;
}

/** Unpacks a word of eight unsigned nibbles into two byte-lane words. */
__device__ __forceinline__ uint2 cuda_unpack_u4x8_ampere(u32_t word) noexcept {
    return {word & 0x0F0F0F0Fu, (word >> 4) & 0x0F0F0F0Fu};
}

/** Thirty-two unsigned @c u4 products of a fragment pair, two @c dp4a per source word. */
__device__ __forceinline__ u32_t cuda_dot_u4x32_ampere(uint4 a, uint4 b, u32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked_a = cuda_unpack_u4x8_ampere(words_a[word]);
        uint2 const unpacked_b = cuda_unpack_u4x8_ampere(words_b[word]);
        sum = __dp4a(unpacked_a.x, unpacked_b.x, sum);
        sum = __dp4a(unpacked_a.y, unpacked_b.y, sum);
    }
    return sum;
}

/** A @c u4 fragment's sum & sum of squares. */
__device__ __forceinline__ uint2 cuda_moments_u4x32_ampere(uint4 a, uint2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = cuda_unpack_u4x8_ampere(words[word]);
        moments.x = __dp4a(unpacked.x, 0x01010101u, moments.x);
        moments.x = __dp4a(unpacked.y, 0x01010101u, moments.x);
        moments.y = __dp4a(unpacked.x, unpacked.x, moments.y);
        moments.y = __dp4a(unpacked.y, unpacked.y, moments.y);
    }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c i4 fragment's thirty-two values, through the
 *  same unpacking; the even column of a byte is its high nibble. */
__device__ __forceinline__ std::array<f32_t, 32> cuda_weighted_i4x32_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 32> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = cuda_unpack_i4x8_ampere(words[word]);
#pragma unroll
        for (u32_t byte = 0; byte != 4; ++byte) {
            sums[8 * word + 2 * byte] += weight * static_cast<f32_t>(static_cast<i8_t>(unpacked.y >> (8 * byte)));
            sums[8 * word + 2 * byte + 1] += weight * static_cast<f32_t>(static_cast<i8_t>(unpacked.x >> (8 * byte)));
        }
    }
    return sums;
}

/** The @c u4 twin of @ref cuda_weighted_i4x32_ampere. */
__device__ __forceinline__ std::array<f32_t, 32> cuda_weighted_u4x32_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 32> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word) {
        uint2 const unpacked = cuda_unpack_u4x8_ampere(words[word]);
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
    using warp_t = cuda_warp_t;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_i4x32_ampere(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return cuda_moments_i4x32_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_i4x32_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

/** u4x2 kit - the sm_80 floor, the byte engine over unpacked nibbles. */
struct u4x2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    using warp_t = cuda_warp_t;
    using scalar_t = nk::u4x2_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 32>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_u4x32_ampere(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return cuda_moments_u4x32_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_u4x32_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ distance_t distance(nk_kernel_kind_t kernel, dot_t dot, norms_t query,
                                                          norms_t target) noexcept {
        return cuda_distance(kernel, dot, query, target);
    }
    static __device__ __forceinline__ distance_t convert_to_distance(nk_kernel_kind_t kernel,
                                                                     distance_t ranked) noexcept {
        return cuda_convert_to_distance(kernel, ranked);
    }
};

#pragma endregion Packed Nibbles

#pragma region Packed Bits

/** The intersection popcount of a fragment pair via two 64-bit @c popc - the counter, not the
 *  logic, limits throughput, so the AND folds in front of as few @c popc as possible. */
__device__ __forceinline__ u32_t cuda_intersect_u1x128_ampere(uint4 a, uint4 b) noexcept {
    u64_t const low = (static_cast<u64_t>(a.y) << 32 | a.x) & (static_cast<u64_t>(b.y) << 32 | b.x);
    u64_t const high = (static_cast<u64_t>(a.w) << 32 | a.z) & (static_cast<u64_t>(b.w) << 32 | b.z);
    return static_cast<u32_t>(__popcll(low) + __popcll(high));
}

/** Returns @p sums plus @p weight times each of a fragment's 128 bits, most significant bit of each
 *  byte first, as NumKong's casts order them. */
__device__ __forceinline__ std::array<f32_t, 128> cuda_weighted_u1x128_ampere( //
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
    using warp_t = cuda_warp_t;
    using scalar_t = nk::u1x8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 128>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return sum + cuda_intersect_u1x128_ampere(a, b);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        moments.x += cuda_intersect_u1x128_ampere(a, a);
        return moments;
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_u1x128_ampere(a, weight, sums);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.x};
    }

    /** Hamming is |a| + |b| - 2·intersection, never negative since the intersection is bounded by
     *  either count; Jaccard is 1 - intersection / union, and two empty sets are identical,
     *  distance zero. */
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

/**
 *  @brief The Ampere and Ada staging: every lane queues register-free @c cp.async copies of the
 *      fragments it strides over, one commit group per stage, so a consumer walking the rows in
 *      order starts on the first stage while the later ones land.
 *
 *  A @c cp.async lands only for the lane that queued it, so every await ends in a warp barrier.
 */
struct ampere_cp_async_schedule_t {
    inline static constexpr u32_t stages_k = 2;

    /** Nothing shared: the commit groups live in each lane's own pipeline. */
    struct shared_t {};

    /** Rows per stage of the latest staging, and the leading rows landed for every lane. */
    u32_t rows_per_stage_ = 0, landed_rows_ = 0;

    __device__ __forceinline__ ampere_cp_async_schedule_t(shared_t&, u32_t) noexcept {}

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        rows_per_stage_ = divide_round_up(rows, stages_k), landed_rows_ = 0;
        // Every stage commits, empty or not, so the groups a stage waits behind are a constant.
        for (u32_t stage = 0; stage != stages_k; ++stage) {
            u32_t const end_word = min_of((stage + 1) * rows_per_stage_, rows) * storage.row_fragments;
            for (u32_t word = min_of(stage * rows_per_stage_, rows) * storage.row_fragments + lane; word < end_word;
                 word += cuda_warp_t::width_k) {
                u32_t const row = storage.row_of_word(word), fragment = storage.fragment_of_word(word, row);
                __pipeline_memcpy_async(&cache.at(row, fragment), storage.vector_span_for(ids[row]) + fragment,
                                        sizeof(typename storage_::fragment_t));
            }
            __pipeline_commit();
        }
    }

    __device__ __forceinline__ void await_rows(u32_t rows) noexcept {
        if (rows <= landed_rows_) return;
        u32_t const stages = divide_round_up(rows, rows_per_stage_);
        wait_pending_(stages_k - stages);
        landed_rows_ = stages * rows_per_stage_;
        __syncwarp();
    }

  private:

    /** Waits until at most @p pending of this lane's newest commit groups remain in flight, through
     *  the immediate @c cp.async.wait_group takes. */
    template <u32_t ceiling_ = stages_k - 1>
    static __device__ __forceinline__ void wait_pending_(u32_t pending) noexcept {
        if constexpr (ceiling_ == 0) __pipeline_wait_prior(0);
        else if (pending >= ceiling_) __pipeline_wait_prior(ceiling_);
        else wait_pending_<ceiling_ - 1>(pending);
    }
};

/**
 *  @brief The Hopper and Blackwell staging: the Tensor Memory Accelerator copies every row whole in
 *      one @c cp.async.bulk, and each stage's bytes complete on that stage's @c mbarrier, so the
 *      copies hold no registers and every lane waits on the barrier, not on its own copies.
 *
 *  A row is one contiguous span of whole 16-byte fragments, so no tensor map is needed and no row
 *  length is capped. Each barrier advances one phase per staging that fills its stage.
 */
struct hopper_tma_schedule_t {
    inline static constexpr u32_t stages_k = 2;

    /** One barrier per stage, in the warp's static shared memory. */
    struct shared_t {
        u64_t barriers[stages_k];
    };

    /** The stages' barriers. */
    u64_t* barriers_;

    /** Rows per stage of the latest staging, and the leading rows landed. */
    u32_t rows_per_stage_ = 0, landed_rows_ = 0;

    /** Bit @b s is the parity of the phase stage @b s is filling. */
    u32_t phases_ = 0;

    __device__ __forceinline__ hopper_tma_schedule_t(shared_t& shared, u32_t lane) noexcept
        : barriers_(shared.barriers) {
        if (lane == 0) {
            for (u32_t stage = 0; stage != stages_k; ++stage)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(barrier_(stage)), "r"(1u) : "memory");
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncwarp();
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        u32_t const row_bytes = storage.row_fragments * sizeof(typename storage_::fragment_t);
        rows_per_stage_ = divide_round_up(rows, stages_k), landed_rows_ = 0;
        // The cache rows' earlier generic reads must precede the copies' asynchronous writes.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (lane == 0)
            for (u32_t first_row = 0; first_row < rows; first_row += rows_per_stage_) {
                u32_t const barrier = barrier_(first_row / rows_per_stage_);
                u32_t const stage_bytes = (min_of(first_row + rows_per_stage_, rows) - first_row) * row_bytes;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(stage_bytes)
                             : "memory");
            }
        __syncwarp();
        for (u32_t row = lane; row < rows; row += cuda_warp_t::width_k) {
            u32_t const destination = cuda_shared_address(cache.row(row));
            u32_t const barrier = barrier_(row / rows_per_stage_);
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" //
                         ::"r"(destination),
                         "l"(storage.vector_span_for(ids[row])), "r"(row_bytes), "r"(barrier)
                         : "memory");
        }
    }

    __device__ __forceinline__ void await_rows(u32_t rows) noexcept {
        for (; landed_rows_ < rows; landed_rows_ += rows_per_stage_) {
            u32_t const stage = landed_rows_ / rows_per_stage_;
            u32_t landed;
            do {
                asm volatile("{\n.reg .pred ready;\n"                                      //
                             "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2;\n" //
                             "selp.u32 %0, 1, 0, ready;\n}\n"
                             : "=r"(landed)
                             : "r"(barrier_(stage)), "r"((phases_ >> stage) & 1u)
                             : "memory");
            } while (!landed);
            phases_ ^= 1u << stage;
        }
    }

  private:
    __device__ __forceinline__ u32_t barrier_(u32_t stage) const noexcept {
        return cuda_shared_address(barriers_ + stage);
    }
};

#pragma endregion Schedules

#pragma region Chains

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::f16_t, caps_> {
    using kit_t = typename first_available_kit<                                       //
        f16_ampere_quads_kit_t,                                                       //
        chain_layer<(caps_ & nk_cap_blackwellrtx_k) != 0, f16_blackwell_quads_kit_t>, //
        chain_layer<(caps_ & nk_cap_blackwell_k) != 0, f16_blackwell_quads_kit_t>     //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::bf16_t, caps_> {
    using kit_t = typename first_available_kit<                                        //
        bf16_ampere_quads_kit_t,                                                       //
        chain_layer<(caps_ & nk_cap_blackwellrtx_k) != 0, bf16_blackwell_quads_kit_t>, //
        chain_layer<(caps_ & nk_cap_blackwell_k) != 0, bf16_blackwell_quads_kit_t>     //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::i8_t, caps_> {
    using kit_t = i8_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::u8_t, caps_> {
    using kit_t = u8_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::e5m2_t, caps_> {
    using kit_t = typename first_available_kit<                                        //
        e5m2_ampere_quads_kit_t,                                                       //
        chain_layer<(caps_ & nk_cap_blackwellrtx_k) != 0, e5m2_blackwell_quads_kit_t>, //
        chain_layer<(caps_ & nk_cap_blackwell_k) != 0, e5m2_blackwell_quads_kit_t>,    //
        chain_layer<(caps_ & nk_cap_ada_k) != 0, e5m2_ada_quads_kit_t>                 //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::e4m3_t, caps_> {
    using kit_t = typename first_available_kit<                                        //
        e4m3_ampere_quads_kit_t,                                                       //
        chain_layer<(caps_ & nk_cap_blackwellrtx_k) != 0, e4m3_blackwell_quads_kit_t>, //
        chain_layer<(caps_ & nk_cap_blackwell_k) != 0, e4m3_blackwell_quads_kit_t>,    //
        chain_layer<(caps_ & nk_cap_ada_k) != 0, e4m3_ada_quads_kit_t>                 //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::e3m2_t, caps_> {
    using kit_t = typename first_available_kit<                                        //
        e3m2_ampere_quads_kit_t,                                                       //
        chain_layer<(caps_ & nk_cap_blackwellrtx_k) != 0, e3m2_blackwell_quads_kit_t>, //
        chain_layer<(caps_ & nk_cap_blackwell_k) != 0, e3m2_blackwell_quads_kit_t>     //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::e2m3_t, caps_> {
    using kit_t = typename first_available_kit<                                        //
        e2m3_ampere_quads_kit_t,                                                       //
        chain_layer<(caps_ & nk_cap_blackwellrtx_k) != 0, e2m3_blackwell_quads_kit_t>, //
        chain_layer<(caps_ & nk_cap_blackwell_k) != 0, e2m3_blackwell_quads_kit_t>     //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::i4x2_t, caps_> {
    using kit_t = i4x2_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::u4x2_t, caps_> {
    using kit_t = u4x2_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_chain<nk::u1x8_t, caps_> {
    using kit_t = u1x8_ampere_quads_kit_t;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

/** The staging schedule under @p caps_: whole-row Tensor Memory Accelerator copies on every class
 *  that has one, Hopper on, and the @c cp.async ring below. */
template <nk_capability_t caps_>
using cuda_schedule_chain = typename first_available_kit<                                      //
    ampere_cp_async_schedule_t,                                                                //
    chain_layer<(caps_ & (nk_cap_hopper_k | nk_cap_blackwell_k | nk_cap_blackwellrtx_k)) != 0, //
                hopper_tma_schedule_t>                                                         //
    >::type;

#pragma endregion Chains

} // namespace unum::usearch::dense
