/**
 *  @file include/usearch/dense/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Device metric kits for the dense VRAM indexes - the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one storage dtype to one GPU generation, named like the CPU kits -
 *  @b dtype_generation_quads_kit_t. The generation slot names the oldest tier whose instructions
 *  the kit needs: @c ampere for the sm_80 floor, @c ada for the fp8 conversions, @c blackwell for
 *  the mixed-precision multiply-adds and the fp6 conversions. Each kit states the capability it
 *  runs under, @c capability_k, and whether this unit carries it, @c compiled_k, from NumKong's
 *  cumulative @c NUMKONG_ARCH_CUDA_* switches, so a Blackwell unit keeps the Ampere floor. A ladder
 *  picks the first kit both admit through @c highest_kit, and only the picked kit's instructions
 *  are emitted, so a unit compiled for one architecture carries no newer one's. The arithmetic
 *  lives in named device primitives; a kit binds them.
 *
 *  Kits stay pure arithmetic. How a warp's candidate rows reach shared memory is a separate
 *  @b schedule, picked by @c schedule_ladder the same way: @c cp.async commit groups through Ada,
 *  whole-row Tensor Memory Accelerator copies from Hopper on. The schedule also scores a walk's
 *  batch: through the kit's @c mma.sync step from registers where the kit has one, and otherwise
 *  through its scalar dots, a lane loading one fragment at once through Hopper and two from
 *  Blackwell on, whose 256-bit loads serve the random row gathers better. The datacenter Blackwell
 *  instead copies the batch's rows four at a time through @c tile::gather4 copies of one tensor
 *  map, scoring each window of four rows while the next copies in.
 *
 *  Fragments are 16 bytes of scalars held in registers, and metrics with precomputable norms
 *  finalize "through the dot". Row geometry - how many fragments span one vector, how many lanes
 *  cooperate on a row - and the metric are runtime state of the flat view, not the kit: dimensions
 *  are fixed at construction, never by a type, and the metric only steers the scalar epilogue, so
 *  one kernel serves every metric of a dtype. The cross-lane reduction is deliberately not a kit
 *  member - one shuffle tree in @c vram_index.cuh serves every kit, over the warp policy each
 *  schedule names. Distances take both sides' norms, as the CPU finalizers do, so a symmetric
 *  candidate-to-candidate distance is the same call.
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
#include "usearch/utilities/vram_nvidia_device.cuh"
#include "usearch/dense/vram_index.cuh"

namespace unum::usearch::dense {

/** The shared-window address of @p pointer, the one the asynchronous copies and barriers take. */
__device__ __forceinline__ u32_t cuda_shared_address(void const* pointer) noexcept {
    return static_cast<u32_t>(__cvta_generic_to_shared(pointer));
}

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
 *  first appear at the sm_80 floor. */
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

/** One @c mma.sync m16n8k16 step over @c f16 pairs into f32: @p a is a lane's A fragment, @p b
 *  its B pair, and @p tile its D fragment. */
__device__ __forceinline__ std::array<f32_t, 4> cuda_mma_f16_ampere(uint4 a, uint2 b,
                                                                    std::array<f32_t, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, " //
                 "{%0, %1, %2, %3};"
                 : "+f"(tile[0]), "+f"(tile[1]), "+f"(tile[2]), "+f"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** The @c bf16 twin of @ref cuda_mma_f16_ampere. */
__device__ __forceinline__ std::array<f32_t, 4> cuda_mma_bf16_ampere(uint4 a, uint2 b,
                                                                     std::array<f32_t, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, " //
                 "{%0, %1, %2, %3};"
                 : "+f"(tile[0]), "+f"(tile[1]), "+f"(tile[2]), "+f"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** f16 kit - the sm_80 floor. */
struct f16_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_f16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_f16_ampere(a, b, tile);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_bf16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_bf16_ampere(a, b, tile);
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
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k | nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_ || NUMKONG_TARGET_BLACKWELLRTX;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_f16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_f16_ampere(a, b, tile);
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
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k | nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_ || NUMKONG_TARGET_BLACKWELLRTX;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 8>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_bf16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_bf16_ampere(a, b, tile);
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

/** One @c mma.sync m16n8k32 step over signed bytes into s32, laid out as
 *  @ref cuda_mma_f16_ampere's. */
__device__ __forceinline__ std::array<int, 4> cuda_mma_i8_ampere(uint4 a, uint2 b, std::array<int, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, " //
                 "{%0, %1, %2, %3};"
                 : "+r"(tile[0]), "+r"(tile[1]), "+r"(tile[2]), "+r"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** The unsigned twin of @ref cuda_mma_i8_ampere, wrapping like the @c dp4a sums. */
__device__ __forceinline__ std::array<u32_t, 4> cuda_mma_u8_ampere(uint4 a, uint2 b,
                                                                   std::array<u32_t, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, " //
                 "{%0, %1, %2, %3};"
                 : "+r"(tile[0]), "+r"(tile[1]), "+r"(tile[2]), "+r"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** i8 kit - the sm_80 floor. */
struct i8_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_i8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_i8_ampere(a, b, tile);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::u8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_u8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_u8_ampere(a, b, tile);
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
 *  and one generation's kit differs from another's only in the conversion its loops call. */

/** One @c e4m3 byte placed into an f32 by hand, as the Ampere floor has no fp8 conversion:
 *  subnormals scale by 2⁻⁹, and the all-ones magnitude is the format's only NaN. */
__device__ __forceinline__ f32_t cuda_e4m3_to_f32_ampere(u32_t byte) noexcept {
    u32_t const exponent = (byte >> 3) & 0xF, mantissa = byte & 0x7;
    f32_t const magnitude = (byte & 0x7F) == 0x7F ? __int_as_float(0x7FC00000)
                            : exponent ? __int_as_float(static_cast<int>(((exponent + 120u) << 23) | (mantissa << 20)))
                                       : static_cast<f32_t>(mantissa) * 0x1p-9f;
    return byte & 0x80 ? -magnitude : magnitude;
}

/** A packed @c e4m3 pair widened to two floats by hand, at the Ampere floor. */
__device__ __forceinline__ float2 cuda_e4m3x2_to_f32x2_ampere(u32_t packed_pair) noexcept {
    return {cuda_e4m3_to_f32_ampere(packed_pair & 0xFF), cuda_e4m3_to_f32_ampere((packed_pair >> 8) & 0xFF)};
}

/** A packed @c e5m2 pair widened at the Ampere floor: @c e5m2 is exactly the high byte of a half,
 *  so shifting each byte up is the whole conversion, infinities and NaNs included. */
__device__ __forceinline__ float2 cuda_e5m2x2_to_f32x2_ampere(u32_t packed_pair) noexcept {
    u32_t const half_pair = ((packed_pair & 0xFF) << 8) | ((packed_pair & 0xFF00) << 16);
    return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
}

/** A packed @c e3m2 pair widened by hand at the Ampere floor. The OCP fp6 formats sit low in a
 *  byte, with the sign at bit 5 and no infinities or NaN, so the five magnitude bits drop into a
 *  half's exponent and mantissa fields as they stand, the sign moves to bit 15, and one exact
 *  power-of-two product rebiases, subnormals too: two mantissa bits, bias 3 against the half's
 *  15. */
__device__ __forceinline__ float2 cuda_e3m2x2_to_f32x2_ampere(u32_t packed_pair) noexcept {
    u32_t const magnitudes = ((packed_pair & 0x001Fu) | ((packed_pair & 0x1F00u) << 8)) << (10 - 2);
    u32_t const signs = ((packed_pair & 0x0020u) | ((packed_pair & 0x2000u) << 8)) << 10;
    u32_t const unscaled = magnitudes | signs;
    __half2 const scale = __half2half2(__ushort_as_half(static_cast<unsigned short>((15u + 12u) << 10)));
    return __half22float2(__hmul2(*reinterpret_cast<__half2 const*>(&unscaled), scale));
}

/** The @c e2m3 twin of @c cuda_e3m2x2_to_f32x2_ampere: three mantissa bits, bias 1 against the
 *  half's 15. */
__device__ __forceinline__ float2 cuda_e2m3x2_to_f32x2_ampere(u32_t packed_pair) noexcept {
    u32_t const magnitudes = ((packed_pair & 0x001Fu) | ((packed_pair & 0x1F00u) << 8)) << (10 - 3);
    u32_t const signs = ((packed_pair & 0x0020u) | ((packed_pair & 0x2000u) << 8)) << 10;
    u32_t const unscaled = magnitudes | signs;
    __half2 const scale = __half2half2(__ushort_as_half(static_cast<unsigned short>((15u + 14u) << 10)));
    return __half22float2(__hmul2(*reinterpret_cast<__half2 const*>(&unscaled), scale));
}

/** Sixteen @c e5m2 products of a fragment pair, each byte pair widened into f32 by shifts. */
__device__ __forceinline__ f32_t cuda_dot_e5m2x16_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = cuda_e5m2x2_to_f32x2_ampere(words_a[word] >> (16 * pair));
            float2 const values_b = cuda_e5m2x2_to_f32x2_ampere(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** An @c e5m2 fragment's sum & sum of squares, widened into f32 by shifts. */
__device__ __forceinline__ float2 cuda_moments_e5m2x16_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e5m2x2_to_f32x2_ampere(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c e5m2 fragment's sixteen values. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e5m2x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e5m2x2_to_f32x2_ampere(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e4m3 twin of @ref cuda_dot_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ f32_t cuda_dot_e4m3x16_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = cuda_e4m3x2_to_f32x2_ampere(words_a[word] >> (16 * pair));
            float2 const values_b = cuda_e4m3x2_to_f32x2_ampere(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** The @c e4m3 twin of @ref cuda_moments_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ float2 cuda_moments_e4m3x16_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e4m3x2_to_f32x2_ampere(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e4m3 twin of @ref cuda_weighted_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e4m3x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e4m3x2_to_f32x2_ampere(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e3m2 twin of @ref cuda_dot_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ f32_t cuda_dot_e3m2x16_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = cuda_e3m2x2_to_f32x2_ampere(words_a[word] >> (16 * pair));
            float2 const values_b = cuda_e3m2x2_to_f32x2_ampere(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** The @c e3m2 twin of @ref cuda_moments_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ float2 cuda_moments_e3m2x16_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e3m2x2_to_f32x2_ampere(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e3m2 twin of @ref cuda_weighted_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e3m2x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e3m2x2_to_f32x2_ampere(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e2m3 twin of @ref cuda_dot_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ f32_t cuda_dot_e2m3x16_ampere(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = cuda_e2m3x2_to_f32x2_ampere(words_a[word] >> (16 * pair));
            float2 const values_b = cuda_e2m3x2_to_f32x2_ampere(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** The @c e2m3 twin of @ref cuda_moments_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ float2 cuda_moments_e2m3x16_ampere(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e2m3x2_to_f32x2_ampere(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e2m3 twin of @ref cuda_weighted_e5m2x16_ampere, widening by hand. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e2m3x16_ampere( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e2m3x2_to_f32x2_ampere(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** e5m2 kit - the sm_80 floor, widening by shifts. */
struct e5m2_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e5m2x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e5m2x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e5m2x16_ampere(a, weight, sums);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e4m3x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e4m3x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e4m3x16_ampere(a, weight, sums);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e3m2x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e3m2x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e3m2x16_ampere(a, weight, sums);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e2m3x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e2m3x16_ampere(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e2m3x16_ampere(a, weight, sums);
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

/** A packed @c e4m3 pair widened to two halves in one @c cvt.rn.f16x2.e4m3x2, on Ada and every tier
 *  since. */
__device__ __forceinline__ u32_t cuda_e4m3x2_to_f16x2_ada(u32_t packed_pair) noexcept {
    u32_t half_pair;
    asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
    return half_pair;
}

/** A packed @c e4m3 pair widened to two floats through @ref cuda_e4m3x2_to_f16x2_ada. */
__device__ __forceinline__ float2 cuda_e4m3x2_to_f32x2_ada(u32_t packed_pair) noexcept {
    u32_t const half_pair = cuda_e4m3x2_to_f16x2_ada(packed_pair);
    return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
}

/** The @c e5m2 twin of @ref cuda_e4m3x2_to_f16x2_ada, through @c cvt.rn.f16x2.e5m2x2. */
__device__ __forceinline__ u32_t cuda_e5m2x2_to_f16x2_ada(u32_t packed_pair) noexcept {
    u32_t half_pair;
    asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
    return half_pair;
}

/** A packed @c e5m2 pair widened to two floats through @ref cuda_e5m2x2_to_f16x2_ada. */
__device__ __forceinline__ float2 cuda_e5m2x2_to_f32x2_ada(u32_t packed_pair) noexcept {
    u32_t const half_pair = cuda_e5m2x2_to_f16x2_ada(packed_pair);
    return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
}

/** Sixteen @c e5m2 products of a fragment pair, each byte pair widened into f32 through the Ada
 *  conversion. */
__device__ __forceinline__ f32_t cuda_dot_e5m2x16_ada(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = cuda_e5m2x2_to_f32x2_ada(words_a[word] >> (16 * pair));
            float2 const values_b = cuda_e5m2x2_to_f32x2_ada(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** An @c e5m2 fragment's sum & sum of squares, widened into f32 through the Ada conversion. */
__device__ __forceinline__ float2 cuda_moments_e5m2x16_ada(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e5m2x2_to_f32x2_ada(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c e5m2 fragment's sixteen values, widened
 *  through the Ada conversion. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e5m2x16_ada( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e5m2x2_to_f32x2_ada(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e4m3 twin of @ref cuda_dot_e5m2x16_ada. */
__device__ __forceinline__ f32_t cuda_dot_e4m3x16_ada(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values_a = cuda_e4m3x2_to_f32x2_ada(words_a[word] >> (16 * pair));
            float2 const values_b = cuda_e4m3x2_to_f32x2_ada(words_b[word] >> (16 * pair));
            sum += values_a.x * values_b.x + values_a.y * values_b.y;
        }
    return sum;
}

/** The @c e4m3 twin of @ref cuda_moments_e5m2x16_ada. */
__device__ __forceinline__ float2 cuda_moments_e4m3x16_ada(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e4m3x2_to_f32x2_ada(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e4m3 twin of @ref cuda_weighted_e5m2x16_ada. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e4m3x16_ada( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e4m3x2_to_f32x2_ada(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** One @c mma.sync m16n8k32 step over @c e5m2 bytes into f32, laid out as
 *  @ref cuda_mma_f16_ampere's. */
__device__ __forceinline__ std::array<f32_t, 4> cuda_mma_e5m2_ada(uint4 a, uint2 b,
                                                                  std::array<f32_t, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, " //
                 "{%0, %1, %2, %3};"
                 : "+f"(tile[0]), "+f"(tile[1]), "+f"(tile[2]), "+f"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** The @c e4m3 twin of @ref cuda_mma_e5m2_ada. */
__device__ __forceinline__ std::array<f32_t, 4> cuda_mma_e4m3_ada(uint4 a, uint2 b,
                                                                  std::array<f32_t, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, " //
                 "{%0, %1, %2, %3};"
                 : "+f"(tile[0]), "+f"(tile[1]), "+f"(tile[2]), "+f"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** e5m2 kit - Ada, widening through the conversion. */
struct e5m2_ada_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ada_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_ADA_;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e5m2x16_ada(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_e5m2_ada(a, b, tile);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e5m2x16_ada(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e5m2x16_ada(a, weight, sums);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_ADA_;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e4m3x16_ada(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_e4m3_ada(a, b, tile);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e4m3x16_ada(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e4m3x16_ada(a, weight, sums);
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

/** A packed @c e3m2 pair widened to two halves in one @c cvt.rn.f16x2.e3m2x2, which only
 *  Blackwell's family-specific code encodes. */
__device__ __forceinline__ u32_t cuda_e3m2x2_to_f16x2_blackwell(u32_t packed_pair) noexcept {
    u32_t half_pair;
    asm("{cvt.rn.f16x2.e3m2x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
    return half_pair;
}

/** A packed @c e3m2 pair widened to two floats through @ref cuda_e3m2x2_to_f16x2_blackwell. */
__device__ __forceinline__ float2 cuda_e3m2x2_to_f32x2_blackwell(u32_t packed_pair) noexcept {
    u32_t const half_pair = cuda_e3m2x2_to_f16x2_blackwell(packed_pair);
    return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
}

/** The @c e2m3 twin of @ref cuda_e3m2x2_to_f16x2_blackwell, through @c cvt.rn.f16x2.e2m3x2. */
__device__ __forceinline__ u32_t cuda_e2m3x2_to_f16x2_blackwell(u32_t packed_pair) noexcept {
    u32_t half_pair;
    asm("{cvt.rn.f16x2.e2m3x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
    return half_pair;
}

/** A packed @c e2m3 pair widened to two floats through @ref cuda_e2m3x2_to_f16x2_blackwell. */
__device__ __forceinline__ float2 cuda_e2m3x2_to_f32x2_blackwell(u32_t packed_pair) noexcept {
    u32_t const half_pair = cuda_e2m3x2_to_f16x2_blackwell(packed_pair);
    return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
}

/** Sixteen @c e5m2 products of a fragment pair, each byte pair converted into halves and multiplied
 *  straight into the f32 sum, so the dot never widens to f32 - the norms still do. */
__device__ __forceinline__ f32_t cuda_dot_e5m2x16_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair)
            sum = cuda_fma_f16x2_blackwell(cuda_e5m2x2_to_f16x2_ada(words_a[word] >> (16 * pair)),
                                           cuda_e5m2x2_to_f16x2_ada(words_b[word] >> (16 * pair)), sum);
    return sum;
}

/** The @c e4m3 twin of @ref cuda_dot_e5m2x16_blackwell. */
__device__ __forceinline__ f32_t cuda_dot_e4m3x16_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair)
            sum = cuda_fma_f16x2_blackwell(cuda_e4m3x2_to_f16x2_ada(words_a[word] >> (16 * pair)),
                                           cuda_e4m3x2_to_f16x2_ada(words_b[word] >> (16 * pair)), sum);
    return sum;
}

/** The @c e3m2 twin of @ref cuda_dot_e5m2x16_blackwell, through the fp6 conversion. */
__device__ __forceinline__ f32_t cuda_dot_e3m2x16_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair)
            sum = cuda_fma_f16x2_blackwell(cuda_e3m2x2_to_f16x2_blackwell(words_a[word] >> (16 * pair)),
                                           cuda_e3m2x2_to_f16x2_blackwell(words_b[word] >> (16 * pair)), sum);
    return sum;
}

/** An @c e3m2 fragment's sum & sum of squares, widened into f32 through the fp6 conversion. */
__device__ __forceinline__ float2 cuda_moments_e3m2x16_blackwell(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e3m2x2_to_f32x2_blackwell(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** Returns @p sums plus @p weight times each of an @c e3m2 fragment's sixteen values, widened
 *  through the fp6 conversion. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e3m2x16_blackwell( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e3m2x2_to_f32x2_blackwell(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** The @c e2m3 twin of @ref cuda_dot_e3m2x16_blackwell. */
__device__ __forceinline__ f32_t cuda_dot_e2m3x16_blackwell(uint4 a, uint4 b, f32_t sum) noexcept {
    u32_t const words_a[4] = {a.x, a.y, a.z, a.w}, words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair)
            sum = cuda_fma_f16x2_blackwell(cuda_e2m3x2_to_f16x2_blackwell(words_a[word] >> (16 * pair)),
                                           cuda_e2m3x2_to_f16x2_blackwell(words_b[word] >> (16 * pair)), sum);
    return sum;
}

/** The @c e2m3 twin of @ref cuda_moments_e3m2x16_blackwell. */
__device__ __forceinline__ float2 cuda_moments_e2m3x16_blackwell(uint4 a, float2 moments) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e2m3x2_to_f32x2_blackwell(words[word] >> (16 * pair));
            moments.x += values.x + values.y;
            moments.y += values.x * values.x + values.y * values.y;
        }
    return moments;
}

/** The @c e2m3 twin of @ref cuda_weighted_e3m2x16_blackwell. */
__device__ __forceinline__ std::array<f32_t, 16> cuda_weighted_e2m3x16_blackwell( //
    uint4 a, f32_t weight, std::array<f32_t, 16> sums) noexcept {
    u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (u32_t word = 0; word != 4; ++word)
#pragma unroll
        for (u32_t pair = 0; pair != 2; ++pair) {
            float2 const values = cuda_e2m3x2_to_f32x2_blackwell(words[word] >> (16 * pair));
            sums[4 * word + 2 * pair] += weight * values.x;
            sums[4 * word + 2 * pair + 1] += weight * values.y;
        }
    return sums;
}

/** e5m2 kit - Blackwell, converted halves through @c fma.rn.f32.f16. */
struct e5m2_blackwell_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k | nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_ || NUMKONG_TARGET_BLACKWELLRTX;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e5m2x16_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_e5m2_ada(a, b, tile);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e5m2x16_ada(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e5m2x16_ada(a, weight, sums);
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
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k | nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_ || NUMKONG_TARGET_BLACKWELLRTX;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e4m3x16_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_e4m3_ada(a, b, tile);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e4m3x16_ada(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e4m3x16_ada(a, weight, sums);
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
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k | nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_ || NUMKONG_TARGET_BLACKWELLRTX;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e3m2x16_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e3m2x16_blackwell(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e3m2x16_blackwell(a, weight, sums);
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
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k | nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_ || NUMKONG_TARGET_BLACKWELLRTX;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using weighted_t = std::array<f32_t, 16>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return cuda_dot_e2m3x16_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return cuda_moments_e2m3x16_blackwell(a, moments);
    }
    static __device__ __forceinline__ weighted_t accumulate_weighted(fragment_t a, f32_t weight,
                                                                     weighted_t sums) noexcept {
        return cuda_weighted_e2m3x16_blackwell(a, weight, sums);
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
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
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
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

/** One @c mma.sync m16n8k256 step over packed bits into s32, each product the popcount of an
 *  intersection, laid out as @ref cuda_mma_f16_ampere's. */
__device__ __forceinline__ std::array<u32_t, 4> cuda_mma_u1_ampere(uint4 a, uint2 b,
                                                                   std::array<u32_t, 4> tile) noexcept {
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0, %1, %2, %3}, {%4, %5, %6, %7}, " //
                 "{%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(tile[0]), "+r"(tile[1]), "+r"(tile[2]), "+r"(tile[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
    return tile;
}

/** u1x8 kit - the sm_80 floor: the "dot" is the intersection popcount, and the norms store each
 *  row's set-bit count, the currency of Jaccard's union, mirrored into both fields. */
struct u1x8_ampere_quads_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using scalar_t = nk::u1x8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using weighted_t = std::array<f32_t, 128>;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return sum + cuda_intersect_u1x128_ampere(a, b);
    }
    static __device__ __forceinline__ std::array<dot_t, 4> accumulate_tile(fragment_t a, uint2 b,
                                                                           std::array<dot_t, 4> tile) noexcept {
        return cuda_mma_u1_ampere(a, b, tile);
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
 *  @brief Scores a walk's batch through the kit's @c mma.sync step, fed straight from its loads.
 *
 *  A lane's 16-byte chunk is one tile's A fragment as loaded: words 0 and 2 come from row g, words
 *  1 and 3 in row g + 8, and the B columns carry the query's matching words by column parity, so
 *  the query takes the rows' permutation of the dimensions and row g's candidate scores D[g][even]
 *  + D[g + 8][odd]. Each lane loads 32 bytes of a row a step, so @p lanes_per_row_ lanes make one
 *  request of 32 × @p lanes_per_row_ bytes per row: under 4, every group of 4 lanes owns one
 *  candidate; under 8, a candidate's pieces split over two groups and its word halves over their D
 *  entries, which a reduce-scatter folds once per pass of 16 candidates. Rows past a short batch
 *  re-read its first row, and rows split into whole 32-byte pieces.
 */
template <vram_quads_kit kit_, u32_t lanes_per_row_>
__device__ __forceinline__ vram_scored_node<node_id_t> cuda_score_tiles( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, vram_gathered<kit_, cuda_warp_t>& gathered, u32_t gathered_count,
    u32_t lane) noexcept {

    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;
    using tile_t = std::array<dot_t, 4>;
    // Aligned to its own size, so the compiler issues each lane's piece as one request.
    struct alignas(2 * sizeof(fragment_t)) piece_t {
        fragment_t low, high;
    };
    constexpr u32_t rows_per_load_k = cuda_warp_t::width_k / lanes_per_row_;
    // A pass keeps four pieces in flight per lane, one per tile, the window the scalar batch keeps.
    constexpr u32_t passes_k = lanes_per_row_ / 4, rows_per_pass_k = 4 * rows_per_load_k;
    u32_t const piece = lane % lanes_per_row_, slot = lane / lanes_per_row_;
    u32_t const row_pieces = storage.geometry.row_fragments / 2;
    bool const odd_column = slot & 1;
    bool const live = lane < gathered_count;
    typename kit_::norms_t const target_state = live ? storage.norm_for(gathered.ids[lane]) : typename kit_::norms_t {};

    dot_t pass_dots[passes_k] = {};
#pragma unroll
    for (u32_t pass = 0; pass != passes_k; ++pass) {
        // Warp-uniform, as most of a walk's batches fill only the first pass.
        if (pass * rows_per_pass_k >= gathered_count) break;
        piece_t const* rows[4];
#pragma unroll
        for (u32_t load = 0; load != 4; ++load) {
            u32_t const row = (4 * pass + load) * rows_per_load_k + slot;
            rows[load] = reinterpret_cast<piece_t const*>(
                storage.vector_span_for(gathered.ids[row < gathered_count ? row : 0]));
        }
        tile_t tiles[4] = {};
#pragma unroll 1
        for (u32_t first_piece = 0; first_piece < row_pieces; first_piece += lanes_per_row_) {
            // A lane past the row's end re-reads the row's first piece against a zero query.
            u32_t const index = first_piece + piece;
            bool const in_row = index < row_pieces;
            u32_t const loaded_index = in_row ? index : 0;
            fragment_t const query_low = in_row ? query_span[2 * index] : fragment_t {};
            fragment_t const query_high = in_row ? query_span[2 * index + 1] : fragment_t {};
            uint2 const pairs_low = odd_column ? uint2 {query_low.y, query_low.w} : uint2 {query_low.x, query_low.z};
            uint2 const pairs_high = odd_column ? uint2 {query_high.y, query_high.w}
                                                : uint2 {query_high.x, query_high.z};
            piece_t loaded[4];
#pragma unroll
            for (u32_t load = 0; load != 4; ++load) loaded[load] = rows[load][loaded_index];
#pragma unroll
            for (u32_t load = 0; load != 4; ++load) {
                tiles[load] = kit_::accumulate_tile(loaded[load].low, pairs_low, tiles[load]);
                tiles[load] = kit_::accumulate_tile(loaded[load].high, pairs_high, tiles[load]);
            }
        }
        if constexpr (lanes_per_row_ == 4) {
            // Lane (g, c) holds candidates 8t + g and keeps candidate 8c + g.
            dot_t sums[4];
#pragma unroll
            for (u32_t load = 0; load != 4; ++load) sums[load] = tiles[load][0] + tiles[load][3];
            dot_t const low = lane & 1 ? sums[1] : sums[0], high = lane & 1 ? sums[3] : sums[2];
            pass_dots[pass] = lane & 2 ? high : low;
        }
        else {
            // One of a lane's D entries meets its rows' piece half and word half with the matching
            // query column.
            dot_t partials[4];
#pragma unroll
            for (u32_t load = 0; load != 4; ++load) {
                dot_t const row_low = piece >= 4 ? tiles[load][1] : tiles[load][0];
                dot_t const row_high = piece >= 4 ? tiles[load][3] : tiles[load][2];
                partials[load] = lane & 1 ? row_high : row_low;
            }
            bool const odd = lane & 1, upper = lane & 4;
            dot_t halves[2];
#pragma unroll
            for (u32_t half = 0; half != 2; ++half)
                halves[half] = (odd ? partials[half + 2] : partials[half]) +
                               cuda_warp_t::shuffle_xor(odd ? partials[half] : partials[half + 2], 1u);
            pass_dots[pass] = (upper ? halves[1] : halves[0]) +
                              cuda_warp_t::shuffle_xor(upper ? halves[0] : halves[1], 4u);
        }
    }
    // Lane l of the 4-lane form holds candidate 8 (l % 4) + l / 4; of the 8-lane form, candidate
    // 16p + 4i + l / 8 for p = (l >> 1) & 1 and i = ((l >> 2) & 1) + 2 (l & 1).
    dot_t const held = lane & 2 ? pass_dots[passes_k - 1] : pass_dots[0];
    u32_t const holder = lanes_per_row_ == 4
                             ? 4 * (lane & 7) + (lane >> 3)
                             : 8 * (lane & 3) + 4 * ((lane >> 2) & 1) + 2 * ((lane >> 4) & 1) + ((lane >> 3) & 1);
    dot_t const dot = cuda_warp_t::shuffle(held, holder);
    cuda_warp_t::sync();
    if (!live) return vram_scored_node<node_id_t>::sentinel();
    gathered.dots[lane] = dot;
    vram_scored_node<node_id_t> const scored = vram_scored_node<node_id_t>::make(
        kit_::distance(storage.geometry.kernel, dot, query_state, target_state), gathered.ids[lane]);
    gathered.scored[lane] = scored;
    return scored;
}

/** A walk's batch under a kit without a tensor-core step: its scalar dots, @p fragments_per_load_
 *  adjacent fragments a load. */
template <vram_quads_kit kit_, u32_t resident_blocks_, u32_t fragments_per_load_>
__device__ __forceinline__ vram_scored_node<node_id_t> cuda_score_batch( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, vram_gathered<kit_, cuda_warp_t>& gathered, u32_t gathered_count,
    u32_t lane) noexcept {
    return vram_score_batch<kit_, cuda_warp_t, resident_blocks_, fragments_per_load_>( //
        storage, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
}

/** A walk's batch under a kit with a tensor-core step: tiles of 256-byte requests from 256-byte
 *  rows up and of 128-byte ones under them, and the scalar dots for rows of an odd fragment count,
 *  which split into no whole 32-byte pieces. Binary rows keep their scalar popcounts, which score
 *  one query's batch faster than the bit tile. */
template <vram_quads_kit kit_, u32_t resident_blocks_, u32_t fragments_per_load_>
    requires requires(typename kit_::fragment_t a, uint2 b, std::array<typename kit_::dot_t, 4> tile) {
        kit_::accumulate_tile(a, b, tile);
    }
__device__ __forceinline__ vram_scored_node<node_id_t> cuda_score_batch( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, vram_gathered<kit_, cuda_warp_t>& gathered, u32_t gathered_count,
    u32_t lane) noexcept {
    if constexpr (!std::is_same_v<typename kit_::scalar_t, nk::u1x8_t>)
        if (storage.geometry.row_fragments % 2 == 0)
            return storage.geometry.row_fragments >= 16
                       ? cuda_score_tiles<kit_, 8>(storage, query_span, query_state, gathered, gathered_count, lane)
                       : cuda_score_tiles<kit_, 4>(storage, query_span, query_state, gathered, gathered_count, lane);
    return vram_score_batch<kit_, cuda_warp_t, resident_blocks_, fragments_per_load_>( //
        storage, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
}

/** Rows one @c mma.sync tile spans: the block the tile greedy walks the sorted candidates in. */
inline constexpr u32_t cuda_tile_rows_k = 16;

/**
 *  @brief Two m16n8 tiles of dots between the cache rows at @p top and @p bottom, a lane's rows
 *      @b g and @b g+8, and the cache rows at @p columns, the lane's column @b g of each tile.
 *
 *  The scan's operand permutation: lane (g, q) loads fragment 4·span + q of every row it holds,
 *  words x and y feed one step and z and w the next, so both sides take one permutation of the
 *  dimensions, and a fragment past the row reads as zeros. Every tile-capable kit's step spans 32
 *  bytes of depth, so one permutation serves them all.
 */
template <vram_quads_kit kit_, typename storage_>
__device__ __forceinline__ std::array<std::array<typename kit_::dot_t, 4>, 2> cuda_block_tiles( //
    storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache, u32_t top, u32_t bottom,
    u32_t const (&columns)[2], u32_t lane) noexcept {

    using fragment_t = typename kit_::fragment_t;
    using tile_t = std::array<typename kit_::dot_t, 4>;
    u32_t const quarter = lane % 4, spans = divide_round_up(storage.geometry.row_fragments, 4u);
    fragment_t const* const top_row = cache.staged_row(storage, top);
    fragment_t const* const bottom_row = cache.staged_row(storage, bottom);
    fragment_t const* const column_rows[2] = {cache.staged_row(storage, columns[0]),
                                              cache.staged_row(storage, columns[1])};
    std::array<tile_t, 2> tiles = {};
#pragma unroll 1
    for (u32_t span = 0; span != spans; ++span) {
        u32_t const fragment = span * 4 + quarter;
        bool const in_row = fragment < storage.geometry.row_fragments;
        fragment_t const upper = in_row ? top_row[fragment] : fragment_t {};
        fragment_t const lower = in_row ? bottom_row[fragment] : fragment_t {};
        fragment_t const first {upper.x, lower.x, upper.y, lower.y}, second {upper.z, lower.z, upper.w, lower.w};
#pragma unroll
        for (u32_t tile = 0; tile != 2; ++tile) {
            fragment_t const piece = in_row ? column_rows[tile][fragment] : fragment_t {};
            tiles[tile] = kit_::accumulate_tile(first, uint2 {piece.x, piece.y}, tiles[tile]);
            tiles[tile] = kit_::accumulate_tile(second, uint2 {piece.z, piece.w}, tiles[tile]);
        }
    }
    return tiles;
}

/**
 *  @brief The greedy diversity prefix of @ref unum::usearch::dense::vram_keep_diverse on the kit's
 *      tensor-core tiles.
 *
 *  The sorted candidates run in blocks of @ref unum::usearch::dense::cuda_tile_rows_k. Each block
 *  first scores against the kept ones, sixteen columns a pass, each row falling once any of its
 *  lanes sees a nearer kept one, then against itself, and keeps its rows in order by one vote a
 *  row. Every pair is scored once and stays in registers, a row's kept set is the one the serial
 *  greedy would test, and the zero rule and the stop at a full row cut where it would: the verdicts
 *  are the serial greedy's for the distances the tiles give.
 */
template <vram_quads_kit kit_, typename schedule_, typename storage_>
    requires requires(typename kit_::fragment_t a, uint2 b, std::array<typename kit_::dot_t, 4> tile) {
        kit_::accumulate_tile(a, b, tile);
    }
__device__ __forceinline__ u32_t cuda_keep_diverse_tiles( //
    storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_sorted_buffer<cuda_warp_t, node_id_t> const& sorted_candidates, u32_t position_mask, u32_t candidates_count,
    vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
    vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using norms_t = typename kit_::norms_t;
    using tiles_t = std::array<std::array<typename kit_::dot_t, 4>, 2>;
    constexpr u32_t half_k = cuda_tile_rows_k / 2;
    u32_t const group = lane / 4, quarter = lane % 4;
    u32_t* const selected_ids = selection.selected_ids;
    f32_t const* const to_query = selection.candidate_distances;
    u32_t selected_count = 0, staged_through = 0;
#pragma unroll 1
    for (u32_t block_first = 0; block_first < candidates_count && selected_count < storage.connectivity;
         block_first += cuda_tile_rows_k) {
        u32_t const block_count = min_of(cuda_tile_rows_k, candidates_count - block_first);
        // Lane r keeps block row r's cache row, modulo the block; rows past a short block re-read
        // its first.
        u32_t const block_row = lane % cuda_tile_rows_k < block_count ? lane % cuda_tile_rows_k : 0;
        u32_t const position = sorted_candidates.at(block_first + block_row).id() & position_mask;
        // Ties visit rows out of staging order, so the wait covers every row the block names.
        u32_t required_rows = position + 1;
        for (u32_t offset = half_k; offset; offset >>= 1)
            required_rows = max_of(required_rows, cuda_warp_t::shuffle_xor(required_rows, offset));
        staged_through = max_of(staged_through, required_rows);
        schedule.await_rows(staged_through);
        u32_t const top = cuda_warp_t::shuffle(position, group);
        u32_t const bottom = cuda_warp_t::shuffle(position, group + half_k);
        norms_t const top_norms = candidate_norms[top], bottom_norms = candidate_norms[bottom];
        u32_t const kept_before = selected_count;

        // Against the kept ones: a row falls once a lane of it sees a kept one nearer than the row.
        bool top_rejected = false, bottom_rejected = false;
#pragma unroll 1
        for (u32_t first_column = 0; first_column < kept_before; first_column += cuda_tile_rows_k) {
            u32_t const columns[2] = {selected_ids[min_of(first_column + group, kept_before - 1)],
                                      selected_ids[min_of(first_column + half_k + group, kept_before - 1)]};
            tiles_t const tiles = cuda_block_tiles<kit_>(storage, cache, top, bottom, columns, lane);
#pragma unroll
            for (u32_t tile = 0; tile != 2; ++tile)
#pragma unroll
                for (u32_t column = 0; column != 2; ++column) {
                    u32_t const kept = first_column + tile * half_k + quarter * 2 + column;
                    if (kept >= kept_before) continue;
                    norms_t const kept_norms = candidate_norms[selected_ids[kept]];
                    f32_t const from_top = kit_::distance(storage.geometry.kernel, tiles[tile][column], top_norms,
                                                          kept_norms);
                    f32_t const from_bottom = kit_::distance(storage.geometry.kernel, tiles[tile][2 + column],
                                                             bottom_norms, kept_norms);
                    top_rejected |= !(from_top >= to_query[top]);
                    bottom_rejected |= !(from_bottom >= to_query[bottom]);
                }
        }
        // Lanes 4g to 4g + 3 hold rows g and g + 8.
        typename cuda_warp_t::mask_t const top_lanes = cuda_warp_t::ballot(top_rejected);
        typename cuda_warp_t::mask_t const bottom_lanes = cuda_warp_t::ballot(bottom_rejected);
        u32_t rejected = 0;
        for (u32_t row = 0; row != half_k; ++row)
            rejected |= ((top_lanes >> (4 * row)) & 0xFu ? 1u : 0u) << row |
                        ((bottom_lanes >> (4 * row)) & 0xFu ? 1u : 0u) << (row + half_k);

        // Against itself: tile t's column g is block row 8t + g, the lane's own rows.
        u32_t const own_columns[2] = {top, bottom};
        tiles_t const own = cuda_block_tiles<kit_>(storage, cache, top, bottom, own_columns, lane);
        // Entry e is block column 8 (e / 2) + 2q + e % 2, from the lane's rows g and g + 8.
        f32_t between[2][4];
#pragma unroll
        for (u32_t entry = 0; entry != 4; ++entry) {
            u32_t const column = (entry >> 1) * half_k + quarter * 2 + (entry & 1u);
            norms_t const column_norms = candidate_norms[cuda_warp_t::shuffle(position, column)];
            between[0][entry] = kit_::distance(storage.geometry.kernel, own[entry >> 1][entry & 1u], top_norms,
                                               column_norms);
            between[1][entry] = kit_::distance(storage.geometry.kernel, own[entry >> 1][2 + (entry & 1u)], bottom_norms,
                                               column_norms);
        }
        u32_t kept_rows = 0;
#pragma unroll 1
        for (u32_t row = 0; row != block_count && selected_count < storage.connectivity; ++row) {
            u32_t const candidate = cuda_warp_t::shuffle(position, row);
            f32_t const distance = to_query[candidate];
            bool clash = false;
            if (group == row % half_k)
#pragma unroll
                for (u32_t entry = 0; entry != 4; ++entry) {
                    u32_t const column = (entry >> 1) * half_k + quarter * 2 + (entry & 1u);
                    f32_t const to_kept = row < half_k ? between[0][entry] : between[1][entry];
                    clash |= column < row && ((kept_rows >> column) & 1u) && !(to_kept >= distance);
                }
            // A second duplicate of the row would only repeat the first one's links.
            bool const diverse = !((rejected >> row) & 1u) && !(distance == 0 && kept_zero) && !cuda_warp_t::any(clash);
            if (diverse) {
                kept_zero |= distance == 0;
                kept_rows |= 1u << row;
                if (lane == 0) selected_ids[selected_count] = candidate;
                ++selected_count;
                cuda_warp_t::sync();
            }
        }
        telemetry.scored += block_count * kept_before + block_count * (block_count - 1) / 2;
    }
    return selected_count;
}

/** The greedy diversity prefix: on the kit's tensor-core tiles where it has a step, the serial
 *  greedy otherwise. No argument deduces the kit, so a branch picks the body, not an overload. */
template <vram_quads_kit kit_, typename schedule_, typename storage_>
__device__ __forceinline__ u32_t cuda_keep_diverse( //
    storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_sorted_buffer<cuda_warp_t, node_id_t> const& sorted_candidates, u32_t position_mask, u32_t candidates_count,
    vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
    vram_telemetry_t& telemetry, u32_t lane) noexcept {
    if constexpr (requires(typename kit_::fragment_t a, uint2 b, std::array<typename kit_::dot_t, 4> tile) {
                      kit_::accumulate_tile(a, b, tile);
                  })
        return cuda_keep_diverse_tiles<kit_>(storage, cache, schedule, sorted_candidates, position_mask,
                                             candidates_count, selection, candidate_norms, kept_zero, telemetry, lane);
    else
        return vram_keep_diverse<kit_>(storage, cache, schedule, sorted_candidates, position_mask, candidates_count,
                                       selection, candidate_norms, kept_zero, telemetry, lane);
}

/**
 *  @brief The Ampere and Ada staging: every lane queues register-free @c cp.async copies of the
 *      fragments it strides over, one commit group per stage, so a consumer walking the rows in
 *      order starts on the first stage while the later ones copy in.
 *
 *  A @c cp.async completes only for the lane that queued it, so every await ends in a warp barrier.
 */
struct ampere_cp_async_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    using warp_t = cuda_warp_t;
    inline static constexpr u32_t stages_k = 2;
    inline static constexpr u32_t fragments_per_load_k = 1;

    /** Nothing shared: the commit groups live in each lane's own pipeline. */
    struct shared_t {};

    /** Rows per stage of the latest staging, and the leading rows copied in for every lane. */
    u32_t rows_per_stage_ = 0, arrived_rows_ = 0;

    __device__ __forceinline__ ampere_cp_async_schedule_t(shared_t&, u32_t) noexcept {}

    /** Scores a walk's batch as @ref cuda_score_batch does at this schedule's load width. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return cuda_score_batch<kit_, resident_blocks_, fragments_per_load_k>( //
            storage, query_span, query_state, gathered, gathered_count, lane);
    }

    /** The greedy diversity prefix as @ref cuda_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return cuda_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        rows_per_stage_ = divide_round_up(rows, stages_k), arrived_rows_ = 0;
        if (cache.ids_) {
            arrived_rows_ = rows;
            return;
        }
        // Every stage commits, empty or not, so the groups a stage waits behind are a constant.
        for (u32_t stage = 0; stage != stages_k; ++stage) {
            u32_t const end_word = min_of((stage + 1) * rows_per_stage_, rows) * storage.geometry.row_fragments;
            for (u32_t word = min_of(stage * rows_per_stage_, rows) * storage.geometry.row_fragments + lane;
                 word < end_word; word += cuda_warp_t::width_k) {
                u32_t const row = storage.geometry.row_of_word(word),
                            fragment = storage.geometry.fragment_of_word(word, row);
                __pipeline_memcpy_async(&cache.at(row, fragment), storage.vector_span_for(ids[row]) + fragment,
                                        sizeof(typename storage_::fragment_t));
            }
            __pipeline_commit();
        }
    }

    __device__ __forceinline__ void await_rows(u32_t rows) noexcept {
        if (rows <= arrived_rows_) return;
        u32_t const stages = divide_round_up(rows, rows_per_stage_);
        wait_pending_(stages_k - stages);
        arrived_rows_ = stages * rows_per_stage_;
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

/*  The Hopper and Blackwell staging: the Tensor Memory Accelerator copies every row whole in one
 *  @c cp.async.bulk, and each stage's bytes complete on that stage's @c mbarrier, so the copies
 *  hold no registers and every lane waits on the barrier, not on its own copies. A row is one
 *  contiguous span of whole 16-byte fragments, so no tensor map is needed and no row length is
 *  capped. Each barrier advances one phase per staging that fills its stage. The schedules differ
 *  in how many adjacent fragments a walking lane loads at once, and the datacenter Blackwell one
 *  scores its walks' batches from rows a tensor map gathers four at a time. */

/** Arms @p stages barriers from @p barriers on, one arrival each, from the warp's first lane. */
__device__ __forceinline__ void cuda_tma_initialize_barriers(u64_t* barriers, u32_t stages, u32_t lane) noexcept {
    if (lane == 0) {
        for (u32_t stage = 0; stage != stages; ++stage)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(cuda_shared_address(barriers + stage)), "r"(1u)
                         : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncwarp();
}

/** Queues one @c cp.async.bulk per row an ID span names into the cache's leading rows, each
 *  completing on the barrier of its stage of @p rows_per_stage rows. */
template <typename storage_, typename cache_>
__device__ __forceinline__ void cuda_tma_copy_rows(u64_t* barriers, u32_t rows_per_stage, storage_ const& storage,
                                                   cache_ cache, u32_t const* ids, u32_t rows, u32_t lane) noexcept {
    u32_t const row_bytes = storage.geometry.row_fragments * sizeof(typename storage_::fragment_t);
    // The cache rows' earlier generic reads must precede the copies' asynchronous writes.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (lane == 0)
        for (u32_t first_row = 0; first_row < rows; first_row += rows_per_stage) {
            u32_t const barrier = cuda_shared_address(barriers + first_row / rows_per_stage);
            u32_t const stage_bytes = (min_of(first_row + rows_per_stage, rows) - first_row) * row_bytes;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(stage_bytes)
                         : "memory");
        }
    __syncwarp();
    for (u32_t row = lane; row < rows; row += cuda_warp_t::width_k) {
        u32_t const destination = cuda_shared_address(cache.row(row));
        u32_t const barrier = cuda_shared_address(barriers + row / rows_per_stage);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" //
                     ::"r"(destination),
                     "l"(storage.vector_span_for(ids[row])), "r"(row_bytes), "r"(barrier)
                     : "memory");
    }
}

/** Spins until @p barrier completes the phase of parity @p parity. */
__device__ __forceinline__ void cuda_tma_wait_barrier(u64_t* barrier, u32_t parity) noexcept {
    u32_t phase_ready;
    do {
        asm volatile("{\n.reg .pred ready;\n"                                      //
                     "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2;\n" //
                     "selp.u32 %0, 1, 0, ready;\n}\n"
                     : "=r"(phase_ready)
                     : "r"(cuda_shared_address(barrier)), "r"(parity)
                     : "memory");
    } while (!phase_ready);
}

/** One warp's gathered rows in static shared memory: two stages of four-row windows, each staging
 *  on its own barrier, beside the tensor map they come through. */
template <u32_t window_bytes_>
struct cuda_gather_stages {

    /** Two stages of four rows' windows of up to @p window_bytes_ each, aligned as copies write
     * them. */
    alignas(128) byte_t stages[2][4 * window_bytes_];

    /** One barrier per stage. */
    u64_t barriers[2];

    /** The kernel parameter's tensor map, by its generic address. */
    cuda_tensor_map_t const* tensor_map;

    /** Rows the map spans, zero where the warp gathers none. */
    u32_t rows_count;

    /** Bytes of one row a copy moves. */
    u32_t window_bytes;

    /** Bit @b s is the parity of the phase stage @b s is filling. */
    u32_t phases;
};

/** Queues @p unit of a batch of @p gathered_count candidates into its stage of @p gather: window
 *  @p unit modulo @p windows of up to @p quads_per_unit quads, every quad's four rows copied back
 *  to back at @p quad_bytes apart through one @c tile::gather4 copy its own lane issues. Rows past
 *  the batch re-read its first row. */
template <u32_t window_bytes_>
__device__ __forceinline__ void cuda_tma_gather_unit(cuda_gather_stages<window_bytes_>& gather, u32_t const* ids,
                                                     u32_t gathered_count, u32_t unit, u32_t windows,
                                                     u32_t quads_per_unit, u32_t quad_bytes, u32_t lane) noexcept {
    u32_t const stage = unit & 1u, window = unit % windows, first_quad = unit / windows * quads_per_unit;
    u32_t const unit_quads = min_of(quads_per_unit, divide_round_up(gathered_count, 4u) - first_quad);
    u32_t const barrier = cuda_shared_address(gather.barriers + stage);
    // The stage's earlier generic reads must precede the copies' asynchronous writes.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (lane == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier),
                     "r"(unit_quads * 4 * gather.window_bytes)
                     : "memory");
    __syncwarp();
    if (lane >= unit_quads) return;
    u32_t const first_row = 4 * (first_quad + lane);
    u32_t rows[4];
#pragma unroll
    for (u32_t row = 0; row != 4; ++row) rows[row] = ids[first_row + row < gathered_count ? first_row + row : 0];
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes" //
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(cuda_shared_address(gather.stages[stage] + lane * quad_bytes)),
        "l"(reinterpret_cast<u64_t>(gather.tensor_map)), "r"(window * gather.window_bytes / 8), "r"(rows[0]),
        "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(barrier)
        : "memory");
}

/**
 *  @brief Scores a walk's batch from rows the Tensor Memory Accelerator gathers into @p gather,
 *      each @c tile::gather4 copy filling the same window of four candidates' rows.
 *
 *  A unit is one window of as many quads of candidates as a stage fits and the lanes cover, a
 *  single quad once rows outgrow a window; the warp scores one stage while the next unit copies
 *  into the other. A power-of-two group of lanes owns each quad, every lane a 16-byte column of its
 *  window and the quad's four accumulators, so one query fragment serves four rows. Past a quad's
 *  last window, each accumulator folds into the group's first lane. A window reaching past the row
 *  reads zeros, which no query fragment meets.
 *
 *  @return This lane's scored entry, or the sentinel on lanes past @p gathered_count.
 */
template <vram_quads_kit kit_, u32_t window_bytes_>
__device__ __forceinline__ vram_scored_node<node_id_t> cuda_tma_gather_quads( //
    cuda_gather_stages<window_bytes_>& gather, vram_flat_view<kit_> const& storage,
    typename kit_::fragment_t const* query_span, typename kit_::norms_t query_state,
    vram_gathered<kit_, cuda_warp_t>& gathered, u32_t gathered_count, u32_t lane) noexcept {

    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;
    constexpr u32_t width_k = cuda_warp_t::width_k;
    u32_t const window_bytes = gather.window_bytes;
    u32_t const window_fragments = window_bytes / sizeof(fragment_t);
    u32_t const windows = divide_round_up(storage.geometry.row_fragments, window_fragments);
    // Every copy writes on a 128-byte boundary, as tensor copies require.
    u32_t const quad_bytes = round_up_to_multiple(4 * window_bytes, 128u);
    u32_t const lanes_per_quad = min_of(1u << (32 - __clz(window_fragments - 1)), width_k);
    u32_t const quads_per_unit = min_of(width_k / lanes_per_quad,
                                        static_cast<u32_t>(sizeof(gather.stages[0])) / quad_bytes);
    u32_t const quads = divide_round_up(gathered_count, 4u);
    u32_t const units = divide_round_up(quads, quads_per_unit) * windows;
    u32_t const unit_quad = lane / lanes_per_quad, column = lane % lanes_per_quad;
    bool const live = lane < gathered_count;
    // Requested before the rows, so the two round trips overlap.
    typename kit_::norms_t const target_state = live ? storage.norm_for(gathered.ids[lane]) : typename kit_::norms_t {};
    u32_t phases = gather.phases;
    for (u32_t unit = 0; unit != min_of(units, 2u); ++unit)
        cuda_tma_gather_unit(gather, gathered.ids, gathered_count, unit, windows, quads_per_unit, quad_bytes, lane);

    dot_t accumulators[4];
#pragma unroll 1
    for (u32_t unit = 0; unit != units; ++unit) {
        u32_t const stage = unit & 1u, window = unit % windows;
        u32_t const quad = unit / windows * quads_per_unit + unit_quad;
        bool const scoring = unit_quad < quads_per_unit && quad < quads;
        cuda_tma_wait_barrier(gather.barriers + stage, (phases >> stage) & 1u);
        phases ^= 1u << stage;
        if (window == 0) {
#pragma unroll
            for (u32_t row = 0; row != 4; ++row) accumulators[row] = {};
        }
        for (u32_t fragment = column; scoring && fragment < window_fragments; fragment += lanes_per_quad) {
            u32_t const row_fragment = window * window_fragments + fragment;
            if (row_fragment >= storage.geometry.row_fragments) break;
            fragment_t const query_value = query_span[row_fragment];
            byte_t const* const columns = gather.stages[stage] + unit_quad * quad_bytes + fragment * sizeof(fragment_t);
#pragma unroll
            for (u32_t row = 0; row != 4; ++row)
                accumulators[row] = kit_::accumulate_dot(
                    *reinterpret_cast<fragment_t const*>(columns + row * window_bytes), query_value, accumulators[row]);
        }
        // Warp-uniform, so the reduction's full-mask shuffles stay legal.
        if (window + 1 == windows) {
#pragma unroll
            for (u32_t row = 0; row != 4; ++row) {
                dot_t const dot = vram_reduce<cuda_warp_t>(accumulators[row], lanes_per_quad);
                if (scoring && column == 0 && 4 * quad + row < gathered_count) gathered.dots[4 * quad + row] = dot;
            }
        }
        if (unit + 2 < units)
            cuda_tma_gather_unit(gather, gathered.ids, gathered_count, unit + 2, windows, quads_per_unit, quad_bytes,
                                 lane);
    }
    if (lane == 0) gather.phases = phases;
    cuda_warp_t::sync();
    if (!live) return vram_scored_node<node_id_t>::sentinel();
    vram_scored_node<node_id_t> const scored = vram_scored_node<node_id_t>::make(
        kit_::distance(storage.geometry.kernel, gathered.dots[lane], query_state, target_state), gathered.ids[lane]);
    gathered.scored[lane] = scored;
    return scored;
}

/** The Hopper staging: whole-row Tensor Memory Accelerator copies, a walking lane loading one
 *  fragment at once. */
struct hopper_tma_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_hopper_k;
    inline static constexpr bool compiled_k = NUMKONG_TARGET_HOPPER || NUMKONG_ARCH_CUDA_BLACKWELL_ ||
                                              NUMKONG_TARGET_BLACKWELLRTX;
    using warp_t = cuda_warp_t;
    inline static constexpr u32_t stages_k = 2;
    inline static constexpr u32_t fragments_per_load_k = 1;

    /** One barrier per stage, in the warp's static shared memory. */
    struct shared_t {
        u64_t barriers[stages_k];
    };

    /** The stages' barriers. */
    u64_t* barriers_;

    /** Rows per stage of the latest staging, and the leading rows copied in. */
    u32_t rows_per_stage_ = 0, arrived_rows_ = 0;

    /** Bit @b s is the parity of the phase stage @b s is filling. */
    u32_t phases_ = 0;

    __device__ __forceinline__ hopper_tma_schedule_t(shared_t& shared, u32_t lane) noexcept
        : barriers_(shared.barriers) {
        cuda_tma_initialize_barriers(barriers_, stages_k, lane);
    }

    /** Scores a walk's batch as @ref cuda_score_batch does at this schedule's load width. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return cuda_score_batch<kit_, resident_blocks_, fragments_per_load_k>( //
            storage, query_span, query_state, gathered, gathered_count, lane);
    }

    /** The greedy diversity prefix as @ref cuda_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return cuda_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        rows_per_stage_ = divide_round_up(rows, stages_k), arrived_rows_ = 0;
        if (cache.ids_) {
            arrived_rows_ = rows;
            return;
        }
        cuda_tma_copy_rows(barriers_, rows_per_stage_, storage, cache, ids, rows, lane);
    }

    __device__ __forceinline__ void await_rows(u32_t rows) noexcept {
        for (; arrived_rows_ < rows; arrived_rows_ += rows_per_stage_) {
            u32_t const stage = arrived_rows_ / rows_per_stage_;
            cuda_tma_wait_barrier(barriers_ + stage, (phases_ >> stage) & 1u);
            phases_ ^= 1u << stage;
        }
    }
};

/**
 *  @brief The datacenter Blackwell staging: the Hopper copies, a walking lane loading two adjacent
 *      fragments at once, whose 256-bit loads serve the random row gathers better, and a walk's
 *      batches scored from rows one @c tile::gather4 copy moves four at a time.
 *
 *  A walking kernel binds each warp's gathers to its parameter's tensor map at entry, through
 *  @ref bind; @ref score_batch then copies the batch through @ref cuda_tma_gather_quads, or loads
 *  it lane by lane where the kernel bound no map or the map misses any of the batch's rows. The
 *  stages of every walking warp live in static shared memory, sized by the kernel's resident blocks
 *  through @ref gather_window_bytes.
 */
struct blackwell_tma_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwell_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_BLACKWELL_;
    using warp_t = cuda_warp_t;
    inline static constexpr u32_t stages_k = 2;
    inline static constexpr u32_t fragments_per_load_k = 2;

    /** Shared memory of one multiprocessor, and what the driver keeps of it per resident block. */
    inline static constexpr u32_t multiprocessor_shared_bytes_k = 228 * 1024;
    inline static constexpr u32_t block_reserved_shared_bytes_k = 1024;

    /** Bytes of one row a gather copies at most under @p resident_blocks blocks per multiprocessor:
     *  one 16-byte column per lane, halved while the two stages of four rows that every warp of a
     *  block keeps outgrow the block's share of the multiprocessor. */
    static constexpr u32_t gather_window_bytes(u32_t resident_blocks) noexcept {
        u32_t window = warp_t::width_k * 16;
        while (vram_warps_per_block_k * stages_k * 4 * window >
               multiprocessor_shared_bytes_k / resident_blocks - block_reserved_shared_bytes_k)
            window /= 2;
        return window;
    }

    /** One warp's gathered rows under @p resident_blocks_ blocks per multiprocessor. */
    template <u32_t resident_blocks_>
    using gather_stages = cuda_gather_stages<gather_window_bytes(resident_blocks_)>;

    /** One barrier per stage, in the warp's static shared memory. */
    struct shared_t {
        u64_t barriers[stages_k];
    };

    /** The stages' barriers. */
    u64_t* barriers_;

    /** Rows per stage of the latest staging, and the leading rows copied in. */
    u32_t rows_per_stage_ = 0, arrived_rows_ = 0;

    /** Bit @b s is the parity of the phase stage @b s is filling. */
    u32_t phases_ = 0;

    __device__ __forceinline__ blackwell_tma_schedule_t(shared_t& shared, u32_t lane) noexcept
        : barriers_(shared.barriers) {
        cuda_tma_initialize_barriers(barriers_, stages_k, lane);
    }

    /** Binds the calling warp's gathers under @p resident_blocks_ blocks per multiprocessor to
     *  @p rows_map, the kernel's own parameter, warming its tensor map, and arms their barriers;
     *  a map whose windows outgrow the stages binds no rows. */
    template <u32_t resident_blocks_>
    static __device__ __forceinline__ void bind(cuda_rows_map_t const& rows_map, u32_t lane) noexcept {
        gather_stages<resident_blocks_>& gather = gather_stages_<resident_blocks_>();
        if (lane == 0) {
            bool const fits = 4 * rows_map.window_bytes <= sizeof(gather.stages[0]);
            gather.tensor_map = &rows_map.tensor_map;
            gather.rows_count = fits ? rows_map.rows_count : 0;
            gather.window_bytes = rows_map.window_bytes;
            gather.phases = 0;
            if (gather.rows_count)
                asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<u64_t>(&rows_map.tensor_map))
                             : "memory");
        }
        cuda_tma_initialize_barriers(gather.barriers, stages_k, lane);
    }

    /** Scores a walk's batch from rows gathered four at a time, or as @ref cuda_score_batch does at
     *  this schedule's load width where the bound map misses any of them. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        gather_stages<resident_blocks_>& gather = gather_stages_<resident_blocks_>();
        bool const unmapped = lane < gathered_count && gathered.ids[lane] >= gather.rows_count;
        if (warp_t::ballot(unmapped))
            return cuda_score_batch<kit_, resident_blocks_, fragments_per_load_k>( //
                storage, query_span, query_state, gathered, gathered_count, lane);
        return cuda_tma_gather_quads<kit_>(gather, storage, query_span, query_state, gathered, gathered_count, lane);
    }

    /** The greedy diversity prefix as @ref cuda_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return cuda_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        rows_per_stage_ = divide_round_up(rows, stages_k), arrived_rows_ = 0;
        if (cache.ids_) {
            arrived_rows_ = rows;
            return;
        }
        cuda_tma_copy_rows(barriers_, rows_per_stage_, storage, cache, ids, rows, lane);
    }

    __device__ __forceinline__ void await_rows(u32_t rows) noexcept {
        for (; arrived_rows_ < rows; arrived_rows_ += rows_per_stage_) {
            u32_t const stage = arrived_rows_ / rows_per_stage_;
            cuda_tma_wait_barrier(barriers_ + stage, (phases_ >> stage) & 1u);
            phases_ ^= 1u << stage;
        }
    }

  private:

    /** The calling warp's gathered rows, one per warp of the block, in its static shared memory. */
    template <u32_t resident_blocks_>
    static __device__ __forceinline__ gather_stages<resident_blocks_>& gather_stages_() noexcept {
        __shared__ gather_stages<resident_blocks_> gathers[vram_warps_per_block_k];
        return gathers[threadIdx.x / warp_t::width_k];
    }
};

/** The RTX Blackwell staging: the Hopper copies, a walking lane loading two adjacent fragments at
 *  once, whose 256-bit loads serve the random row gathers better. */
struct blackwellrtx_tma_schedule_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_blackwellrtx_k;
    inline static constexpr bool compiled_k = NUMKONG_TARGET_BLACKWELLRTX;
    using warp_t = cuda_warp_t;
    inline static constexpr u32_t stages_k = 2;
    inline static constexpr u32_t fragments_per_load_k = 2;

    /** One barrier per stage, in the warp's static shared memory. */
    struct shared_t {
        u64_t barriers[stages_k];
    };

    /** The stages' barriers. */
    u64_t* barriers_;

    /** Rows per stage of the latest staging, and the leading rows copied in. */
    u32_t rows_per_stage_ = 0, arrived_rows_ = 0;

    /** Bit @b s is the parity of the phase stage @b s is filling. */
    u32_t phases_ = 0;

    __device__ __forceinline__ blackwellrtx_tma_schedule_t(shared_t& shared, u32_t lane) noexcept
        : barriers_(shared.barriers) {
        cuda_tma_initialize_barriers(barriers_, stages_k, lane);
    }

    /** Scores a walk's batch as @ref cuda_score_batch does at this schedule's load width. */
    template <vram_quads_kit kit_, u32_t resident_blocks_>
    static __device__ __forceinline__ vram_scored_node<node_id_t> score_batch( //
        vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
        typename kit_::norms_t query_state, vram_gathered<kit_, warp_t>& gathered, u32_t gathered_count,
        u32_t lane) noexcept {
        return cuda_score_batch<kit_, resident_blocks_, fragments_per_load_k>( //
            storage, query_span, query_state, gathered, gathered_count, lane);
    }

    /** The greedy diversity prefix as @ref cuda_keep_diverse runs it, on this schedule. */
    template <vram_quads_kit kit_, typename storage_>
    __device__ __forceinline__ u32_t keep_diverse( //
        storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache,
        vram_sorted_buffer<warp_t, node_id_t> const& sorted, u32_t position_mask, u32_t candidates_count,
        vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {
        return cuda_keep_diverse<kit_>(storage, cache, *this, sorted, position_mask, candidates_count, selection,
                                       candidate_norms, kept_zero, telemetry, lane);
    }

    template <typename storage_, typename cache_>
    __device__ __forceinline__ void stage_rows(storage_ const& storage, cache_ cache, u32_t const* ids, u32_t rows,
                                               u32_t lane) noexcept {
        rows_per_stage_ = divide_round_up(rows, stages_k), arrived_rows_ = 0;
        if (cache.ids_) {
            arrived_rows_ = rows;
            return;
        }
        cuda_tma_copy_rows(barriers_, rows_per_stage_, storage, cache, ids, rows, lane);
    }

    __device__ __forceinline__ void await_rows(u32_t rows) noexcept {
        for (; arrived_rows_ < rows; arrived_rows_ += rows_per_stage_) {
            u32_t const stage = arrived_rows_ / rows_per_stage_;
            cuda_tma_wait_barrier(barriers_ + stage, (phases_ >> stage) & 1u);
            phases_ ^= 1u << stage;
        }
    }
};

#pragma endregion Schedules

#pragma region Ladders

/*  Each ladder lists the newest kit first. The Blackwell kits serve datacenter and RTX Blackwell
 *  alike, so their @c capability_k names both bits, and @c picked_k keeps the one the class
 *  carries; each Blackwell has its own schedule. */

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::f16_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        f16_ampere_quads_kit_t,         //
        f16_blackwell_quads_kit_t       //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::bf16_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        bf16_ampere_quads_kit_t,        //
        bf16_blackwell_quads_kit_t      //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::i8_t, caps_> {
    using kit_t = typename highest_kit<caps_, i8_ampere_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::u8_t, caps_> {
    using kit_t = typename highest_kit<caps_, u8_ampere_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::e5m2_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        e5m2_ampere_quads_kit_t,        //
        e5m2_blackwell_quads_kit_t,     //
        e5m2_ada_quads_kit_t            //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::e4m3_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        e4m3_ampere_quads_kit_t,        //
        e4m3_blackwell_quads_kit_t,     //
        e4m3_ada_quads_kit_t            //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::e3m2_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        e3m2_ampere_quads_kit_t,        //
        e3m2_blackwell_quads_kit_t      //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::e2m3_t, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        e2m3_ampere_quads_kit_t,        //
        e2m3_blackwell_quads_kit_t      //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::i4x2_t, caps_> {
    using kit_t = typename highest_kit<caps_, i4x2_ampere_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::u4x2_t, caps_> {
    using kit_t = typename highest_kit<caps_, u4x2_ampere_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct quads_ladder<nk::u1x8_t, caps_> {
    using kit_t = typename highest_kit<caps_, u1x8_ampere_quads_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

/** The staging schedule under @p caps_: whole-row Tensor Memory Accelerator copies on every class
 *  that has one, Hopper on, paired-fragment loads from Blackwell on, four-row gathers for the
 *  datacenter Blackwell's walks, and the @c cp.async ring below. */
template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct schedule_ladder<caps_> {
    using schedule_t = typename highest_kit< //
        caps_,                               //
        ampere_cp_async_schedule_t,          //
        blackwell_tma_schedule_t,            //
        blackwellrtx_tma_schedule_t,         //
        hopper_tma_schedule_t                //
        >::type;
    static constexpr nk_capability_t picked_k = caps_ & schedule_t::capability_k;
};

#pragma endregion Ladders

} // namespace unum::usearch::dense
