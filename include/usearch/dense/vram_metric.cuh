/**
 *  @file include/usearch/dense/vram_metric.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Device metric kits for the dense VRAM indexes - the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one storage dtype to one GPU generation, named like the CPU kits -
 *  @b dtype_generation_kit - with the metric as its parameter. The generation slot names the oldest
 *  tier whose instructions the kit needs: @c ampere for the sm_80 floor, @c ada for the fp8
 *  conversions, @c blackwell for the mixed-precision multiply-adds and the fp6 conversions. Each
 *  tier's kits sit behind its @c NUMKONG_TARGET_* gate and a chain picks among them under the
 *  caller's ceiling, exactly like the CPU chains, so a unit compiled for one architecture compiles
 *  the kits that architecture runs. The arithmetic lives in named device primitives; a kit binds them.
 *
 *  Fragments are 16-byte register tiles, and metrics with precomputable norms finalize "through the
 *  dot". Row geometry - how many fragments span one vector, how many lanes cooperate on a row - is
 *  runtime state of the planes view, not the kit: dimensions are frozen by the first @c reserve,
 *  never by a type. The cross-lane reduction is deliberately not a kit member - one shuffle tree
 *  serves every kit. Distances take both sides' norms, as the CPU finalizers do, so a symmetric
 *  candidate-to-candidate distance is the same call.
 *
 *  Families sort from the widest numeric types to the smallest & most exotic: 16-bit floats, 8-bit
 *  integers, minifloats, packed nibbles, packed bits. Norms are the CPU's @c vector_norms over each
 *  kit's dot type, so the unsigned families accumulate in unsigned words with defined wrapping and no
 *  cast chatter, while @c i8 keeps the CPU planes' signed @b {sum,sumsq} layout.
 */
#pragma once

#include <cuda_bf16.h> // `__nv_bfloat162` packed math, sm_80+
#include <cuda_fp16.h> // `__half2` packed math

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the chains key on it
#include <numkong/numkong.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"

#if defined(__CUDACC__)

namespace unum {
namespace usearch {

template <typename kit_>
concept dense_cuda_kit = requires(typename kit_::fragment_t fragment, typename kit_::dot_t dot,
                                  typename kit_::norms_accumulator_t norms_accumulator, typename kit_::norms_t norms,
                                  dense_distance_t ranked) {
    typename kit_::scalar_t;
    { kit_::arch_k } -> std::convertible_to<cuda_arch_t>;
    { kit_::accumulate_dot(fragment, fragment, dot) } -> std::same_as<typename kit_::dot_t>;
    { kit_::accumulate_norms(fragment, norms_accumulator) } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::partials_to_norms(norms_accumulator) } -> std::same_as<typename kit_::norms_t>;
    { kit_::distance(dot, norms, norms) } -> std::same_as<dense_distance_t>;
    { kit_::convert_to_distance(ranked) } -> std::same_as<dense_distance_t>;
};

/** Generic cross-lane reduction: folds sub-warp partials of width @p lanes_per_row
 *  (a warp-uniform power of two) down to the leader lane. One tree serves every kit. */
template <typename dot_type_>
__device__ __forceinline__ dot_type_ dense_cuda_reduce(dot_type_ partial, u32_t lanes_per_row) noexcept {
    for (u32_t offset = lanes_per_row / 2; offset; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFFu, partial, offset, lanes_per_row);
    return partial;
}

/** Full-warp reduction of a two-component norms accumulator - {sum,sumsq} for the
 *  arithmetic kits, {popcount,popcount} for the binary ones. */
__device__ __forceinline__ int2 dense_cuda_reduce_norms(int2 partial) noexcept {
#pragma unroll
    for (u32_t offset = 16; offset; offset >>= 1) {
        partial.x += __shfl_down_sync(0xFFFFFFFFu, partial.x, offset);
        partial.y += __shfl_down_sync(0xFFFFFFFFu, partial.y, offset);
    }
    return partial;
}
__device__ __forceinline__ uint2 dense_cuda_reduce_norms(uint2 partial) noexcept {
#pragma unroll
    for (u32_t offset = 16; offset; offset >>= 1) {
        partial.x += __shfl_down_sync(0xFFFFFFFFu, partial.x, offset);
        partial.y += __shfl_down_sync(0xFFFFFFFFu, partial.y, offset);
    }
    return partial;
}
__device__ __forceinline__ float2 dense_cuda_reduce_norms(float2 partial) noexcept {
#pragma unroll
    for (u32_t offset = 16; offset; offset >>= 1) {
        partial.x += __shfl_down_sync(0xFFFFFFFFu, partial.x, offset);
        partial.y += __shfl_down_sync(0xFFFFFFFFu, partial.y, offset);
    }
    return partial;
}

/** Rebroadcasts lane 0's reduced norms to every lane - values that feed
 *  lane-divergent control flow must be warp-uniform. */
__device__ __forceinline__ int2 dense_cuda_broadcast_norms(int2 reduced) noexcept {
    reduced.x = __shfl_sync(0xFFFFFFFFu, reduced.x, 0);
    reduced.y = __shfl_sync(0xFFFFFFFFu, reduced.y, 0);
    return reduced;
}
__device__ __forceinline__ uint2 dense_cuda_broadcast_norms(uint2 reduced) noexcept {
    reduced.x = __shfl_sync(0xFFFFFFFFu, reduced.x, 0);
    reduced.y = __shfl_sync(0xFFFFFFFFu, reduced.y, 0);
    return reduced;
}
__device__ __forceinline__ float2 dense_cuda_broadcast_norms(float2 reduced) noexcept {
    reduced.x = __shfl_sync(0xFFFFFFFFu, reduced.x, 0);
    reduced.y = __shfl_sync(0xFFFFFFFFu, reduced.y, 0);
    return reduced;
}

#pragma region Metrics

/** Angular distance from a dot and both sides' squared norms, finalized exactly like the CPU's
 *  @c nk_angular_through_f32_from_dot: clamped at zero, and a zero-norm side scores zero against a
 *  vanishing dot and one otherwise. The query's root is loop-invariant, so the compiler hoists it. */
__device__ __forceinline__ dense_distance_t dense_cuda_angular_through_dot(f32_t dot, f32_t query_sumsq,
                                                                           f32_t target_sumsq) noexcept {
    if (query_sumsq > 0.0f && target_sumsq > 0.0f)
        return fmaxf(1.0f - dot * (rsqrtf(query_sumsq) * rsqrtf(target_sumsq)), 0.0f);
    return dot == 0.0f ? 0.0f : 1.0f;
}

/** A numeric kit's distance for @p kernel_ "through the dot": @b 1-dot for the inner product, monotone
 *  decreasing and negative-safe, the clamped angular, and the squared Euclidean the walk ranks by. */
template <nk_kernel_kind_t kernel_, typename dot_type_>
__device__ __forceinline__ dense_distance_t dense_cuda_distance(dot_type_ dot, vector_norms<dot_type_> query,
                                                                vector_norms<dot_type_> target) noexcept {
    f32_t const dot_value = static_cast<f32_t>(dot);
    if constexpr (kernel_ == nk_kernel_dot_k) return 1.0f - dot_value;
    else if constexpr (kernel_ == nk_kernel_angular_k)
        return dense_cuda_angular_through_dot(dot_value, static_cast<f32_t>(query.sumsq),
                                              static_cast<f32_t>(target.sumsq));
    else {
        static_assert(kernel_ == nk_kernel_sqeuclidean_k || kernel_ == nk_kernel_euclidean_k,
                      "Numeric vectors score by dot, angular or euclidean");
        return static_cast<f32_t>(query.sumsq) + static_cast<f32_t>(target.sumsq) - 2.0f * dot_value;
    }
}

/** Output-boundary finalize, the CPU kits' @c convert_to_distances in miniature: the traversal ranks
 *  on raw @c distance values end to end, and only the matches a caller receives pass through here -
 *  identity unless a metric owes a monotone transform, like Euclidean's square root. */
template <nk_kernel_kind_t kernel_>
__device__ __forceinline__ dense_distance_t dense_cuda_convert_to_distance(dense_distance_t ranked) noexcept {
    if constexpr (kernel_ == nk_kernel_euclidean_k) return sqrtf(fmaxf(ranked, 0.0f));
    else return ranked;
}

#pragma endregion Metrics

/** The kits every chain names, declared whether or not their tier is compiled in. */
template <nk_kernel_kind_t kernel_>
struct f16_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct f16_blackwell_kit;
template <nk_kernel_kind_t kernel_>
struct bf16_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct bf16_blackwell_kit;
template <nk_kernel_kind_t kernel_>
struct i8_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct u8_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct e5m2_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct e5m2_ada_kit;
template <nk_kernel_kind_t kernel_>
struct e5m2_blackwell_kit;
template <nk_kernel_kind_t kernel_>
struct e4m3_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct e4m3_ada_kit;
template <nk_kernel_kind_t kernel_>
struct e4m3_blackwell_kit;
template <nk_kernel_kind_t kernel_>
struct e3m2_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct e3m2_blackwell_kit;
template <nk_kernel_kind_t kernel_>
struct e2m3_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct e2m3_blackwell_kit;
template <nk_kernel_kind_t kernel_>
struct i4x2_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct u4x2_ampere_kit;
template <nk_kernel_kind_t kernel_>
struct u1x8_ampere_kit;

#pragma region Half Precision Floats

/** Eight @c f16 products of a fragment pair, each rounded to half by @c hmul2, summed in f32 per fragment
 *  so no half-range overflow can accumulate. */
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

/** The @c bf16 twin of @ref dense_cuda_dot_f16x8_ampere, through the @c bf16x2 instructions that land
 *  exactly at the sm_80 floor. */
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

/** f16 kit - the sm_80 floor. */
template <nk_kernel_kind_t kernel_>
struct f16_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_f16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_f16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** bf16 kit - the sm_80 floor. */
template <nk_kernel_kind_t kernel_>
struct bf16_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_bf16x8_ampere(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_bf16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

#if NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

/** Both halves of two @c f16x2 words multiplied straight into an f32 sum by @c fma.rn.f32.f16 - exact
 *  products, so none rounds to half or overflows its range. */
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
template <nk_kernel_kind_t kernel_>
struct f16_blackwell_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_blackwell_k;
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_f16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_f16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** bf16 kit - Blackwell, @c fma.rn.f32.bf16. */
template <nk_kernel_kind_t kernel_>
struct bf16_blackwell_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_blackwell_k;
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_bf16x8_blackwell(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_bf16x8_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
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

/** An @c i8 fragment's sum & sum of squares, through the same instruction against splat-one and self. */
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

/** Sixteen unsigned @c u8 products of a fragment pair - the unsigned @c dp4a form, whose native operand
 *  types need no cast chatter; the dot alone can pass the signed range at wide rows. */
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

/** i8 kit - the sm_80 floor. */
template <nk_kernel_kind_t kernel_>
struct i8_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using dot_t = int;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_i8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return dense_cuda_moments_i8x16_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** u8 kit - the sm_80 floor. */
template <nk_kernel_kind_t kernel_>
struct u8_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::u8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_u8x16_ampere(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return dense_cuda_moments_u8x16_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

#pragma endregion Byte Sized Integers

#pragma region Minifloats

/*  No generation has fp8 or fp6 arithmetic outside the tensor cores - conversion is all the ISA offers -
 *  so every minifloat kit widens byte pairs and accumulates like the 16-bit float kits, and one
 *  generation's kit differs from another's only in its widening. */

/** Sixteen minifloat products of a fragment pair, each byte pair widened by @p widening_type_ into f32. */
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

/** A minifloat fragment's sum & sum of squares, each byte pair widened by @p widening_type_ into f32. */
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

/** The OCP fp6 formats sit low in a byte - sign at bit 5, no infinities or NaN - so the floor widens a
 *  pair by hand: the five magnitude bits drop into a half's exponent and mantissa fields as they stand,
 *  the sign moves to bit 15, and one exact power-of-two product rebiases, subnormals too. */
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
template <nk_kernel_kind_t kernel_>
struct e5m2_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e5m2_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e5m2_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e4m3 kit - the sm_80 floor, widening by hand. */
template <nk_kernel_kind_t kernel_>
struct e4m3_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e4m3_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e4m3_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e3m2 kit - the sm_80 floor, widening by hand. */
template <nk_kernel_kind_t kernel_>
struct e3m2_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e3m2_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e3m2_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e2m3 kit - the sm_80 floor, widening by hand. */
template <nk_kernel_kind_t kernel_>
struct e2m3_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e2m3_ampere_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e2m3_ampere_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
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
template <nk_kernel_kind_t kernel_>
struct e5m2_ada_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ada_k;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e5m2_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e5m2_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e4m3 kit - Ada, widening through the conversion. */
template <nk_kernel_kind_t kernel_>
struct e4m3_ada_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ada_k;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_ampere<e4m3_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e4m3_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

#endif // NUMKONG_TARGET_ADA

#if NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

/** Sixteen minifloat products of a fragment pair, each byte pair widened by @p widening_type_ into
 *  halves and multiplied straight into the f32 sum, so the dot never widens to f32 - the norms still do. */
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

/** Blackwell widens an fp6 pair in one @c cvt.rn.f16x2.e3m2x2, which only family-specific code encodes. */
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
template <nk_kernel_kind_t kernel_>
struct e5m2_blackwell_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_blackwell_k;
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e5m2_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e5m2_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e4m3 kit - Blackwell, converted halves through @c fma.rn.f32.f16. */
template <nk_kernel_kind_t kernel_>
struct e4m3_blackwell_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_blackwell_k;
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e4m3_ada_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e4m3_ada_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e3m2 kit - Blackwell, the one-instruction conversion into @c fma.rn.f32.f16. */
template <nk_kernel_kind_t kernel_>
struct e3m2_blackwell_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_blackwell_k;
    using scalar_t = nk::e3m2_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e3m2_blackwell_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e3m2_blackwell_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** e2m3 kit - Blackwell, the one-instruction conversion into @c fma.rn.f32.f16. */
template <nk_kernel_kind_t kernel_>
struct e2m3_blackwell_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_blackwell_k;
    using scalar_t = nk::e2m3_t;
    using fragment_t = uint4;
    using dot_t = f32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = float2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_minifloats_blackwell<e2m3_blackwell_widening_t>(a, b, sum);
    }
    static __device__ __forceinline__ float2 accumulate_norms(fragment_t a, float2 moments) noexcept {
        return dense_cuda_moments_minifloats_ampere<e2m3_blackwell_widening_t>(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(float2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

#endif // NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX

#pragma endregion Minifloats

#pragma region Packed Nibbles

/** Unpacks a word of eight signed nibbles into two byte-lane words, sign-extended through an xor-add-xor
 *  bias, since a word-wide subtraction would borrow across lanes. */
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

/** i4x2 kit - the sm_80 floor, the byte engine over unpacked nibbles. */
template <nk_kernel_kind_t kernel_>
struct i4x2_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using dot_t = int;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = int2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_i4x32_ampere(a, b, sum);
    }
    static __device__ __forceinline__ int2 accumulate_norms(fragment_t a, int2 moments) noexcept {
        return dense_cuda_moments_i4x32_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(int2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

/** u4x2 kit - the sm_80 floor, the byte engine over unpacked nibbles. */
template <nk_kernel_kind_t kernel_>
struct u4x2_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::u4x2_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return dense_cuda_dot_u4x32_ampere(a, b, sum);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        return dense_cuda_moments_u4x32_ampere(a, moments);
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.y};
    }
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        return dense_cuda_distance<kernel_>(dot, query, target);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return dense_cuda_convert_to_distance<kernel_>(ranked);
    }
};

#pragma endregion Packed Nibbles

#pragma region Packed Bits

/** The intersection popcount of a fragment pair via two 64-bit @c popc - the counter, not the logic,
 *  limits throughput, so the AND folds in front of as few @c popc as possible. */
__device__ __forceinline__ u32_t dense_cuda_intersect_u1x128_ampere(uint4 a, uint4 b) noexcept {
    u64_t const low = (static_cast<u64_t>(a.y) << 32 | a.x) & (static_cast<u64_t>(b.y) << 32 | b.x);
    u64_t const high = (static_cast<u64_t>(a.w) << 32 | a.z) & (static_cast<u64_t>(b.w) << 32 | b.z);
    return static_cast<u32_t>(__popcll(low) + __popcll(high));
}

/** u1x8 kit - the sm_80 floor: the "dot" is the intersection popcount, and the norms store each row's
 *  set-bit count, the currency of Jaccard's union, mirrored into both fields. */
template <nk_kernel_kind_t kernel_>
struct u1x8_ampere_kit {
    inline static constexpr cuda_arch_t arch_k = cuda_arch_ampere_k;
    using scalar_t = nk::u1x8_t;
    using fragment_t = uint4;
    using dot_t = u32_t;
    using norms_t = vector_norms<dot_t>;
    using norms_accumulator_t = uint2;

    static __device__ __forceinline__ dot_t accumulate_dot(fragment_t a, fragment_t b, dot_t sum) noexcept {
        return sum + dense_cuda_intersect_u1x128_ampere(a, b);
    }
    static __device__ __forceinline__ uint2 accumulate_norms(fragment_t a, uint2 moments) noexcept {
        moments.x += dense_cuda_intersect_u1x128_ampere(a, a);
        return moments;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(uint2 moments) noexcept {
        return {moments.x, moments.x};
    }

    /** Hamming is |a| + |b| - 2·intersection, never negative since the intersection is bounded by either
     *  count; Jaccard is 1 - intersection / union, and two empty sets are identical, distance zero. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, norms_t query, norms_t target) noexcept {
        if constexpr (kernel_ == nk_kernel_dot_k) return 1.0f - static_cast<f32_t>(dot);
        else if constexpr (kernel_ == nk_kernel_hamming_k) return static_cast<f32_t>(query.sum + target.sum - 2 * dot);
        else {
            static_assert(kernel_ == nk_kernel_jaccard_k, "Bit vectors score by dot, hamming or jaccard");
            u32_t const unions = query.sum + target.sum - dot;
            return unions ? 1.0f - static_cast<f32_t>(dot) / static_cast<f32_t>(unions) : 0.0f;
        }
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

#pragma endregion Packed Bits

#pragma region Chains

/** The dispatch entry every index instantiation resolves through - the CPU chains' twin: one row per
 *  generation, newest first, each gated on its tier being compiled in and admitted by the caller's
 *  @p caps_, over the Ampere floor. Two ceilings in one binary run different rows on one device. */
template <nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain;

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_f16_k, kernel_, caps_> {
    using kit_t = typename first_available_kit<                                                                    //
        f16_ampere_kit<kernel_>,                                                                                   //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && caps_ >= cuda_arch_blackwellrtx_k, f16_blackwell_kit<kernel_>>, //
        chain_layer<NUMKONG_TARGET_BLACKWELL && caps_ >= cuda_arch_blackwell_k, f16_blackwell_kit<kernel_>>        //
        >::type;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_bf16_k, kernel_, caps_> {
    using kit_t = typename first_available_kit< //
        bf16_ampere_kit<kernel_>,               //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && caps_ >= cuda_arch_blackwellrtx_k,
                    bf16_blackwell_kit<kernel_>>,                                                            //
        chain_layer<NUMKONG_TARGET_BLACKWELL && caps_ >= cuda_arch_blackwell_k, bf16_blackwell_kit<kernel_>> //
        >::type;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_i8_k, kernel_, caps_> {
    using kit_t = i8_ampere_kit<kernel_>;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_u8_k, kernel_, caps_> {
    using kit_t = u8_ampere_kit<kernel_>;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_e5m2_k, kernel_, caps_> {
    using kit_t = typename first_available_kit< //
        e5m2_ampere_kit<kernel_>,               //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && caps_ >= cuda_arch_blackwellrtx_k,
                    e5m2_blackwell_kit<kernel_>>,                                                             //
        chain_layer<NUMKONG_TARGET_BLACKWELL && caps_ >= cuda_arch_blackwell_k, e5m2_blackwell_kit<kernel_>>, //
        chain_layer<NUMKONG_TARGET_ADA && caps_ >= cuda_arch_ada_k,
                    e5m2_ada_kit<kernel_>> //
        >::type;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_e4m3_k, kernel_, caps_> {
    using kit_t = typename first_available_kit< //
        e4m3_ampere_kit<kernel_>,               //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && caps_ >= cuda_arch_blackwellrtx_k,
                    e4m3_blackwell_kit<kernel_>>,                                                             //
        chain_layer<NUMKONG_TARGET_BLACKWELL && caps_ >= cuda_arch_blackwell_k, e4m3_blackwell_kit<kernel_>>, //
        chain_layer<NUMKONG_TARGET_ADA && caps_ >= cuda_arch_ada_k,
                    e4m3_ada_kit<kernel_>> //
        >::type;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_e3m2_k, kernel_, caps_> {
    using kit_t = typename first_available_kit< //
        e3m2_ampere_kit<kernel_>,               //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && caps_ >= cuda_arch_blackwellrtx_k,
                    e3m2_blackwell_kit<kernel_>>,                                                            //
        chain_layer<NUMKONG_TARGET_BLACKWELL && caps_ >= cuda_arch_blackwell_k, e3m2_blackwell_kit<kernel_>> //
        >::type;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_e2m3_k, kernel_, caps_> {
    using kit_t = typename first_available_kit< //
        e2m3_ampere_kit<kernel_>,               //
        chain_layer<NUMKONG_TARGET_BLACKWELLRTX && caps_ >= cuda_arch_blackwellrtx_k,
                    e2m3_blackwell_kit<kernel_>>,                                                            //
        chain_layer<NUMKONG_TARGET_BLACKWELL && caps_ >= cuda_arch_blackwell_k, e2m3_blackwell_kit<kernel_>> //
        >::type;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_i4_k, kernel_, caps_> {
    using kit_t = i4x2_ampere_kit<kernel_>;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_u4_k, kernel_, caps_> {
    using kit_t = u4x2_ampere_kit<kernel_>;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

template <nk_kernel_kind_t kernel_, cuda_arch_t caps_>
struct dense_cuda_kit_chain<nk_u1_k, kernel_, caps_> {
    using kit_t = u1x8_ampere_kit<kernel_>;
    static constexpr cuda_arch_t picked_k = kit_t::arch_k;
};

#pragma endregion Chains

} // namespace usearch
} // namespace unum

#endif // defined(__CUDACC__)
