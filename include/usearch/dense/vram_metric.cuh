/**
 *  @file include/usearch/dense/vram_metric.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Device metric kits for the dense VRAM indexes - the GPU twin of @c ram_metric.hpp.
 *
 *  A kit binds one storage dtype to one metric on one GPU generation, named like the CPU
 *  kits: @b dtype_arch_metric_kit_t, where the dot-product base takes the @c inner slot
 *  and the architecture slot names the oldest generation whose instructions it needs -
 *  @c ampere for the sm_80 floor, @c ada for the fp8 conversions, @c blackwell reserved
 *  for the sm_100-family formats. Fragments are 16-byte register tiles, and metrics with
 *  precomputable norms finalize "through the dot". Row geometry - how many fragments span
 *  one vector, how many lanes cooperate on a row - is runtime state of the planes view,
 *  not the kit: dimensions are frozen by the first @c reserve, never by a type. The
 *  cross-lane reduction is deliberately not a kit member - one shuffle tree serves every
 *  kit. Symmetric candidate-to-candidate distances reuse
 *  @c distance(dot, make_query_state(norms_a), norms_b).
 *
 *  Families sort from the widest numeric types to the smallest & most exotic: 16-bit
 *  floats, 8-bit integers, 8-bit floats, packed nibbles, packed bits. Every family is a
 *  concrete struct per dtype - no flag parameters selecting behavior, and each kit owns
 *  its record types outright, so the unsigned families accumulate in unsigned words with
 *  defined wrapping and no cast chatter, while @c i8 keeps the CPU planes' signed
 *  @b {sum,sumsq} layout.
 */
#pragma once

#include <cuda_bf16.h> // `__nv_bfloat162` packed math, sm_80+
#include <cuda_fp16.h> // `__half2` packed math

#include <numkong/numkong.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"

#if defined(__CUDACC__)

namespace unum {
namespace usearch {

/** The generation this compilation pass can encode: the device passes carry their
 *  target's @c __CUDA_ARCH__, the host pass admits only the floor - harmless, since
 *  generation rows inherit their floor's record layouts and the host touches layouts
 *  alone. Kit-chain layers intersect their architecture requirement with this AND with
 *  the caller's ceiling, the CPU-style side-by-side selector. */
#if defined(__CUDA_ARCH__)
inline constexpr cuda_arch_t dense_cuda_target_arch_k = static_cast<cuda_arch_t>(__CUDA_ARCH__);
#else
inline constexpr cuda_arch_t dense_cuda_target_arch_k = cuda_arch_ampere_k;
#endif

template <typename kit_>
concept dense_cuda_kit = requires(typename kit_::fragment_t fragment, typename kit_::accumulator_t accumulator,
                                  typename kit_::norms_accumulator_t norms_accumulator, typename kit_::norms_t norms,
                                  typename kit_::query_state_t query_state, typename kit_::dot_t dot,
                                  dense_distance_t ranked) {
    typename kit_::scalar_t;
    typename kit_::fragment_t;
    typename kit_::accumulator_t;
    typename kit_::dot_t;
    typename kit_::norms_t;
    typename kit_::norms_accumulator_t;
    typename kit_::query_state_t;
    { kit_::fragment_bytes_k } -> std::convertible_to<u32_t>;
    { kit_::max_gathers_in_flight_k } -> std::convertible_to<u32_t>;
    { kit_::has_norms_k } -> std::convertible_to<bool>;
    { kit_::zero_accumulator() } -> std::same_as<typename kit_::accumulator_t>;
    { kit_::accumulate_dot(fragment, fragment, accumulator) } -> std::same_as<typename kit_::accumulator_t>;
    { kit_::partial_to_dot(accumulator) } -> std::same_as<typename kit_::dot_t>;
    { kit_::zero_norms_accumulator() } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::accumulate_norms(fragment, norms_accumulator) } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::partials_to_norms(norms_accumulator) } -> std::same_as<typename kit_::norms_t>;
    { kit_::make_query_state(norms) } -> std::same_as<typename kit_::query_state_t>;
    { kit_::distance(dot, query_state, norms) } -> std::same_as<dense_distance_t>;
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

#pragma region Single Precision Floats

/** The @c f32 dot base over 4-value fragments - plain fused multiply-adds, the
 *  reference arithmetic every narrower family approximates. */
struct f32_ampere_inner_kit_t {
    using scalar_t = nk::f32_t;
    using fragment_t = uint4;
    using accumulator_t = f32_t;
    using dot_t = f32_t;
    struct norms_t {
        f32_t sum, sumsq;
    };
    using norms_accumulator_t = float2;
    struct query_state_t {
        f32_t rsqrt;
        f32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0.0f; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        accumulator = fmaf(__uint_as_float(a.x), __uint_as_float(b.x), accumulator);
        accumulator = fmaf(__uint_as_float(a.y), __uint_as_float(b.y), accumulator);
        accumulator = fmaf(__uint_as_float(a.z), __uint_as_float(b.z), accumulator);
        return fmaf(__uint_as_float(a.w), __uint_as_float(b.w), accumulator);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0.0f, 0.0f}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            f32_t const value = __uint_as_float(words[word]);
            accumulator.x += value;
            accumulator.y = fmaf(value, value, accumulator.y);
        }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - dot;
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct f32_ampere_angular_kit_t : f32_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - dot * query_state.rsqrt * (norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f);
    }
};
struct f32_ampere_sqeuclidean_kit_t : f32_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return query_state.sumsq + norms.sumsq - 2.0f * dot;
    }
};
struct f32_ampere_euclidean_kit_t : f32_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

#pragma endregion

#pragma region Half Precision Floats

/** The @c f16 dot base over 8-value fragments: packed multiplies widened into an
 *  @c f32 accumulator every fragment, so no half-range overflow can accumulate. */
struct f16_ampere_inner_kit_t {
    using scalar_t = nk::f16_t;
    using fragment_t = uint4;
    using accumulator_t = f32_t;
    using dot_t = f32_t;
    struct norms_t {
        f32_t sum, sumsq;
    };
    using norms_accumulator_t = float2;
    struct query_state_t {
        f32_t rsqrt;
        f32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;

    /** Ceiling on concurrently in-flight candidate-vector gathers per lane - the
     *  memory-parallelism window over the @b candidate axis, never the dimension axis. */
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ float2 pair_products(u32_t word_a, u32_t word_b) noexcept {
        __half2 const packed_a = *reinterpret_cast<__half2 const*>(&word_a);
        __half2 const packed_b = *reinterpret_cast<__half2 const*>(&word_b);
        return __half22float2(__hmul2(packed_a, packed_b));
    }
    static __device__ __forceinline__ float2 pair_values(u32_t word) noexcept {
        return __half22float2(*reinterpret_cast<__half2 const*>(&word));
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0.0f; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        float2 const products_x = pair_products(a.x, b.x), products_y = pair_products(a.y, b.y);
        float2 const products_z = pair_products(a.z, b.z), products_w = pair_products(a.w, b.w);
        return accumulator + (products_x.x + products_x.y) + (products_y.x + products_y.y) +
               (products_z.x + products_z.y) + (products_w.x + products_w.y);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0.0f, 0.0f}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            float2 const values = pair_values(words[word]);
            accumulator.x += values.x + values.y;
            accumulator.y += values.x * values.x + values.y * values.y;
        }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - dot;
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct f16_ampere_angular_kit_t : f16_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - dot * query_state.rsqrt * (norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f);
    }
};
struct f16_ampere_sqeuclidean_kit_t : f16_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return query_state.sumsq + norms.sumsq - 2.0f * dot;
    }
};
struct f16_ampere_euclidean_kit_t : f16_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

/** The @c bf16 dot base - the same widening discipline as f16 through the
 *  @c bf16x2 instructions that land exactly at the sm_80 floor. */
struct bf16_ampere_inner_kit_t {
    using scalar_t = nk::bf16_t;
    using fragment_t = uint4;
    using accumulator_t = f32_t;
    using dot_t = f32_t;
    struct norms_t {
        f32_t sum, sumsq;
    };
    using norms_accumulator_t = float2;
    struct query_state_t {
        f32_t rsqrt;
        f32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ float2 pair_products(u32_t word_a, u32_t word_b) noexcept {
        __nv_bfloat162 const packed_a = *reinterpret_cast<__nv_bfloat162 const*>(&word_a);
        __nv_bfloat162 const packed_b = *reinterpret_cast<__nv_bfloat162 const*>(&word_b);
        return __bfloat1622float2(__hmul2(packed_a, packed_b));
    }
    static __device__ __forceinline__ float2 pair_values(u32_t word) noexcept {
        return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&word));
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0.0f; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        float2 const products_x = pair_products(a.x, b.x), products_y = pair_products(a.y, b.y);
        float2 const products_z = pair_products(a.z, b.z), products_w = pair_products(a.w, b.w);
        return accumulator + (products_x.x + products_x.y) + (products_y.x + products_y.y) +
               (products_z.x + products_z.y) + (products_w.x + products_w.y);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0.0f, 0.0f}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            float2 const values = pair_values(words[word]);
            accumulator.x += values.x + values.y;
            accumulator.y += values.x * values.x + values.y * values.y;
        }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - dot;
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct bf16_ampere_angular_kit_t : bf16_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - dot * query_state.rsqrt * (norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f);
    }
};
struct bf16_ampere_sqeuclidean_kit_t : bf16_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return query_state.sumsq + norms.sumsq - 2.0f * dot;
    }
};
struct bf16_ampere_euclidean_kit_t : bf16_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

#pragma endregion

#pragma region Byte Sized Integers

/** The @c i8 dot base - one signed @c dp4a per fragment word, norms via the same
 *  instruction against splat-one and self. Signed records throughout: sums are
 *  genuinely negative, and the layout matches the CPU planes' @b {sum,sumsq}. */
struct i8_ampere_inner_kit_t {
    using scalar_t = nk::i8_t;
    using fragment_t = uint4;
    using accumulator_t = int;
    using dot_t = int;
    struct norms_t {
        int sum, sumsq;
    };
    using norms_accumulator_t = int2;
    struct query_state_t {
        f32_t rsqrt;
        int sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        accumulator = __dp4a(static_cast<int>(a.x), static_cast<int>(b.x), accumulator);
        accumulator = __dp4a(static_cast<int>(a.y), static_cast<int>(b.y), accumulator);
        accumulator = __dp4a(static_cast<int>(a.z), static_cast<int>(b.z), accumulator);
        return __dp4a(static_cast<int>(a.w), static_cast<int>(b.w), accumulator);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0, 0}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        accumulator.x = __dp4a(static_cast<int>(a.x), 0x01010101, accumulator.x);
        accumulator.x = __dp4a(static_cast<int>(a.y), 0x01010101, accumulator.x);
        accumulator.x = __dp4a(static_cast<int>(a.z), 0x01010101, accumulator.x);
        accumulator.x = __dp4a(static_cast<int>(a.w), 0x01010101, accumulator.x);
        accumulator.y = __dp4a(static_cast<int>(a.x), static_cast<int>(a.x), accumulator.y);
        accumulator.y = __dp4a(static_cast<int>(a.y), static_cast<int>(a.y), accumulator.y);
        accumulator.y = __dp4a(static_cast<int>(a.z), static_cast<int>(a.z), accumulator.y);
        accumulator.y = __dp4a(static_cast<int>(a.w), static_cast<int>(a.w), accumulator.y);
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - static_cast<f32_t>(dot);
    }

    /** Output-boundary finalize, the CPU kits' @c convert_to_distances in miniature:
     *  the traversal ranks on raw @c distance values end to end, and only the matches a
     *  caller receives pass through here - identity unless a metric owes a monotone
     *  transform, like Euclidean's square root. */
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct i8_ampere_angular_kit_t : i8_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(int dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - static_cast<f32_t>(dot) * query_state.rsqrt *
                          (norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f);
    }
};
struct i8_ampere_sqeuclidean_kit_t : i8_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(int dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return static_cast<f32_t>(query_state.sumsq) + static_cast<f32_t>(norms.sumsq) - 2.0f * static_cast<f32_t>(dot);
    }
};

/** Metric-true Euclidean over the squared-distance walk: rankings are identical
 *  under the monotone square root, so only reported matches pay one @c sqrtf each -
 *  numerically matching the CPU euclidean kits. */
struct i8_ampere_euclidean_kit_t : i8_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

/** The @c u8 dot base - the unsigned @c dp4a form with unsigned records: every
 *  quantity is non-negative, the dot alone can pass the signed range at wide rows,
 *  and the instruction's native operand types need no cast chatter. */
struct u8_ampere_inner_kit_t {
    using scalar_t = nk::u8_t;
    using fragment_t = uint4;
    using accumulator_t = u32_t;
    using dot_t = u32_t;
    struct norms_t {
        u32_t sum, sumsq;
    };
    using norms_accumulator_t = uint2;
    struct query_state_t {
        f32_t rsqrt;
        u32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        accumulator = __dp4a(a.x, b.x, accumulator);
        accumulator = __dp4a(a.y, b.y, accumulator);
        accumulator = __dp4a(a.z, b.z, accumulator);
        return __dp4a(a.w, b.w, accumulator);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0, 0}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        accumulator.x = __dp4a(a.x, 0x01010101u, accumulator.x);
        accumulator.x = __dp4a(a.y, 0x01010101u, accumulator.x);
        accumulator.x = __dp4a(a.z, 0x01010101u, accumulator.x);
        accumulator.x = __dp4a(a.w, 0x01010101u, accumulator.x);
        accumulator.y = __dp4a(a.x, a.x, accumulator.y);
        accumulator.y = __dp4a(a.y, a.y, accumulator.y);
        accumulator.y = __dp4a(a.z, a.z, accumulator.y);
        accumulator.y = __dp4a(a.w, a.w, accumulator.y);
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - static_cast<f32_t>(dot);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct u8_ampere_angular_kit_t : u8_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(u32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - static_cast<f32_t>(dot) * query_state.rsqrt *
                          (norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f);
    }
};
struct u8_ampere_sqeuclidean_kit_t : u8_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(u32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return static_cast<f32_t>(query_state.sumsq) + static_cast<f32_t>(norms.sumsq) - 2.0f * static_cast<f32_t>(dot);
    }
};
struct u8_ampere_euclidean_kit_t : u8_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

#pragma endregion

#pragma region Byte Sized Floats

#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)

/** The @c e4m3 dot base: no generation has fp8 arithmetic - conversion is all the
 *  ISA offers - so pairs widen through the sm_89+ @c cvt.rn.f16x2.e4m3x2 and
 *  accumulate like the 16-bit float kits; hence the @c ada architecture slot. */
struct e4m3_ada_inner_kit_t {
    using scalar_t = nk::e4m3_t;
    using fragment_t = uint4;
    using accumulator_t = f32_t;
    using dot_t = f32_t;
    struct norms_t {
        f32_t sum, sumsq;
    };
    using norms_accumulator_t = float2;
    struct query_state_t {
        f32_t rsqrt;
        f32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t half_pair;
        asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0.0f; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        u32_t const words_a[4] = {a.x, a.y, a.z, a.w};
        u32_t const words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word)
#pragma unroll
            for (u32_t pair = 0; pair != 2; ++pair) {
                float2 const values_a = pair_to_floats(words_a[word] >> (16 * pair));
                float2 const values_b = pair_to_floats(words_b[word] >> (16 * pair));
                accumulator += values_a.x * values_b.x + values_a.y * values_b.y;
            }
        return accumulator;
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0.0f, 0.0f}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word)
#pragma unroll
            for (u32_t pair = 0; pair != 2; ++pair) {
                float2 const values = pair_to_floats(words[word] >> (16 * pair));
                accumulator.x += values.x + values.y;
                accumulator.y += values.x * values.x + values.y * values.y;
            }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - dot;
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct e4m3_ada_angular_kit_t : e4m3_ada_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - dot * query_state.rsqrt * (norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f);
    }
};
struct e4m3_ada_sqeuclidean_kit_t : e4m3_ada_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return query_state.sumsq + norms.sumsq - 2.0f * dot;
    }
};
struct e4m3_ada_euclidean_kit_t : e4m3_ada_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

/** The @c e5m2 dot base - the wide-exponent fp8, widening through
 *  @c cvt.rn.f16x2.e5m2x2. */
struct e5m2_ada_inner_kit_t {
    using scalar_t = nk::e5m2_t;
    using fragment_t = uint4;
    using accumulator_t = f32_t;
    using dot_t = f32_t;
    struct norms_t {
        f32_t sum, sumsq;
    };
    using norms_accumulator_t = float2;
    struct query_state_t {
        f32_t rsqrt;
        f32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ float2 pair_to_floats(u32_t packed_pair) noexcept {
        u32_t half_pair;
        asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}" : "=r"(half_pair) : "h"(static_cast<unsigned short>(packed_pair)));
        return __half22float2(*reinterpret_cast<__half2 const*>(&half_pair));
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0.0f; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        u32_t const words_a[4] = {a.x, a.y, a.z, a.w};
        u32_t const words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word)
#pragma unroll
            for (u32_t pair = 0; pair != 2; ++pair) {
                float2 const values_a = pair_to_floats(words_a[word] >> (16 * pair));
                float2 const values_b = pair_to_floats(words_b[word] >> (16 * pair));
                accumulator += values_a.x * values_b.x + values_a.y * values_b.y;
            }
        return accumulator;
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0.0f, 0.0f}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word)
#pragma unroll
            for (u32_t pair = 0; pair != 2; ++pair) {
                float2 const values = pair_to_floats(words[word] >> (16 * pair));
                accumulator.x += values.x + values.y;
                accumulator.y += values.x * values.x + values.y * values.y;
            }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - dot;
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct e5m2_ada_angular_kit_t : e5m2_ada_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - dot * query_state.rsqrt * (norms.sumsq > 0.0f ? rsqrtf(norms.sumsq) : 0.0f);
    }
};
struct e5m2_ada_sqeuclidean_kit_t : e5m2_ada_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(f32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return query_state.sumsq + norms.sumsq - 2.0f * dot;
    }
};
struct e5m2_ada_euclidean_kit_t : e5m2_ada_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

#endif

#pragma endregion

#pragma region Packed Nibbles

/** The @c i4x2 dot base: SWAR-unpacks each word of 8 signed nibbles into two
 *  byte-lane words - sign-extended through the classic xor-sub bias - and feeds the
 *  byte engine, two @c dp4a per source word. Signed records, like @c i8. */
struct i4x2_ampere_inner_kit_t {
    using scalar_t = nk::i4x2_t;
    using fragment_t = uint4;
    using accumulator_t = int;
    using dot_t = int;
    struct norms_t {
        int sum, sumsq;
    };
    using norms_accumulator_t = int2;
    struct query_state_t {
        f32_t rsqrt;
        int sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ uint2 unpack(u32_t word) noexcept {
        u32_t const low = ((word & 0x0F0F0F0Fu) ^ 0x08080808u) - 0x08080808u;
        u32_t const high = (((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u) - 0x08080808u;
        return {low, high};
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        u32_t const words_a[4] = {a.x, a.y, a.z, a.w};
        u32_t const words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            uint2 const unpacked_a = unpack(words_a[word]), unpacked_b = unpack(words_b[word]);
            accumulator = __dp4a(static_cast<int>(unpacked_a.x), static_cast<int>(unpacked_b.x), accumulator);
            accumulator = __dp4a(static_cast<int>(unpacked_a.y), static_cast<int>(unpacked_b.y), accumulator);
        }
        return accumulator;
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0, 0}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            uint2 const unpacked = unpack(words[word]);
            accumulator.x = __dp4a(static_cast<int>(unpacked.x), 0x01010101, accumulator.x);
            accumulator.x = __dp4a(static_cast<int>(unpacked.y), 0x01010101, accumulator.x);
            accumulator.y = __dp4a(static_cast<int>(unpacked.x), static_cast<int>(unpacked.x), accumulator.y);
            accumulator.y = __dp4a(static_cast<int>(unpacked.y), static_cast<int>(unpacked.y), accumulator.y);
        }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - static_cast<f32_t>(dot);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct i4x2_ampere_angular_kit_t : i4x2_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(int dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - static_cast<f32_t>(dot) * query_state.rsqrt *
                          (norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f);
    }
};
struct i4x2_ampere_sqeuclidean_kit_t : i4x2_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(int dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return static_cast<f32_t>(query_state.sumsq) + static_cast<f32_t>(norms.sumsq) - 2.0f * static_cast<f32_t>(dot);
    }
};
struct i4x2_ampere_euclidean_kit_t : i4x2_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

/** The @c u4x2 dot base - the unsigned unpack, no sign bias, unsigned records. */
struct u4x2_ampere_inner_kit_t {
    using scalar_t = nk::u4x2_t;
    using fragment_t = uint4;
    using accumulator_t = u32_t;
    using dot_t = u32_t;
    struct norms_t {
        u32_t sum, sumsq;
    };
    using norms_accumulator_t = uint2;
    struct query_state_t {
        f32_t rsqrt;
        u32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ uint2 unpack(u32_t word) noexcept {
        return {word & 0x0F0F0F0Fu, (word >> 4) & 0x0F0F0F0Fu};
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        u32_t const words_a[4] = {a.x, a.y, a.z, a.w};
        u32_t const words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            uint2 const unpacked_a = unpack(words_a[word]), unpacked_b = unpack(words_b[word]);
            accumulator = __dp4a(unpacked_a.x, unpacked_b.x, accumulator);
            accumulator = __dp4a(unpacked_a.y, unpacked_b.y, accumulator);
        }
        return accumulator;
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0, 0}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        u32_t const words[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (u32_t word = 0; word != 4; ++word) {
            uint2 const unpacked = unpack(words[word]);
            accumulator.x = __dp4a(unpacked.x, 0x01010101u, accumulator.x);
            accumulator.x = __dp4a(unpacked.y, 0x01010101u, accumulator.x);
            accumulator.y = __dp4a(unpacked.x, unpacked.x, accumulator.y);
            accumulator.y = __dp4a(unpacked.y, unpacked.y, accumulator.y);
        }
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.y};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f, norms.sumsq};
    }

    /** The inner base IS the dot-kernel kit, like the CPU's: similarity flips into
     *  a distance as @b 1-dot, monotone decreasing and negative-safe. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - static_cast<f32_t>(dot);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

struct u4x2_ampere_angular_kit_t : u4x2_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(u32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return 1.0f - static_cast<f32_t>(dot) * query_state.rsqrt *
                          (norms.sumsq ? rsqrtf(static_cast<f32_t>(norms.sumsq)) : 0.0f);
    }
};
struct u4x2_ampere_sqeuclidean_kit_t : u4x2_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(u32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return static_cast<f32_t>(query_state.sumsq) + static_cast<f32_t>(norms.sumsq) - 2.0f * static_cast<f32_t>(dot);
    }
};
struct u4x2_ampere_euclidean_kit_t : u4x2_ampere_sqeuclidean_kit_t {
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return sqrtf(ranked);
    }
};

#pragma endregion

#pragma region Packed Bits

/** The @c u1x8 base over 128-bit fragments: the "dot" is the intersection popcount
 *  via two 64-bit @c popc per fragment - the counter, not the logic, limits throughput,
 *  so XOR/AND fold in front of as few @c popc as possible. Unsigned records: popcounts
 *  are counts; the norms store each row's set-bit count, the currency of Jaccard's
 *  union, mirrored into both fields. */
struct u1x8_ampere_inner_kit_t {
    using scalar_t = nk::u1x8_t;
    using fragment_t = uint4;
    using accumulator_t = u32_t;
    using dot_t = u32_t;
    struct norms_t {
        u32_t sum, sumsq;
    };
    using norms_accumulator_t = uint2;
    struct query_state_t {
        f32_t rsqrt;
        u32_t sumsq;
    };

    inline static constexpr u32_t fragment_bytes_k = 16;
    inline static constexpr u32_t max_gathers_in_flight_k = 8;
    inline static constexpr bool has_norms_k = true;

    static __device__ __forceinline__ u32_t intersect_count(fragment_t a, fragment_t b) noexcept {
        u64_t const low = (static_cast<u64_t>(a.y) << 32 | a.x) & (static_cast<u64_t>(b.y) << 32 | b.x);
        u64_t const high = (static_cast<u64_t>(a.w) << 32 | a.z) & (static_cast<u64_t>(b.w) << 32 | b.z);
        return static_cast<u32_t>(__popcll(low) + __popcll(high));
    }

    static __device__ __forceinline__ accumulator_t zero_accumulator() noexcept { return 0; }
    static __device__ __forceinline__ accumulator_t accumulate_dot(fragment_t a, fragment_t b,
                                                                   accumulator_t accumulator) noexcept {
        return accumulator + intersect_count(a, b);
    }
    static __device__ __forceinline__ dot_t partial_to_dot(accumulator_t accumulator) noexcept { return accumulator; }

    static __device__ __forceinline__ norms_accumulator_t zero_norms_accumulator() noexcept { return {0, 0}; }
    static __device__ __forceinline__ norms_accumulator_t accumulate_norms(fragment_t a,
                                                                           norms_accumulator_t accumulator) noexcept {
        accumulator.x += intersect_count(a, a);
        return accumulator;
    }
    static __device__ __forceinline__ norms_t partials_to_norms(norms_accumulator_t accumulator) noexcept {
        return {accumulator.x, accumulator.x};
    }
    static __device__ __forceinline__ query_state_t make_query_state(norms_t norms) noexcept {
        return {0.0f, norms.sum};
    }

    /** The inner base IS the dot-kernel kit: for bit vectors the dot is the
     *  intersection popcount, flipped into a distance as @b 1-dot. */
    static __device__ __forceinline__ dense_distance_t distance(dot_t dot, query_state_t, norms_t) noexcept {
        return 1.0f - static_cast<f32_t>(dot);
    }
    static __device__ __forceinline__ dense_distance_t convert_to_distance(dense_distance_t ranked) noexcept {
        return ranked;
    }
};

/** Hamming distance: differing bits = |a| + |b| - 2·intersection - never negative,
 *  since the intersection is bounded by either operand's count - so the stored
 *  popcount norms turn the intersection dot into the XOR count without a second pass. */
struct u1x8_ampere_hamming_kit_t : u1x8_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(u32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        return static_cast<f32_t>(query_state.sumsq + norms.sum - 2 * dot);
    }
};

/** Jaccard distance: 1 - intersection / union, union = |a| + |b| - intersection;
 *  two empty sets are identical, distance zero. */
struct u1x8_ampere_jaccard_kit_t : u1x8_ampere_inner_kit_t {
    static __device__ __forceinline__ dense_distance_t distance(u32_t dot, query_state_t query_state,
                                                                norms_t norms) noexcept {
        u32_t const unions = query_state.sumsq + norms.sum - dot;
        return unions ? 1.0f - static_cast<f32_t>(dot) / static_cast<f32_t>(unions) : 0.0f;
    }
};

#pragma endregion

/** One selectable generation row of a kit chain - the CPU @c chain_layer's twin.
 *  A row is admissible when its architecture requirement clears both gates: the
 *  caller's ceiling - the ask - and this compilation pass's target - the can. */
template <cuda_arch_t required_arch_, cuda_arch_t arch_ceiling_, typename kit_>
struct dense_cuda_arch_layer {
    static constexpr bool available_k = required_arch_ <= arch_ceiling_ && required_arch_ <= dense_cuda_target_arch_k;
    static constexpr cuda_arch_t required_arch_k = required_arch_;
    using kit_t = kit_;
};

/** The CPU's @c first_available_kit, verbatim in spirit: the first admissible layer
 *  wins, the floor backs them all, and @c picked_arch_k names the winner for reports. */
template <typename fallback_, typename... layers_>
struct dense_cuda_first_available_kit {
    using type = fallback_;
    static constexpr cuda_arch_t picked_arch_k = cuda_arch_ampere_k;
};
template <typename fallback_, typename head_, typename... tail_>
struct dense_cuda_first_available_kit<fallback_, head_, tail_...> {
    using next_t = dense_cuda_first_available_kit<fallback_, tail_...>;
    using type = std::conditional_t<head_::available_k, typename head_::kit_t, typename next_t::type>;
    static constexpr cuda_arch_t picked_arch_k = head_::available_k ? head_::required_arch_k : next_t::picked_arch_k;
};

/** Maps a (dtype, metric) pair onto its Ampere-floor device kit; unsupported pairs
 *  fail the index instantiation with a one-line constraint error. Rows sort like the
 *  kit families: widest types first, packed bits last. */
template <nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_>
struct dense_cuda_kit_for;

/**
 *  @brief The dispatch entry every index instantiation resolves through: (dtype, kernel,
 *      ceiling) to one kit, side-by-side comparable like the CPU chains - two ceilings in
 *      one binary run different rows on the same device.
 *
 *  The generic body serves the floor from @c dense_cuda_kit_for; a cell earns its own
 *  specialization enumerating @c dense_cuda_arch_layer rows the moment a newer generation lands, e.g.:
 *
 *  @code{.cpp}
 *  template <cuda_arch_t arch_ceiling_>
 *  struct dense_cuda_kit_chain<nk_bf16_k, nk_kernel_angular_k, arch_ceiling_> {
 *      using selected_t = dense_cuda_first_available_kit<
 *          bf16_ampere_angular_kit_t,
 *          dense_cuda_arch_layer<cuda_arch_blackwell_k, arch_ceiling_, bf16_blackwell_angular_kit_t>>;
 *      using kit_t = typename selected_t::type;
 *      static constexpr cuda_arch_t picked_arch_k = selected_t::picked_arch_k;
 *  };
 *  @endcode
 */
template <nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, cuda_arch_t arch_ceiling_ = cuda_arch_blackwell_k>
struct dense_cuda_kit_chain {
    using kit_t = typename dense_cuda_kit_for<vector_dtype_, kernel_kind_>::kit_t;
    static constexpr cuda_arch_t picked_arch_k = cuda_arch_ampere_k;
};

template <>
struct dense_cuda_kit_for<nk_f32_k, nk_kernel_dot_k> {
    using kit_t = f32_ampere_inner_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f32_k, nk_kernel_angular_k> {
    using kit_t = f32_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f32_k, nk_kernel_sqeuclidean_k> {
    using kit_t = f32_ampere_sqeuclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f32_k, nk_kernel_euclidean_k> {
    using kit_t = f32_ampere_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f16_k, nk_kernel_dot_k> {
    using kit_t = f16_ampere_inner_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f16_k, nk_kernel_angular_k> {
    using kit_t = f16_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f16_k, nk_kernel_sqeuclidean_k> {
    using kit_t = f16_ampere_sqeuclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_f16_k, nk_kernel_euclidean_k> {
    using kit_t = f16_ampere_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_bf16_k, nk_kernel_dot_k> {
    using kit_t = bf16_ampere_inner_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_bf16_k, nk_kernel_angular_k> {
    using kit_t = bf16_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_bf16_k, nk_kernel_sqeuclidean_k> {
    using kit_t = bf16_ampere_sqeuclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_bf16_k, nk_kernel_euclidean_k> {
    using kit_t = bf16_ampere_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_i8_k, nk_kernel_dot_k> {
    using kit_t = i8_ampere_inner_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_i8_k, nk_kernel_angular_k> {
    using kit_t = i8_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_i8_k, nk_kernel_sqeuclidean_k> {
    using kit_t = i8_ampere_sqeuclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_i8_k, nk_kernel_euclidean_k> {
    using kit_t = i8_ampere_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u8_k, nk_kernel_dot_k> {
    using kit_t = u8_ampere_inner_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u8_k, nk_kernel_angular_k> {
    using kit_t = u8_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u8_k, nk_kernel_sqeuclidean_k> {
    using kit_t = u8_ampere_sqeuclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u8_k, nk_kernel_euclidean_k> {
    using kit_t = u8_ampere_euclidean_kit_t;
};
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
template <>
struct dense_cuda_kit_for<nk_e4m3_k, nk_kernel_angular_k> {
    using kit_t = e4m3_ada_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_e4m3_k, nk_kernel_euclidean_k> {
    using kit_t = e4m3_ada_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_e5m2_k, nk_kernel_angular_k> {
    using kit_t = e5m2_ada_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_e5m2_k, nk_kernel_euclidean_k> {
    using kit_t = e5m2_ada_euclidean_kit_t;
};
#endif
template <>
struct dense_cuda_kit_for<nk_i4_k, nk_kernel_angular_k> {
    using kit_t = i4x2_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_i4_k, nk_kernel_euclidean_k> {
    using kit_t = i4x2_ampere_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u4_k, nk_kernel_angular_k> {
    using kit_t = u4x2_ampere_angular_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u4_k, nk_kernel_euclidean_k> {
    using kit_t = u4x2_ampere_euclidean_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u1_k, nk_kernel_dot_k> {
    using kit_t = u1x8_ampere_inner_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u1_k, nk_kernel_hamming_k> {
    using kit_t = u1x8_ampere_hamming_kit_t;
};
template <>
struct dense_cuda_kit_for<nk_u1_k, nk_kernel_jaccard_k> {
    using kit_t = u1x8_ampere_jaccard_kit_t;
};

} // namespace usearch
} // namespace unum

#endif // defined(__CUDACC__)
