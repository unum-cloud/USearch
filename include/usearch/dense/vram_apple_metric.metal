/**
 *  @file include/usearch/dense/vram_apple_metric.metal
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Metal metric kits for the dense Apple index - the MSL twin of @c vram_nvidia_metric.cuh.
 *
 *  A kit binds one storage dtype to one Apple family, named like the CPU and CUDA kits -
 *  @b dtype_metal_quads_kit_t, where @c metal is NumKong's capability of the Apple7 floor. It
 *  accumulates the dot of two 16-byte fragments, a fragment's {sum, sum of squares} norms and its
 *  weighted values, and finalizes a distance "through the dot" exactly as the CUDA kits do, so a
 *  graph saved on one GPU reads back on another with the same norms. Each kit states the capability
 *  it runs under, @c capability_k, and @c compiled_k, and @c quads_ladder picks one per dtype
 *  through @c highest_kit, leaving room for Apple9 and Apple10 rungs above the floor. No Apple GPU
 *  this runtime reaches has fp8 or fp6 arithmetic, so the minifloats widen to @c half per byte, as
 *  NumKong's @c apple10.metal does.
 *
 *  The host compiles one library per dtype and Apple class from @c vram_apple_shared.h, a prelude
 *  spelling NumKong's dtypes and capabilities with the @c metal_dtype_k and @c metal_capability_k
 *  of that library, this file, then @c vram_apple_index.metal.
 */
#include <metal_stdlib>
using namespace metal;

namespace unum::usearch::dense {

/**
 *  @brief Picks the first of @p kits_ this library compiled and @p caps_ runs, else @p floor_ - a
 *      ladder's one closing @c ::type, the host's @c highest_kit with a typed capability set.
 *
 *  Every kit states its own needs: @c capability_k, the capability it runs under, and
 *  @c compiled_k, whether this library carries its body.
 */
template <nk_capability_t caps_, typename floor_, typename... kits_>
struct highest_kit {
    using type = floor_;
};

template <nk_capability_t caps_, typename floor_, typename head_, typename... tail_>
struct highest_kit<caps_, floor_, head_, tail_...> {
    using type = conditional_t<head_::compiled_k && (caps_ & head_::capability_k) != 0, head_,
                               typename highest_kit<caps_, floor_, tail_...>::type>;
};

/**
 *  @brief The Apple warp policy the kernels are written against: 32-lane simdgroups with 32-bit lane
 *      masks and device-scope atomics.
 *
 *  Metal's atomics are relaxed, so acquire and release are spelled as a relaxed access beside a
 *  device-scope fence, and it has no value matching, so lanes collapse equal values through a
 *  shuffle loop.
 */
struct metal_warp_t {
    using mask_t = uint;
    static constexpr constant uint width_k = metal_simdgroup_width_k;

    static mask_t ballot(bool predicate) { return mask_t(simd_vote::vote_t(simd_ballot(predicate))); }

    /** The lanes strictly below @p lane. */
    static mask_t below(uint lane) { return (1u << lane) - 1u; }
    static void sync() { simdgroup_barrier(mem_flags::mem_device); }

    /** Lanes are 32-bit everywhere, as on the CUDA and HIP policies; Metal takes 16-bit ones. */
    template <typename value_type_>
    static value_type_ shuffle(value_type_ value, uint source_lane) {
        return simd_shuffle(value, ushort(source_lane));
    }
    template <typename value_type_>
    static value_type_ shuffle_xor(value_type_ value, uint lane_mask) {
        return simd_shuffle_xor(value, ushort(lane_mask));
    }

    /** Metal shuffles no 64-bit word, so a scored node moves as its two halves. */
    static ulong shuffle(ulong value, uint source_lane) {
        return as_type<ulong>(simd_shuffle(as_type<uint2>(value), ushort(source_lane)));
    }
    static ulong shuffle_xor(ulong value, uint lane_mask) {
        return as_type<ulong>(simd_shuffle_xor(as_type<uint2>(value), ushort(lane_mask)));
    }

    /** The lanes holding the same @p value as this one: every lane meets every other. */
    static mask_t match_any(uint value) {
        mask_t peers = 0;
        for (uint lane = 0; lane != width_k; ++lane) peers |= mask_t(shuffle(value, lane) == value) << lane;
        return peers;
    }

    static void fence() { atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); }
    static uint load_acquire(device uint* word) {
        uint const value = atomic_load_explicit(atomic_(word), memory_order_relaxed);
        fence();
        return value;
    }
    static void store_relaxed(device uint* word, uint value) {
        atomic_store_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static void store_release(device uint* word, uint value) {
        fence();
        atomic_store_explicit(atomic_(word), value, memory_order_relaxed);
    }
    static uint exchange_acquire(device uint* word, uint value) {
        uint const previous = atomic_exchange_explicit(atomic_(word), value, memory_order_relaxed);
        fence();
        return previous;
    }

  private:
    static device atomic_uint* atomic_(device uint* word) { return (device atomic_uint*)word; }
};

/** The four bytes of @p word as the codes a minifloat widening takes. */
inline ushort4 metal_bytes(uint word) {
    return ushort4(word & 0xFFu, (word >> 8) & 0xFFu, (word >> 16) & 0xFFu, word >> 24);
}

/** A numeric kit's distance "through the dot", the CUDA kits' formulas line for line. */
template <typename dot_type_, typename norms_type_>
inline float metal_distance(uint metric, dot_type_ dot, norms_type_ query, norms_type_ target) {
    float const dot_value = float(dot);
    if (metric == metal_dot_k) return 1.0f - dot_value;
    if (metric == metal_angular_k) {
        float const query_sumsq = float(query.y), target_sumsq = float(target.y);
        if (query_sumsq > 0.0f && target_sumsq > 0.0f)
            return max(1.0f - dot_value * (rsqrt(query_sumsq) * rsqrt(target_sumsq)), 0.0f);
        return dot_value == 0.0f ? 0.0f : 1.0f;
    }
    return float(query.y) + float(target.y) - 2.0f * dot_value;
}

/** The output-boundary finalize: identity unless a metric owes a monotone transform. */
inline float metal_convert_to_distance(uint metric, float ranked) {
    return metric == metal_euclidean_k ? sqrt(max(ranked, 0.0f)) : ranked;
}

/** The two brain floats of @p word, the lower first, each the high half of a @c float. */
inline float2 metal_bf16x2_to_f32x2(uint word) {
    return float2(as_type<float>(word << 16), as_type<float>(word & 0xFFFF0000u));
}

/** The two IEEE halves of @p word, the lower first. */
inline float2 metal_f16x2_to_f32x2(uint word) { return float2(as_type<half2>(word)); }

/** E5M2 is the top byte of an IEEE half. */
inline float4 metal_e5m2x4_to_f32x4(uint word) { return float4(as_type<half4>(metal_bytes(word) << ushort(8))); }

/** E4M3FN: bias 7, the all-ones codes NaN. */
inline float4 metal_e4m3x4_to_f32x4(uint word) {
    ushort4 const codes = metal_bytes(word);
    ushort4 const bits = ((codes & ushort(0x7F)) << ushort(7)) | ((codes & ushort(0x80)) << ushort(8));
    return float4(select(as_type<half4>(bits) * half(256), half4(NAN), (codes & ushort(0x7F)) == ushort(0x7F)));
}

/** E3M2, OCP FP6 in the low six bits of a byte: bias 3. */
inline float4 metal_e3m2x4_to_f32x4(uint word) {
    ushort4 const codes = metal_bytes(word);
    ushort4 const bits = ((codes & ushort(0x1F)) << ushort(8)) | ((codes & ushort(0x20)) << ushort(10));
    return float4(as_type<half4>(bits) * half(4096));
}

/** E2M3, OCP FP6 in the low six bits of a byte: bias 1. */
inline float4 metal_e2m3x4_to_f32x4(uint word) {
    ushort4 const codes = metal_bytes(word);
    ushort4 const bits = ((codes & ushort(0x1F)) << ushort(7)) | ((codes & ushort(0x20)) << ushort(10));
    return float4(as_type<half4>(bits) * half(16384));
}

/** Brain floats, two per word, widened to @c float by their bits. */
struct bf16_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = float;
    using weighted_t = array<float, 8>;
    using norms_t = float2;
    using norms_accumulator_t = float2;

    static float accumulate_dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(metal_bf16x2_to_f32x2(a[word]), metal_bf16x2_to_f32x2(b[word]));
        return sum;
    }
    static float2 accumulate_norms(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float2 const values = metal_bf16x2_to_f32x2(a[word]);
            moments += float2(values.x + values.y, metal::dot(values, values));
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float2 const values = metal_bf16x2_to_f32x2(a[word]);
            sums[word * 2] += weight * values.x;
            sums[word * 2 + 1] += weight * values.y;
        }
        return sums;
    }
    static float2 partials_to_norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** IEEE halves, two per word. */
struct f16_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = float;
    using weighted_t = array<float, 8>;
    using norms_t = float2;
    using norms_accumulator_t = float2;

    static float accumulate_dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(metal_f16x2_to_f32x2(a[word]), metal_f16x2_to_f32x2(b[word]));
        return sum;
    }
    static float2 accumulate_norms(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float2 const values = metal_f16x2_to_f32x2(a[word]);
            moments += float2(values.x + values.y, metal::dot(values, values));
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float2 const values = metal_f16x2_to_f32x2(a[word]);
            sums[word * 2] += weight * values.x;
            sums[word * 2 + 1] += weight * values.y;
        }
        return sums;
    }
    static float2 partials_to_norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** E5M2 bytes, four per word, widened through @c half. */
struct e5m2_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = float;
    using weighted_t = array<float, 16>;
    using norms_t = float2;
    using norms_accumulator_t = float2;

    static float accumulate_dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(metal_e5m2x4_to_f32x4(a[word]), metal_e5m2x4_to_f32x4(b[word]));
        return sum;
    }
    static float2 accumulate_norms(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e5m2x4_to_f32x4(a[word]);
            moments += float2(values.x + values.y + values.z + values.w, metal::dot(values, values));
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e5m2x4_to_f32x4(a[word]);
            for (uint value = 0; value != 4; ++value) sums[word * 4 + value] += weight * values[value];
        }
        return sums;
    }
    static float2 partials_to_norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** E4M3FN bytes, four per word, widened through @c half. */
struct e4m3_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = float;
    using weighted_t = array<float, 16>;
    using norms_t = float2;
    using norms_accumulator_t = float2;

    static float accumulate_dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(metal_e4m3x4_to_f32x4(a[word]), metal_e4m3x4_to_f32x4(b[word]));
        return sum;
    }
    static float2 accumulate_norms(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e4m3x4_to_f32x4(a[word]);
            moments += float2(values.x + values.y + values.z + values.w, metal::dot(values, values));
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e4m3x4_to_f32x4(a[word]);
            for (uint value = 0; value != 4; ++value) sums[word * 4 + value] += weight * values[value];
        }
        return sums;
    }
    static float2 partials_to_norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** E3M2 bytes, four per word, widened through @c half. */
struct e3m2_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = float;
    using weighted_t = array<float, 16>;
    using norms_t = float2;
    using norms_accumulator_t = float2;

    static float accumulate_dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(metal_e3m2x4_to_f32x4(a[word]), metal_e3m2x4_to_f32x4(b[word]));
        return sum;
    }
    static float2 accumulate_norms(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e3m2x4_to_f32x4(a[word]);
            moments += float2(values.x + values.y + values.z + values.w, metal::dot(values, values));
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e3m2x4_to_f32x4(a[word]);
            for (uint value = 0; value != 4; ++value) sums[word * 4 + value] += weight * values[value];
        }
        return sums;
    }
    static float2 partials_to_norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** E2M3 bytes, four per word, widened through @c half. */
struct e2m3_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = float;
    using weighted_t = array<float, 16>;
    using norms_t = float2;
    using norms_accumulator_t = float2;

    static float accumulate_dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(metal_e2m3x4_to_f32x4(a[word]), metal_e2m3x4_to_f32x4(b[word]));
        return sum;
    }
    static float2 accumulate_norms(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e2m3x4_to_f32x4(a[word]);
            moments += float2(values.x + values.y + values.z + values.w, metal::dot(values, values));
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = metal_e2m3x4_to_f32x4(a[word]);
            for (uint value = 0; value != 4; ++value) sums[word * 4 + value] += weight * values[value];
        }
        return sums;
    }
    static float2 partials_to_norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** Signed bytes: integer products accumulate exactly. */
struct i8_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = array<float, 16>;
    using norms_t = int2;
    using norms_accumulator_t = int2;

    static int accumulate_dot(uint4 a, uint4 b, int sum) {
        for (uint word = 0; word != 4; ++word) {
            int4 const left = int4(as_type<char4>(a[word])), right = int4(as_type<char4>(b[word]));
            sum += left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
        }
        return sum;
    }
    static int2 accumulate_norms(uint4 a, int2 moments) {
        for (uint word = 0; word != 4; ++word) {
            int4 const values = int4(as_type<char4>(a[word]));
            moments += int2(values.x + values.y + values.z + values.w,
                            values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w);
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = float4(as_type<char4>(a[word]));
            for (uint byte = 0; byte != 4; ++byte) sums[word * 4 + byte] += weight * values[byte];
        }
        return sums;
    }
    static int2 partials_to_norms(int2 moments) { return moments; }
    static float distance(uint metric, int dot, int2 query, int2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** Signed nibbles, each sign-extended from four bits; the pairing within a byte is irrelevant to a dot, while
 *  the weighted columns put a byte's high nibble first, as the CUDA kit does. */
struct i4x2_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = int;
    using weighted_t = array<float, 32>;
    using norms_t = int2;
    using norms_accumulator_t = int2;

    static int nibble(uint word, uint index) { return (int((word >> (4 * index)) & 15u) ^ 8) - 8; }
    static int accumulate_dot(uint4 a, uint4 b, int sum) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) sum += nibble(a[word], index) * nibble(b[word], index);
        return sum;
    }
    static int2 accumulate_norms(uint4 a, int2 moments) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) {
                int const value = nibble(a[word], index);
                moments += int2(value, value * value);
            }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index)
                sums[word * 8 + (index ^ 1u)] += weight * float(nibble(a[word], index));
        return sums;
    }
    static int2 partials_to_norms(int2 moments) { return moments; }
    static float distance(uint metric, int dot, int2 query, int2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** Unsigned bytes, accumulating in unsigned words with defined wrapping, as the CUDA kit does. */
struct u8_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = uint;
    using weighted_t = array<float, 16>;
    using norms_t = uint2;
    using norms_accumulator_t = uint2;

    static uint accumulate_dot(uint4 a, uint4 b, uint sum) {
        for (uint word = 0; word != 4; ++word) {
            uint4 const left = uint4(as_type<uchar4>(a[word])), right = uint4(as_type<uchar4>(b[word]));
            sum += left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
        }
        return sum;
    }
    static uint2 accumulate_norms(uint4 a, uint2 moments) {
        for (uint word = 0; word != 4; ++word) {
            uint4 const values = uint4(as_type<uchar4>(a[word]));
            moments += uint2(values.x + values.y + values.z + values.w,
                             values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w);
        }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = float4(as_type<uchar4>(a[word]));
            for (uint byte = 0; byte != 4; ++byte) sums[word * 4 + byte] += weight * values[byte];
        }
        return sums;
    }
    static uint2 partials_to_norms(uint2 moments) { return moments; }
    static float distance(uint metric, uint dot, uint2 query, uint2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** Unsigned nibbles, the weighted columns ordered as the signed kit's. */
struct u4x2_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = uint;
    using weighted_t = array<float, 32>;
    using norms_t = uint2;
    using norms_accumulator_t = uint2;

    static uint nibble(uint word, uint index) { return (word >> (4 * index)) & 15u; }
    static uint accumulate_dot(uint4 a, uint4 b, uint sum) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) sum += nibble(a[word], index) * nibble(b[word], index);
        return sum;
    }
    static uint2 accumulate_norms(uint4 a, uint2 moments) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) {
                uint const value = nibble(a[word], index);
                moments += uint2(value, value * value);
            }
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index)
                sums[word * 8 + (index ^ 1u)] += weight * float(nibble(a[word], index));
        return sums;
    }
    static uint2 partials_to_norms(uint2 moments) { return moments; }
    static float distance(uint metric, uint dot, uint2 query, uint2 target) {
        return metal_distance(metric, dot, query, target);
    }
    static float convert_to_distance(uint metric, float ranked) { return metal_convert_to_distance(metric, ranked); }
};

/** Packed bits: the "dot" is the intersection popcount, and the norms mirror each row's set-bit count into both
 *  fields, the currency of Jaccard's union; the weighted columns take each byte's most significant bit first. */
struct u1x8_metal_quads_kit_t {
    static constexpr constant nk_capability_t capability_k = nk_cap_metal_k;
    static constexpr constant bool compiled_k = true;
    using fragment_t = uint4;
    using dot_t = uint;
    using weighted_t = array<float, 128>;
    using norms_t = uint2;
    using norms_accumulator_t = uint2;

    static uint accumulate_dot(uint4 a, uint4 b, uint sum) {
        uint4 const both = popcount(a & b);
        return sum + both.x + both.y + both.z + both.w;
    }
    static uint2 accumulate_norms(uint4 a, uint2 moments) {
        uint4 const set = popcount(a);
        moments.x += set.x + set.y + set.z + set.w;
        return moments;
    }
    static weighted_t accumulate_weighted(uint4 a, float weight, weighted_t sums) {
        for (uint bit = 0; bit != 128; ++bit)
            sums[bit] += weight * float((a[bit / 32] >> (8 * (bit / 8 % 4) + 7 - bit % 8)) & 1u);
        return sums;
    }
    static uint2 partials_to_norms(uint2 moments) { return uint2(moments.x, moments.x); }
    static float distance(uint metric, uint dot, uint2 query, uint2 target) {
        if (metric == metal_dot_k) return 1.0f - float(dot);
        if (metric == metal_hamming_k) return float(query.x + target.x - 2 * dot);
        uint const unions = query.x + target.x - dot;
        return unions ? 1.0f - float(dot) / float(unions) : 0.0f;
    }
    static float convert_to_distance(uint, float ranked) { return ranked; }
};

#pragma region Ladders

/** The kit @p caps_ runs for each stored dtype, keyed by the dtype the library compiles for. */
template <nk_dtype_t dtype_, nk_capability_t caps_>
struct quads_ladder;

template <nk_capability_t caps_>
struct quads_ladder<nk_bf16_k, caps_> {
    using kit_t = typename highest_kit<caps_, bf16_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_f16_k, caps_> {
    using kit_t = typename highest_kit<caps_, f16_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_e5m2_k, caps_> {
    using kit_t = typename highest_kit<caps_, e5m2_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_e4m3_k, caps_> {
    using kit_t = typename highest_kit<caps_, e4m3_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_e3m2_k, caps_> {
    using kit_t = typename highest_kit<caps_, e3m2_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_e2m3_k, caps_> {
    using kit_t = typename highest_kit<caps_, e2m3_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_i8_k, caps_> {
    using kit_t = typename highest_kit<caps_, i8_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_i4_k, caps_> {
    using kit_t = typename highest_kit<caps_, i4x2_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_u8_k, caps_> {
    using kit_t = typename highest_kit<caps_, u8_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_u4_k, caps_> {
    using kit_t = typename highest_kit<caps_, u4x2_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

template <nk_capability_t caps_>
struct quads_ladder<nk_u1_k, caps_> {
    using kit_t = typename highest_kit<caps_, u1x8_metal_quads_kit_t>::type;
    static constexpr constant nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Ladders

} // namespace unum::usearch::dense
