/**
 *  @file include/usearch/dense/vram_apple_metric.metal
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Metal metric kits for the dense Apple index - the MSL twin of @c vram_nvidia_metric.cuh.
 *
 *  A kit binds one storage dtype: it accumulates the dot of two 16-byte fragments, a fragment's
 *  {sum, sum of squares} norms, and finalizes a distance "through the dot" exactly as the CUDA kits
 *  do, so a graph saved on one GPU reads back on another with the same norms. No Apple GPU this
 *  runtime reaches has fp8 or fp6 arithmetic, so the minifloats widen to @c half per byte, as
 *  NumKong's @c apple10.metal does. The host prepends this file to @c vram_apple_index.metal with a
 *  @c metal_kit_t alias naming the kit, and compiles the pair at first use.
 */
#include <metal_stdlib>
using namespace metal;

namespace unum::usearch::dense {

/** The metrics the host lowers its kernel kind onto: the walk ranks Euclidean by its square. */
enum metal_metric_t : uint {
    metal_dot_k = 0,
    metal_angular_k = 1,
    metal_sqeuclidean_k = 2,
    metal_euclidean_k = 3,
    metal_hamming_k = 4,
    metal_jaccard_k = 5,
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
inline float metal_convert(uint metric, float ranked) {
    return metric == metal_euclidean_k ? sqrt(max(ranked, 0.0f)) : ranked;
}

/** The shape every floating kit shares: widen each word to four floats, then multiply-add. */
template <typename widening_type_>
struct metal_floats_quads_kit {
    using dot_t = float;
    using norms_t = float2;

    static float dot(uint4 a, uint4 b, float sum) {
        for (uint word = 0; word != 4; ++word)
            sum += metal::dot(widening_type_::widen(a[word]), widening_type_::widen(b[word]));
        return sum;
    }
    static float2 moments(uint4 a, float2 moments) {
        for (uint word = 0; word != 4; ++word) {
            float4 const values = widening_type_::widen(a[word]);
            moments += float2(values.x + values.y + values.z + values.w, metal::dot(values, values));
        }
        return moments;
    }
    static float2 norms(float2 moments) { return moments; }
    static float distance(uint metric, float dot, float2 query, float2 target) {
        return metal_distance(metric, dot, query, target);
    }
};

/** The values a word carries as four lanes, the 16-bit floats' two padded with zeros. */
struct f16_metal_widening_t {
    static float4 widen(uint word) { return float4(float2(as_type<half2>(word)), 0.0f, 0.0f); }
};
struct bf16_metal_widening_t {
    static float4 widen(uint word) {
        return float4(as_type<float>(word << 16), as_type<float>(word & 0xFFFF0000u), 0, 0);
    }
};

/** E4M3FN: bias 7, the all-ones codes NaN. */
struct e4m3_metal_widening_t {
    static float4 widen(uint word) {
        ushort4 const codes = metal_bytes(word);
        ushort4 const bits = ((codes & ushort(0x7F)) << ushort(7)) | ((codes & ushort(0x80)) << ushort(8));
        return float4(select(as_type<half4>(bits) * half(256), half4(NAN), (codes & ushort(0x7F)) == ushort(0x7F)));
    }
};

/** E5M2 is the top byte of an IEEE half. */
struct e5m2_metal_widening_t {
    static float4 widen(uint word) { return float4(as_type<half4>(metal_bytes(word) << ushort(8))); }
};

/** E3M2, OCP FP6 in the low six bits of a byte: bias 3. */
struct e3m2_metal_widening_t {
    static float4 widen(uint word) {
        ushort4 const codes = metal_bytes(word);
        ushort4 const bits = ((codes & ushort(0x1F)) << ushort(8)) | ((codes & ushort(0x20)) << ushort(10));
        return float4(as_type<half4>(bits) * half(4096));
    }
};

/** E2M3, OCP FP6 in the low six bits of a byte: bias 1. */
struct e2m3_metal_widening_t {
    static float4 widen(uint word) {
        ushort4 const codes = metal_bytes(word);
        ushort4 const bits = ((codes & ushort(0x1F)) << ushort(7)) | ((codes & ushort(0x20)) << ushort(10));
        return float4(as_type<half4>(bits) * half(16384));
    }
};

using f16_metal_quads_kit_t = metal_floats_quads_kit<f16_metal_widening_t>;
using bf16_metal_quads_kit_t = metal_floats_quads_kit<bf16_metal_widening_t>;
using e4m3_metal_quads_kit_t = metal_floats_quads_kit<e4m3_metal_widening_t>;
using e5m2_metal_quads_kit_t = metal_floats_quads_kit<e5m2_metal_widening_t>;
using e3m2_metal_quads_kit_t = metal_floats_quads_kit<e3m2_metal_widening_t>;
using e2m3_metal_quads_kit_t = metal_floats_quads_kit<e2m3_metal_widening_t>;

/** Signed bytes: integer products accumulate exactly. */
struct i8_metal_quads_kit_t {
    using dot_t = int;
    using norms_t = int2;

    static int dot(uint4 a, uint4 b, int sum) {
        for (uint word = 0; word != 4; ++word) {
            int4 const left = int4(as_type<char4>(a[word])), right = int4(as_type<char4>(b[word]));
            sum += left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
        }
        return sum;
    }
    static int2 moments(uint4 a, int2 moments) {
        for (uint word = 0; word != 4; ++word) {
            int4 const values = int4(as_type<char4>(a[word]));
            moments += int2(values.x + values.y + values.z + values.w,
                            values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w);
        }
        return moments;
    }
    static int2 norms(int2 moments) { return moments; }
    static float distance(uint metric, int dot, int2 query, int2 target) {
        return metal_distance(metric, dot, query, target);
    }
};

/** Unsigned bytes, accumulating in unsigned words with defined wrapping, as the CUDA kit does. */
struct u8_metal_quads_kit_t {
    using dot_t = uint;
    using norms_t = uint2;

    static uint dot(uint4 a, uint4 b, uint sum) {
        for (uint word = 0; word != 4; ++word) {
            uint4 const left = uint4(as_type<uchar4>(a[word])), right = uint4(as_type<uchar4>(b[word]));
            sum += left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
        }
        return sum;
    }
    static uint2 moments(uint4 a, uint2 moments) {
        for (uint word = 0; word != 4; ++word) {
            uint4 const values = uint4(as_type<uchar4>(a[word]));
            moments += uint2(values.x + values.y + values.z + values.w,
                             values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w);
        }
        return moments;
    }
    static uint2 norms(uint2 moments) { return moments; }
    static float distance(uint metric, uint dot, uint2 query, uint2 target) {
        return metal_distance(metric, dot, query, target);
    }
};

/** Signed nibbles, each sign-extended from four bits; the pairing within a byte is irrelevant to a dot. */
struct i4x2_metal_quads_kit_t {
    using dot_t = int;
    using norms_t = int2;

    static int nibble(uint word, uint index) { return (int((word >> (4 * index)) & 15u) ^ 8) - 8; }
    static int dot(uint4 a, uint4 b, int sum) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) sum += nibble(a[word], index) * nibble(b[word], index);
        return sum;
    }
    static int2 moments(uint4 a, int2 moments) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) {
                int const value = nibble(a[word], index);
                moments += int2(value, value * value);
            }
        return moments;
    }
    static int2 norms(int2 moments) { return moments; }
    static float distance(uint metric, int dot, int2 query, int2 target) {
        return metal_distance(metric, dot, query, target);
    }
};

/** Unsigned nibbles. */
struct u4x2_metal_quads_kit_t {
    using dot_t = uint;
    using norms_t = uint2;

    static uint nibble(uint word, uint index) { return (word >> (4 * index)) & 15u; }
    static uint dot(uint4 a, uint4 b, uint sum) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) sum += nibble(a[word], index) * nibble(b[word], index);
        return sum;
    }
    static uint2 moments(uint4 a, uint2 moments) {
        for (uint word = 0; word != 4; ++word)
            for (uint index = 0; index != 8; ++index) {
                uint const value = nibble(a[word], index);
                moments += uint2(value, value * value);
            }
        return moments;
    }
    static uint2 norms(uint2 moments) { return moments; }
    static float distance(uint metric, uint dot, uint2 query, uint2 target) {
        return metal_distance(metric, dot, query, target);
    }
};

/** Packed bits: the "dot" is the intersection popcount, and the norms mirror each row's set-bit count into both
 *  fields, the currency of Jaccard's union. */
struct u1x8_metal_quads_kit_t {
    using dot_t = uint;
    using norms_t = uint2;

    static uint dot(uint4 a, uint4 b, uint sum) {
        uint4 const both = popcount(a & b);
        return sum + both.x + both.y + both.z + both.w;
    }
    static uint2 moments(uint4 a, uint2 moments) {
        uint4 const set = popcount(a);
        moments.x += set.x + set.y + set.z + set.w;
        return moments;
    }
    static uint2 norms(uint2 moments) { return uint2(moments.x, moments.x); }
    static float distance(uint metric, uint dot, uint2 query, uint2 target) {
        if (metric == metal_dot_k) return 1.0f - float(dot);
        if (metric == metal_hamming_k) return float(query.x + target.x - 2 * dot);
        uint const unions = query.x + target.x - dot;
        return unions ? 1.0f - float(dot) / float(unions) : 0.0f;
    }
};

} // namespace unum::usearch::dense
