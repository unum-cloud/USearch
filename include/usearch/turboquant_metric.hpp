/**
 *  @file       turboquant_metric.hpp
 *  @brief      Distance functions for TurboQuant-compressed vectors.
 *
 *  Provides symmetric inner-product computation on pairs of compressed
 *  vectors (both sides quantized).  The query rotation is performed once
 *  per search call and cached; the metric itself only operates on packed
 *  index arrays.
 *
 *  Design:
 *  - Symmetric: compressed × compressed  (used by HNSW for both add & search)
 *  - LUT-based: a 4×4 (2-bit) or 16×16 (4-bit) product table maps every
 *    pair of centroid indices to the product centroid[i]×centroid[j].
 *  - The total IP = (1/padded_dim) × Σ LUT[idx_a[i]][idx_b[i]].
 *  - Distance is returned as  1 − IP  (lower = more similar), matching
 *    USearch's convention for ip_k / cos_k metrics.
 */
#pragma once
#include <usearch/turboquant_codec.hpp>

#include <cstddef>
#include <cstdint>

namespace unum {
namespace usearch {

// ===================================================================
// 2-bit symmetric inner product  (compressed × compressed)
// ===================================================================

/**
 *  @brief  Compute distance = 1 − IP between two 2-bit compressed vectors.
 *
 *  @param  a            First compressed vector.
 *  @param  b            Second compressed vector.
 *  @param  padded_dim   Padded dimension (power of 2).
 *  @return  1.0 − estimated_inner_product.
 *
 *  Uses the pre-computed product LUT from tq_codebook_2bit.
 *  Each byte contains 4 × 2-bit indices, processed together.
 */
inline float tq_distance_ip_2bit(byte_t const* a, byte_t const* b, std::size_t padded_dim) noexcept {
    float const inv_d = 1.0f / static_cast<float>(padded_dim);
    float sum = 0.0f;

    std::size_t const num_bytes = padded_dim / 4; // 4 indices per byte at 2 bits
    for (std::size_t byte_i = 0; byte_i < num_bytes; ++byte_i) {
        std::uint8_t const ba = static_cast<std::uint8_t>(a[byte_i]);
        std::uint8_t const bb = static_cast<std::uint8_t>(b[byte_i]);

        // Unpack 4 pairs of 2-bit indices and accumulate products.
        sum += tq_codebook_2bit::product_lut[(ba >> 0) & 3][(bb >> 0) & 3];
        sum += tq_codebook_2bit::product_lut[(ba >> 2) & 3][(bb >> 2) & 3];
        sum += tq_codebook_2bit::product_lut[(ba >> 4) & 3][(bb >> 4) & 3];
        sum += tq_codebook_2bit::product_lut[(ba >> 6) & 3][(bb >> 6) & 3];
    }

    float ip = sum * inv_d;
    return 1.0f - ip;
}

/**
 *  @brief  Compute distance = 1 − IP between two 2-bit compressed vectors.
 *          Version with the third argument being the padded dimension,
 *          matching metric_punned_t's array_array_size signature.
 */
inline float tq_metric_ip_2bit(std::size_t a_ptr, std::size_t b_ptr, std::size_t padded_dim) noexcept {
    return tq_distance_ip_2bit(reinterpret_cast<byte_t const*>(a_ptr), //
                               reinterpret_cast<byte_t const*>(b_ptr), padded_dim);
}

// ===================================================================
// 4-bit symmetric inner product  (compressed × compressed)
// ===================================================================

/**
 *  @brief  Compute distance = 1 − IP between two 4-bit compressed vectors.
 *
 *  Each byte contains 2 × 4-bit indices (lo nibble + hi nibble).
 *  Uses direct centroid multiplication (16 × 16 = 256 products).
 *  The product is computed on the fly rather than from a static LUT
 *  because 256 floats (1 KB) wouldn't be cache-friendly.
 *
 *  TODO: SIMD fast path with vpshufb for 4-bit → centroid lookup.
 */
inline float tq_distance_ip_4bit(byte_t const* a, byte_t const* b, std::size_t padded_dim) noexcept {
    float const inv_d = 1.0f / static_cast<float>(padded_dim);
    float sum = 0.0f;

    std::size_t const num_bytes = padded_dim / 2; // 2 indices per byte at 4 bits
    float const* cb = tq_codebook_4bit::centroids;

    for (std::size_t byte_i = 0; byte_i < num_bytes; ++byte_i) {
        std::uint8_t const ba = static_cast<std::uint8_t>(a[byte_i]);
        std::uint8_t const bb = static_cast<std::uint8_t>(b[byte_i]);

        // Low nibble.
        sum += cb[ba & 0x0F] * cb[bb & 0x0F];
        // High nibble.
        sum += cb[(ba >> 4) & 0x0F] * cb[(bb >> 4) & 0x0F];
    }

    float ip = sum * inv_d;
    return 1.0f - ip;
}

/**
 *  @brief  metric_punned_t-compatible wrapper for 4-bit IP distance.
 */
inline float tq_metric_ip_4bit(std::size_t a_ptr, std::size_t b_ptr, std::size_t padded_dim) noexcept {
    return tq_distance_ip_4bit(reinterpret_cast<byte_t const*>(a_ptr), //
                               reinterpret_cast<byte_t const*>(b_ptr), padded_dim);
}

// ===================================================================
// L2 squared from IP  (for normalized vectors:  L2² = 2(1 − IP))
// ===================================================================

inline float tq_distance_l2sq_2bit(byte_t const* a, byte_t const* b, std::size_t padded_dim) noexcept {
    float const ip_dist = tq_distance_ip_2bit(a, b, padded_dim); // 1 - IP
    return 2.0f * ip_dist;                                        // 2(1 - IP)
}

inline float tq_metric_l2sq_2bit(std::size_t a_ptr, std::size_t b_ptr, std::size_t pd) noexcept {
    return tq_distance_l2sq_2bit(reinterpret_cast<byte_t const*>(a_ptr), //
                                 reinterpret_cast<byte_t const*>(b_ptr), pd);
}

inline float tq_distance_l2sq_4bit(byte_t const* a, byte_t const* b, std::size_t padded_dim) noexcept {
    float const ip_dist = tq_distance_ip_4bit(a, b, padded_dim);
    return 2.0f * ip_dist;
}

inline float tq_metric_l2sq_4bit(std::size_t a_ptr, std::size_t b_ptr, std::size_t pd) noexcept {
    return tq_distance_l2sq_4bit(reinterpret_cast<byte_t const*>(a_ptr), //
                                 reinterpret_cast<byte_t const*>(b_ptr), pd);
}

// ===================================================================
// Cosine distance  (for normalized vectors: cos_dist = 1 − IP = ip_dist)
// ===================================================================

inline float tq_metric_cos_2bit(std::size_t a_ptr, std::size_t b_ptr, std::size_t pd) noexcept {
    return tq_metric_ip_2bit(a_ptr, b_ptr, pd);
}

inline float tq_metric_cos_4bit(std::size_t a_ptr, std::size_t b_ptr, std::size_t pd) noexcept {
    return tq_metric_ip_4bit(a_ptr, b_ptr, pd);
}

} // namespace usearch
} // namespace unum
