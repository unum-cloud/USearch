/**
 *  @file       turboquant_codec.hpp
 *  @brief      TurboQuant encoder: Lloyd-Max codebooks + bit-packing.
 *
 *  Contains pre-computed optimal scalar quantizer codebooks for the standard
 *  Gaussian N(0,1) distribution (which the HD³-rotated coordinates approximate
 *  in high dimensions), plus encoding and decoding routines.
 *
 *  Encoding is on the CRITICAL PATH (called during add/search).
 *  Decoding is OFF the critical path (tests/export only).
 */
#pragma once
#include <usearch/turboquant_rotation.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace unum {
namespace usearch {

// ===================================================================
// Pre-computed Lloyd-Max codebooks for N(0,1)
// ===================================================================
//
// After HD³ rotation, each coordinate of a unit vector follows
// approximately N(0, 1/d).  Scaling by sqrt(d) yields N(0,1).
//
// The centroids and decision boundaries below are the optimal
// Lloyd-Max solution for N(0,1).  At encode time we scale by
// sqrt(padded_dim); at decode time we scale back by 1/sqrt(padded_dim).
// ===================================================================

/// @brief  2-bit codebook (4 centroids) for N(0,1).
struct tq_codebook_2bit {
    static constexpr unsigned bit_width = 2;
    static constexpr unsigned num_centroids = 4;

    /// Centroids in ascending order.
    static constexpr float centroids[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};

    /// Decision boundaries (num_centroids - 1).
    static constexpr float boundaries[3] = {-0.9816f, 0.0f, 0.9816f};

    /// Symmetric product LUT: lut[i][j] = centroids[i] * centroids[j].
    /// Used for symmetric compressed × compressed inner product.
    static constexpr float product_lut[4][4] = {
        {2.2813f, 0.6839f, -0.6839f, -2.2813f}, //
        {0.6839f, 0.2050f, -0.2050f, -0.6839f}, //
        {-0.6839f, -0.2050f, 0.2050f, 0.6839f}, //
        {-2.2813f, -0.6839f, 0.6839f, 2.2813f}, //
    };
};

/// @brief  4-bit codebook (16 centroids) for N(0,1).
struct tq_codebook_4bit {
    static constexpr unsigned bit_width = 4;
    static constexpr unsigned num_centroids = 16;

    /// Centroids in ascending order.
    static constexpr float centroids[16] = {
        -2.7326f, -2.0691f, -1.6180f, -1.2562f, //
        -0.9424f, -0.6568f, -0.3881f, -0.1284f, //
        0.1284f, 0.3881f, 0.6568f, 0.9424f,      //
        1.2562f, 1.6180f, 2.0691f, 2.7326f,       //
    };

    /// Decision boundaries (num_centroids - 1).
    static constexpr float boundaries[15] = {
        -2.4009f, -1.8436f, -1.4371f, -1.0993f, //
        -0.7996f, -0.5224f, -0.2582f, 0.0f,      //
        0.2582f, 0.5224f, 0.7996f, 1.0993f,       //
        1.4371f, 1.8436f, 2.4009f,                 //
    };
};

// ===================================================================
// Compressed vector size calculation
// ===================================================================

/**
 *  @brief  Size in bytes of a TurboQuant-compressed vector.
 *  @param  padded_dim  Padded dimension (power of 2).
 *  @param  bit_width   Bits per coordinate (2 or 4).
 */
inline std::size_t tq_compressed_bytes(std::size_t padded_dim, unsigned bit_width) noexcept {
    return (padded_dim * bit_width + 7) / 8;
}

// ===================================================================
// 2-bit encode / decode
// ===================================================================

/**
 *  @brief  Find the nearest centroid index for a scaled coordinate.
 *          Branchless version using the 3 decision boundaries.
 */
inline std::uint8_t tq_quantize_2bit(float scaled) noexcept {
    // boundaries = {-0.9816, 0.0, 0.9816}
    // Result: 0 if scaled < -0.9816
    //         1 if -0.9816 <= scaled < 0.0
    //         2 if 0.0 <= scaled < 0.9816
    //         3 if scaled >= 0.9816
    std::uint8_t idx = 0;
    idx += (scaled >= tq_codebook_2bit::boundaries[0]) ? 1 : 0;
    idx += (scaled >= tq_codebook_2bit::boundaries[1]) ? 1 : 0;
    idx += (scaled >= tq_codebook_2bit::boundaries[2]) ? 1 : 0;
    return idx;
}

/**
 *  @brief  Encode a float vector to 2-bit TurboQuant compressed form.
 *
 *  Assumes the input is a UNIT VECTOR (||vector|| ≈ 1).
 *
 *  @param  vector    Input float vector of length @p original_dim.
 *  @param  rotation  HD³ rotation (must be initialized for @p original_dim).
 *  @param  output    Output buffer of length tq_compressed_bytes(padded_dim, 2).
 *  @param  scratch   Temporary float buffer of length rotation.padded_dim().
 *                    Caller must allocate this (avoids per-call allocation).
 */
inline void tq_encode_2bit(float const* vector, turboquant_rotation_t const& rotation, //
                           byte_t* output, float* scratch) noexcept {
    std::size_t const pd = rotation.padded_dim();
    float const sqrt_d = std::sqrt(static_cast<float>(pd));

    // Step 1: Rotate.
    rotation.forward(vector, scratch);

    // Step 2: Quantize each coordinate and pack 4 indices per byte.
    std::size_t byte_idx = 0;
    std::size_t bit_pos = 0;
    byte_t current_byte = 0;

    for (std::size_t i = 0; i < pd; ++i) {
        float scaled = scratch[i] * sqrt_d;
        std::uint8_t idx = tq_quantize_2bit(scaled);

        current_byte |= static_cast<byte_t>(idx << bit_pos);
        bit_pos += 2;

        if (bit_pos == 8) {
            output[byte_idx++] = current_byte;
            current_byte = 0;
            bit_pos = 0;
        }
    }
    // Flush remaining bits (shouldn't happen if pd is power of 2 and >= 4).
    if (bit_pos > 0)
        output[byte_idx] = current_byte;
}

/**
 *  @brief  Find the nearest centroid index for a 4-bit quantization.
 *          Linear scan over 15 boundaries (fast for 16 entries).
 */
inline std::uint8_t tq_quantize_4bit(float scaled) noexcept {
    std::uint8_t idx = 0;
    for (unsigned b = 0; b < 15; ++b)
        idx += (scaled >= tq_codebook_4bit::boundaries[b]) ? 1 : 0;
    return idx;
}

/**
 *  @brief  Encode a float vector to 4-bit TurboQuant compressed form.
 *
 *  Assumes the input is a UNIT VECTOR (||vector|| ≈ 1).
 */
inline void tq_encode_4bit(float const* vector, turboquant_rotation_t const& rotation, //
                           byte_t* output, float* scratch) noexcept {
    std::size_t const pd = rotation.padded_dim();
    float const sqrt_d = std::sqrt(static_cast<float>(pd));

    // Step 1: Rotate.
    rotation.forward(vector, scratch);

    // Step 2: Quantize each coordinate and pack 2 indices per byte.
    for (std::size_t i = 0; i < pd; i += 2) {
        float scaled_lo = scratch[i] * sqrt_d;
        float scaled_hi = (i + 1 < pd) ? scratch[i + 1] * sqrt_d : 0.0f;

        std::uint8_t idx_lo = tq_quantize_4bit(scaled_lo);
        std::uint8_t idx_hi = tq_quantize_4bit(scaled_hi);

        output[i / 2] = static_cast<byte_t>(idx_lo | (idx_hi << 4));
    }
}

// ===================================================================
// Decode (for testing / export only — NOT on the critical path)
// ===================================================================

/**
 *  @brief  Decode a 2-bit compressed vector back to float.
 *          This is NOT used during search; only for testing roundtrip quality.
 */
inline void tq_decode_2bit(byte_t const* compressed, turboquant_rotation_t const& rotation, //
                           float* output) noexcept {
    std::size_t const pd = rotation.padded_dim();
    float const inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(pd));

    // Reconstruct rotated vector from codebook.
    std::vector<float> rotated(pd);
    for (std::size_t i = 0; i < pd; ++i) {
        std::size_t byte_idx = i / 4;
        std::size_t bit_pos = (i % 4) * 2;
        std::uint8_t idx = (compressed[byte_idx] >> bit_pos) & 0x03;
        rotated[i] = tq_codebook_2bit::centroids[idx] * inv_sqrt_d;
    }

    // Inverse rotate.
    rotation.inverse(rotated.data(), output);
}

/**
 *  @brief  Decode a 4-bit compressed vector back to float.
 *          This is NOT used during search; only for testing roundtrip quality.
 */
inline void tq_decode_4bit(byte_t const* compressed, turboquant_rotation_t const& rotation, //
                           float* output) noexcept {
    std::size_t const pd = rotation.padded_dim();
    float const inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(pd));

    std::vector<float> rotated(pd);
    for (std::size_t i = 0; i < pd; ++i) {
        std::size_t byte_idx = i / 2;
        std::uint8_t idx = (i % 2 == 0) ? (compressed[byte_idx] & 0x0F) : ((compressed[byte_idx] >> 4) & 0x0F);
        rotated[i] = tq_codebook_4bit::centroids[idx] * inv_sqrt_d;
    }

    rotation.inverse(rotated.data(), output);
}

} // namespace usearch
} // namespace unum
