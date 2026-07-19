/**
 *  @file       turboquant_rotation.hpp
 *  @brief      HD³ (randomized Hadamard) rotation for TurboQuant.
 *
 *  Implements the fast pseudo-random rotation used by TurboQuant to induce
 *  a near-Gaussian distribution on each coordinate.  Three rounds of
 *  (diagonal-sign-flip + normalized Walsh-Hadamard Transform) give a
 *  rotation quality close to a full random orthogonal matrix while running
 *  in O(d log d) time and requiring only O(d) extra memory (the signs).
 *
 *  The transform is fully deterministic given a 64-bit seed, so the same
 *  seed reproduces the same rotation on any platform.
 */
#pragma once
#include <usearch/index.hpp> // buffer_gt, byte_t, expected_gt

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace unum {
namespace usearch {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the smallest power of two >= n (n > 0).
inline std::size_t tq_next_power_of_2(std::size_t n) noexcept {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (in-place, unnormalized)
// ---------------------------------------------------------------------------

/**
 *  @brief  In-place unnormalized WHT (butterfly formulation).
 *          Result is H_unnorm * data, where H_unnorm has ±1 entries.
 *          Caller must divide by sqrt(n) afterwards for the orthonormal version.
 *  @param  data   Array of length @p n (must be a power of 2).
 *  @param  n      Length of @p data.
 */
inline void tq_fwht_inplace(float* data, std::size_t n) noexcept {
    for (std::size_t len = 1; len < n; len <<= 1) {
        for (std::size_t i = 0; i < n; i += (len << 1)) {
            for (std::size_t j = 0; j < len; ++j) {
                float const u = data[i + j];
                float const v = data[i + j + len];
                data[i + j] = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SplitMix64-based deterministic sign generator
// ---------------------------------------------------------------------------

inline std::uint64_t tq_splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// HD³ Rotation
// ---------------------------------------------------------------------------

/**
 *  @brief  Randomized Hadamard rotation (HD³).
 *
 *  Forward transform:  y = H · D₂ · H · D₁ · H · D₀ · x
 *  (applied as:  D₀ → WHT → D₁ → WHT → D₂ → WHT)
 *
 *  Inverse transform:  x = D₀ · H · D₁ · H · D₂ · H · y
 *  (applied as:  WHT → D₂ → WHT → D₁ → WHT → D₀)
 *
 *  where H is the normalized Hadamard (WHT / √n) and Dₖ are diagonal ±1.
 */
class turboquant_rotation_t {
  public:
    static constexpr std::size_t rounds_k = 3;

    std::size_t original_dim_ = 0;
    std::size_t padded_dim_ = 0;
    std::uint64_t seed_ = 0;

    /// 3 × padded_dim_ sign bytes (+1 or −1).
    buffer_gt<std::int8_t> signs_;

    /// 1 / √padded_dim_  (pre-computed).
    float inv_sqrt_n_ = 0;

  public:
    turboquant_rotation_t() noexcept = default;
    turboquant_rotation_t(turboquant_rotation_t&&) noexcept = default;
    turboquant_rotation_t& operator=(turboquant_rotation_t&&) noexcept = default;

    /**
     *  @brief  Initializes the rotation for a given vector dimensionality.
     *  @param  dim   Original (unpadded) dimensionality.
     *  @param  seed  64-bit seed for reproducibility.
     *  @return True on success, false on allocation failure.
     */
    bool initialize(std::size_t dim, std::uint64_t seed = 42) noexcept {
        original_dim_ = dim;
        padded_dim_ = tq_next_power_of_2(dim);
        seed_ = seed;
        inv_sqrt_n_ = 1.0f / std::sqrt(static_cast<float>(padded_dim_));

        signs_ = buffer_gt<std::int8_t>(rounds_k * padded_dim_);
        if (!signs_)
            return false;

        // Generate deterministic ±1 signs from the seed.
        std::uint64_t state = seed;
        for (std::size_t i = 0; i < rounds_k * padded_dim_; ++i)
            signs_[i] = (tq_splitmix64(state) & 1) ? std::int8_t(1) : std::int8_t(-1);

        return true;
    }

    /**
     *  @brief  Forward HD³ rotation.
     *
     *  @param  input   Source vector of length original_dim().
     *  @param  output  Destination buffer of length padded_dim().
     *                  May alias @p input only if padded_dim() == original_dim().
     *
     *  The caller is responsible for allocating @p output.
     */
    void forward(float const* input, float* output) const noexcept {
        // Copy + zero-pad.
        std::memcpy(output, input, original_dim_ * sizeof(float));
        if (padded_dim_ > original_dim_)
            std::memset(output + original_dim_, 0, (padded_dim_ - original_dim_) * sizeof(float));

        // 3 rounds: Dₖ then normalized WHT.
        for (std::size_t r = 0; r < rounds_k; ++r) {
            std::int8_t const* s = signs_.data() + r * padded_dim_;
            // Diagonal sign flip.
            for (std::size_t i = 0; i < padded_dim_; ++i)
                output[i] *= s[i];
            // Unnormalized WHT then scale.
            tq_fwht_inplace(output, padded_dim_);
            for (std::size_t i = 0; i < padded_dim_; ++i)
                output[i] *= inv_sqrt_n_;
        }
    }

    /**
     *  @brief  Inverse HD³ rotation.
     *
     *  @param  input   Source vector of length padded_dim().
     *  @param  output  Destination buffer of length original_dim().
     *
     *  Only the first original_dim() components of the inverse are written.
     */
    void inverse(float const* input, float* output) const noexcept {
        // We need a full padded_dim_ buffer for the computation.
        // Use the caller-provided output as scratch if large enough, otherwise
        // fall back to a local buffer.  Since padded_dim_ >= original_dim_,
        // output is always large enough if we work carefully.
        //
        // In practice, callers wanting the inverse should allocate padded_dim_
        // floats.  For safety we accept original_dim_ and use a stack buffer
        // only for small dimensions (≤ 4096).  Larger cases require a
        // padded_dim_-sized output.
        //
        // For simplicity here, we do a small dynamic allocation via alloca-style
        // if needed.  In the test-only path this is acceptable.
        //
        // Because inverse is NOT on the critical path (per design), a simple
        // heap allocation is fine.
        std::vector<float> buf(padded_dim_);
        float* tmp = buf.data();
        std::memcpy(tmp, input, padded_dim_ * sizeof(float));

        // Inverse: WHT → D₂ → WHT → D₁ → WHT → D₀
        for (std::size_t r = rounds_k; r-- > 0;) {
            tq_fwht_inplace(tmp, padded_dim_);
            for (std::size_t i = 0; i < padded_dim_; ++i)
                tmp[i] *= inv_sqrt_n_;
            std::int8_t const* s = signs_.data() + r * padded_dim_;
            for (std::size_t i = 0; i < padded_dim_; ++i)
                tmp[i] *= s[i];
        }

        // Copy the first original_dim_ elements.
        std::memcpy(output, tmp, original_dim_ * sizeof(float));
    }

    // -- Accessors ----------------------------------------------------------

    std::size_t original_dim() const noexcept { return original_dim_; }
    std::size_t padded_dim() const noexcept { return padded_dim_; }
    std::uint64_t seed() const noexcept { return seed_; }
    bool initialized() const noexcept { return signs_.size() > 0; }
};

} // namespace usearch
} // namespace unum
