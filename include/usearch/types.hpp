/**
 *  @brief Vector Search helper structures and types
 *  @author Ash Vardanian
 *  @file include/usearch/inner_volatile_index.hpp
 *  @date July 26, 2023
 */
#pragma once
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

namespace unum {
namespace usearch {

using u32_t = std::uint32_t;
using u64_t = std::uint64_t;

enum class status_t {
    success_k,
    /** @brief The underlying allocator failed to provide memory; retrying a smaller batch may succeed. */
    bad_alloc_k,
    /** @brief The structure reached a hard design ceiling, like the 2^32 id space; retrying can't help. */
    overflow_k,
    /** @brief The operation requires a prior successful `reserve()`-like initialization call. */
    uninitialized_k,
};

template <typename value_type_>
struct expected {
    value_type_ value;
    status_t status;
};

enum connectivity_t : std::uint32_t {
    connectivity_unknown_k = 0,
    /**
     *  @brief The minimal recommended graph connectivity level, expanding to 16x 32-bit neighbor IDs,
     *      exactly matching the ZMM register size and cache-line width on x86, but half of the width
     *      of coherency protocols of many CPUs and half the cache line width of many Arm CPUs.
     */
    connectivity_16_k = 16,
    /**
     *  @brief The default recommended graph connectivity level, expanding to 32x 32-bit neighbor IDs,
     *      perfect for in-memory billion-scale search indexes.
     */
    connectivity_32_k = 32,
    /**
     *  @brief The minimal recommended graph connectivity level for on-disk billion- and trillion-scale
     *      search indexes, matching the minimal 512-byte SSD sector size.
     */
    connectivity_128_k = 128,
};

/** @brief Returns 2^ @p exponent as a value of the requested unsigned type. */
template <typename result_type_>
inline constexpr result_type_ power_of_two(u32_t exponent) noexcept {
    return result_type_(1) << exponent;
}

/** @brief Returns a mask selecting the lowest @p exponent bits, i.e. 2^ @p exponent - 1. */
template <typename result_type_>
inline constexpr result_type_ power_of_two_mask(u32_t exponent) noexcept {
    return (result_type_(1) << exponent) - result_type_(1);
}

/** @brief Integer division rounding towards positive infinity; @p divisor must be non-zero. */
template <typename value_type_>
inline constexpr value_type_ divide_round_up(value_type_ value, value_type_ divisor) noexcept {
    return (value + divisor - value_type_(1)) / divisor;
}

} // namespace usearch
} // namespace unum
