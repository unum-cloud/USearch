/**
 *  @brief Vector Search helper structures and types
 *  @author Ash Vardanian
 *  @file include/usearch/inner_volatile_index.hpp
 *  @date July 26, 2023
 */
#pragma once
#include <atomic>  // `std::atomic`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <thread>  // `std::this_thread::yield`

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
    /** @brief The requested configuration contradicts an already-frozen one, like re-reserving
     *      with a different geometry; pass matching parameters or start a fresh structure. */
    config_mismatch_k,
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
constexpr result_type_ power_of_two(u32_t exponent) noexcept {
    return result_type_(1) << exponent;
}

/** @brief Returns a mask selecting the lowest @p exponent bits, i.e. 2^ @p exponent - 1. */
template <typename result_type_>
constexpr result_type_ power_of_two_mask(u32_t exponent) noexcept {
    return (result_type_(1) << exponent) - result_type_(1);
}

/** @brief Integer division rounding towards positive infinity; @p divisor must be non-zero. */
template <typename value_type_>
constexpr value_type_ divide_round_up(value_type_ value, value_type_ divisor) noexcept {
    return (value + divisor - value_type_(1)) / divisor;
}

/**
 *  @brief Portable spin-wait baseline with the same call shape as ForkUnion's hardware wait
 *      monitors, like `fu::x86_tpause_t`, `fu::arm64_wfet_t`, and `fu::risc5_wrs_t`.
 *
 *  Ignores the watched word and simply yields the timeslice - correctness never depends on the
 *  waiter, only latency and power do. The first overload watches a `std::atomic` object; the
 *  second watches an in-place word owned through `std::atomic_ref`, like the slot-0 locks living
 *  inside neighbor rows. The waiting bound (capped or uncapped) arrives as an arbitrary tag type,
 *  so this header stays independent of ForkUnion.
 *
 *  The pointer overload exists because of pre-C++26 STL limitations: `std::atomic_ref` cannot
 *  bind const-qualified objects until P3323, `std::atomic<T>::wait` offers no timed variant, and
 *  neither API exposes the address-armed hardware waits (`UMWAIT`, `WFET`, `WRS`) that ForkUnion
 *  reaches directly - so the raw address plus the observed bit pattern is the portable currency.
 */
struct dummy_waiter_t {
    template <typename value_type_, typename thread_index_type_, typename bound_type_>
    void operator()(std::atomic<value_type_> const&, value_type_, thread_index_type_, bound_type_) const noexcept {
        std::this_thread::yield();
    }
    template <typename value_type_, typename thread_index_type_, typename bound_type_>
    void operator()(value_type_ const*, value_type_, thread_index_type_, bound_type_) const noexcept {
        std::this_thread::yield();
    }
};

} // namespace usearch
} // namespace unum
