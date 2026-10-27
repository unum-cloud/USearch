/**
 *  @file include/usearch/utilities/vram_amd_device.hip
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief Device-side vocabulary every AMD index's kernels share, the twin of
 *      @c vram_nvidia_device.cuh: the warp policy the walks are written against.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <hip/hip_runtime.h> // `__shfl`, `__ballot`, `__lane_id` & the agent-scope atomics

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_amd_device.hpp"
#include "usearch/utilities/vram_queue.cuh"

namespace unum::usearch::utilities {

// Checked where only HIP builds compile, since 32-bit hosts such as wasm32 never load the runtime.
static_assert(sizeof(hip_allocation_properties_t) == 32 && sizeof(hip_access_descriptor_t) == 12,
              "HIP's 64-bit layouts");

/**
 *  @brief The AMD warp policy the shared walk code is written against, the twin of
 *      @c cuda_warp_t: wavefronts of @p width_ lanes and agent-scope atomics.
 *
 *  A wavefront moves in lockstep, so the votes and shuffles take no participation mask, and
 *  @ref sync is a wave-scoped fence pair around the scheduling barrier. Masks are as wide as the
 *  wavefront, @c u64_t for the 64 lanes of CDNA3 and CDNA4. The width is the class's, which a
 *  schedule takes from its ceiling, so one code object may carry kernels of either width.
 */
template <u32_t width_>
struct hip_warp {
    static_assert(width_ == 32 || width_ == 64);
    inline static constexpr u32_t width_k = width_;
    using mask_t = std::conditional_t<width_k == 64, u64_t, u32_t>;
    inline static constexpr mask_t all_k = ~mask_t(0);

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle(value_type_ value, u32_t source_lane) noexcept {
        return __shfl(value, source_lane, width_k);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_xor(value_type_ value, u32_t lane_mask) noexcept {
        return __shfl_xor(value, lane_mask, width_k);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_up(value_type_ value, u32_t delta) noexcept {
        return __shfl_up(value, delta, width_k);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_down(value_type_ value, u32_t delta,
                                                               u32_t width = width_k) noexcept {
        return __shfl_down(value, delta, width);
    }
    // The predicates are `int`, as the vote intrinsics take them, so any nonzero lane votes.
    static __device__ __forceinline__ mask_t ballot(int predicate) noexcept { return __ballot(predicate); }
    static __device__ __forceinline__ bool any(int predicate) noexcept { return __any(predicate) != 0; }
    static __device__ __forceinline__ bool all(int predicate) noexcept { return __all(predicate) != 0; }
    static __device__ __forceinline__ void sync() noexcept { __syncwarp(); }

    /** The lanes holding the same @p value as this one, this lane included: one ballot per distinct
     *  value, since the ISA has no matching vote. */
    template <typename value_type_>
    static __device__ __forceinline__ mask_t match_any(value_type_ value) noexcept {
        mask_t unmatched = all_k, matched = 0;
        while (unmatched) {
            mask_t const same = ballot(shuffle(value, first(unmatched)) == value);
            if ((same >> __lane_id()) & 1u) matched = same;
            unmatched &= ~same;
        }
        return matched;
    }

    static __device__ __forceinline__ u32_t count(mask_t mask) noexcept { return std::popcount(mask); }

    /** The lowest lane of a non-empty @p mask. */
    static __device__ __forceinline__ u32_t first(mask_t mask) noexcept { return std::countr_zero(mask); }

    /** The lanes strictly below @p lane. */
    static __device__ __forceinline__ mask_t below(u32_t lane) noexcept { return (mask_t(1) << lane) - 1; }

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_relaxed(value_type_& word) noexcept {
        return __hip_atomic_load(&word, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_acquire(value_type_& word) noexcept {
        return __hip_atomic_load(&word, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_relaxed(value_type_& word, value_type_ value) noexcept {
        __hip_atomic_store(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_release(value_type_& word, value_type_ value) noexcept {
        __hip_atomic_store(&word, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ exchange_acquire(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_exchange(&word, value, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_add_relaxed(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_fetch_add(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_sub_relaxed(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_fetch_sub(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_or_relaxed(value_type_& word, value_type_ value) noexcept {
        return __hip_atomic_fetch_or(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    /** Swaps @p desired into @p word if it still holds @p expected, answering whether it did. */
    template <typename value_type_>
    static __device__ __forceinline__ bool compare_exchange_relaxed(value_type_& word, value_type_ expected,
                                                                    value_type_ desired) noexcept {
        return __hip_atomic_compare_exchange_strong(&word, &expected, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_AGENT);
    }

    /** Orders every store before it ahead of any store after it, for readers across the device. */
    static __device__ __forceinline__ void fence_release() noexcept { __threadfence(); }

    /** Copies @p bytes from @p source into @p target on the calling lane, one at a time, since
     *  `std::memcpy` names the host function under HIP. */
    static __device__ __forceinline__ void copy_bytes(void* target, void const* source, std::size_t bytes) noexcept {
        for (std::size_t offset = 0; offset != bytes; ++offset)
            static_cast<byte_t*>(target)[offset] = static_cast<byte_t const*>(source)[offset];
    }
};

} // namespace unum::usearch::utilities
