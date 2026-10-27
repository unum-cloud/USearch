/**
 *  @file include/usearch/utilities/vram_nvidia_device.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device-side vocabulary every NVIDIA index's kernels share: the warp policy the walks
 *      are written against and the arena slice's device twin.
 */
#pragma once

#include <cstring> // `std::memcpy` - a lane's byte copies

#include <cuda/atomic> // `cuda::atomic_ref` - the device twin of `std::atomic_ref`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.hpp"
#include "usearch/utilities/vram_queue.cuh"

namespace unum::usearch::utilities {

// Checked where only CUDA builds compile, since 32-bit hosts such as wasm32 never load the driver.
static_assert(sizeof(cuda_allocation_properties_t) == 32 && sizeof(cuda_access_descriptor_t) == 12 &&
                  sizeof(cuda_copy_2d_t) == 128 && sizeof(cuda_launch_config_t) == 56 &&
                  sizeof(cuda_launch_attribute_t) == 72 && sizeof(cuda_tensor_map_t) == 128,
              "The driver's 64-bit layouts");

/** The device-scope atomic reference the NVIDIA kernels reach a shared word through, so each
 *  access names its memory order - relaxed wherever the launch boundary is what publishes the
 *  result. */
template <typename value_type_>
using cuda_atomic_ref = cuda::atomic_ref<value_type_, cuda::thread_scope_device>;

/**
 *  @brief The NVIDIA warp policy the shared walk code is written against: 32-lane warps with
 *      32-bit lane masks and device-scope atomics.
 *
 *  Every collective takes the full-warp mask, since the walks keep their warps converged.
 */
struct cuda_warp_t {
    using mask_t = u32_t;
    inline static constexpr u32_t width_k = 32;
    inline static constexpr mask_t all_k = 0xFFFFFFFFu;

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle(value_type_ value, u32_t source_lane) noexcept {
        return __shfl_sync(all_k, value, source_lane);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_xor(value_type_ value, u32_t lane_mask) noexcept {
        return __shfl_xor_sync(all_k, value, lane_mask);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_up(value_type_ value, u32_t delta) noexcept {
        return __shfl_up_sync(all_k, value, delta);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ shuffle_down(value_type_ value, u32_t delta,
                                                               u32_t width = width_k) noexcept {
        return __shfl_down_sync(all_k, value, delta, width);
    }
    // The predicates are `int`, as the warp intrinsics take them, so any nonzero lane votes.
    static __device__ __forceinline__ mask_t ballot(int predicate) noexcept { return __ballot_sync(all_k, predicate); }
    static __device__ __forceinline__ bool any(int predicate) noexcept { return __any_sync(all_k, predicate); }
    static __device__ __forceinline__ bool all(int predicate) noexcept { return __all_sync(all_k, predicate); }
    static __device__ __forceinline__ void sync() noexcept { __syncwarp(); }

    /** The lanes holding the same @p value as this one, this lane included. */
    template <typename value_type_>
    static __device__ __forceinline__ mask_t match_any(value_type_ value) noexcept {
        return __match_any_sync(all_k, value);
    }

    static __device__ __forceinline__ u32_t count(mask_t mask) noexcept { return __popc(mask); }

    /** The lowest lane of a non-empty @p mask. */
    static __device__ __forceinline__ u32_t first(mask_t mask) noexcept { return __ffs(mask) - 1; }

    /** The lanes strictly below @p lane. */
    static __device__ __forceinline__ mask_t below(u32_t lane) noexcept { return (1u << lane) - 1; }

    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_relaxed(value_type_& word) noexcept {
        return cuda_atomic_ref<value_type_>(word).load(cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ load_acquire(value_type_& word) noexcept {
        return cuda_atomic_ref<value_type_>(word).load(cuda::memory_order_acquire);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_relaxed(value_type_& word, value_type_ value) noexcept {
        cuda_atomic_ref<value_type_>(word).store(value, cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ void store_release(value_type_& word, value_type_ value) noexcept {
        cuda_atomic_ref<value_type_>(word).store(value, cuda::memory_order_release);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ exchange_acquire(value_type_& word, value_type_ value) noexcept {
        return cuda_atomic_ref<value_type_>(word).exchange(value, cuda::memory_order_acquire);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_add_relaxed(value_type_& word, value_type_ value) noexcept {
        return cuda_atomic_ref<value_type_>(word).fetch_add(value, cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_sub_relaxed(value_type_& word, value_type_ value) noexcept {
        return cuda_atomic_ref<value_type_>(word).fetch_sub(value, cuda::memory_order_relaxed);
    }
    template <typename value_type_>
    static __device__ __forceinline__ value_type_ fetch_or_relaxed(value_type_& word, value_type_ value) noexcept {
        return cuda_atomic_ref<value_type_>(word).fetch_or(value, cuda::memory_order_relaxed);
    }

    /** Swaps @p desired into @p word if it still holds @p expected, answering whether it did. */
    template <typename value_type_>
    static __device__ __forceinline__ bool compare_exchange_relaxed(value_type_& word, value_type_ expected,
                                                                    value_type_ desired) noexcept {
        return cuda_atomic_ref<value_type_>(word).compare_exchange_strong(expected, desired,
                                                                          cuda::memory_order_relaxed);
    }

    /** Orders every store before it ahead of any store after it, for readers anywhere on the
     *  device. */
    static __device__ __forceinline__ void fence_release() noexcept {
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
    }

    /** Copies @p bytes from @p source into @p target on the calling lane. */
    static __device__ __forceinline__ void copy_bytes(void* target, void const* source, std::size_t bytes) noexcept {
        std::memcpy(target, source, bytes);
    }
};

} // namespace unum::usearch::utilities
