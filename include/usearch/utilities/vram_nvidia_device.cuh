/**
 *  @file include/usearch/utilities/vram_nvidia_device.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Device-side vocabulary every NVIDIA index's kernels share: the warp policy the walks
 *      are written against, the arena slice's device twin, the touched-rows bitset and the
 *      reverse scatter of a walk's picks.
 */
#pragma once

#include <cuda/atomic> // `cuda::atomic_ref` - the device twin of `std::atomic_ref`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.hpp"
#include "usearch/utilities/vram_queue.cuh"

namespace unum::usearch::utilities {

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
    static __device__ __forceinline__ mask_t match_any(u32_t value) noexcept { return __match_any_sync(all_k, value); }

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

    /** Orders every store before it ahead of any store after it, for readers anywhere on the
     *  device. */
    static __device__ __forceinline__ void fence_release() noexcept {
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
    }
};

/** The kernels' twin of a host-computed arena slice layout. */
constexpr vram_arena_shape_t cuda_arena_shape(cuda_arena_shape_t shape) noexcept {
    return {shape.visited_slots, shape.visited_hash_shift, shape.tail_entries, shape.cache_bytes};
}

/** Whether @p bitset - a reachability audit's reached set or frontier, a refinement pass's touched
 *  rows - holds @p row. Every access is relaxed: the launch boundary orders the marks before any
 *  later launch or host copy reads them. */
__device__ __forceinline__ bool cuda_bitset_contains(cuda_bitset_t bitset, u32_t row) noexcept {
    return (cuda_atomic_ref<u32_t>(bitset.words[row >> 5]).load(cuda::memory_order_relaxed) >> (row & 31u)) & 1u;
}

/** Sets the bit of @p row, answering whether this call was the one that set it. */
__device__ __forceinline__ bool cuda_bitset_insert(cuda_bitset_t bitset, u32_t row) noexcept {
    u32_t const bit = 1u << (row & 31u);
    return !(cuda_atomic_ref<u32_t>(bitset.words[row >> 5]).fetch_or(bit, cuda::memory_order_relaxed) & bit);
}

/** Scatters the walked rows' shadow neighbors in reverse: every forward pick @b u → @b v lands
 *  @b u in @b v's reverse bucket, and marks both rows in @p touched - the rows the merge
 *  re-selects. The walked rows are the range from @p first_id, or a heal round's islands listed in
 *  @p node_ids, which mark only themselves and land in no bucket; appends past @p bucket_width are
 *  counted, never written. */
__device__ __forceinline__ void cuda_scatter_back_links(u32_t const* shadow_neighbors, u32_t rows_count, u32_t first_id,
                                                        u32_t walk_count, u32_t const* node_ids, u32_t connectivity,
                                                        u32_t bucket_width, u32_t* bucket_ids, u32_t* bucket_counts,
                                                        cuda_bitset_t touched) {
    vram_grid_warp_t const warp = vram_grid_warp<cuda_warp_t>();
    for (u32_t offset = warp.index; offset < walk_count; offset += warp.count) {
        u32_t const row = node_ids ? node_ids[offset] : first_id + offset;
        if (warp.lane == 0) cuda_bitset_insert(touched, row);
        if (node_ids) continue;
        vram_for_each_link<cuda_warp_t>(
            shadow_neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
                cuda_bitset_insert(touched, target);
                u32_t const landing = cuda_warp_t::fetch_add_relaxed(bucket_counts[target], 1u);
                u32_t* const bucket = vram_row(bucket_ids, target, bucket_width);
                if (landing < bucket_width) bucket[landing] = row;
            });
    }
}

} // namespace unum::usearch::utilities
