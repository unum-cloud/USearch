/**
 *  @file test/sequence/vram_kernels_amd.hip
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief The AMD suite's sequence kernel: alignment candidate pairs scored with the single scorer
 *      twice and the paired scorer once, which @c test/dense/vram_kernels_amd.hip compiles
 *      device-only into the suite's own code object.
 *
 *  @c vram_index.cpp, compiled host-only, finds the kernel by its unmangled name.
 */
#pragma once

#include "usearch/sequence/vram_amd_metric.hip"

namespace unum::usearch::sequence {

/** A wavefront of the CDNA3 kit per task @c t: lays out sequence @c 3t of @p tape, sequence @c i
 *  spanning @c offsets[i] to @c offsets[i+1], as a query under @p costs in its own @p span_bytes
 *  of @p spans, then scores it against sequences @c 3t+1 and @c 3t+2 under @c radii[2t] and
 *  @c radii[2t+1], into @c distances[4t] onwards: the single scorer's two, then the pair's. */
extern "C" __global__ void alignment_pairs_kernel(byte_t const* costs, byte_t const* tape, u32_t const* offsets,
                                                  distance_t const* radii, byte_t* spans, u32_t span_bytes,
                                                  distance_t* distances) {
    using warp_t = packed_cdna3_alignment_kit_t::warp_t;
    constexpr u32_t rows_k = vram_alignment_rows_k[std::size(vram_alignment_rows_k) - 1];
    u32_t const task = blockIdx.x, lane = threadIdx.x;
    auto const entry = [&](u32_t sequence) noexcept {
        return basic::vram_entry_t {tape + offsets[sequence], offsets[sequence + 1] - offsets[sequence]};
    };
    hip_alignment_query_t const query = hip_alignment_prepare<warp_t>(costs, entry(3 * task),
                                                                      spans + std::size_t {task} * span_bytes, lane);
    distance_t const first_radius = radii[2 * task], second_radius = radii[2 * task + 1];
    distance_t const first = hip_alignment_score<rows_k, warp_t>(query, entry(3 * task + 1), first_radius, lane);
    distance_t const second = hip_alignment_score<rows_k, warp_t>(query, entry(3 * task + 2), second_radius, lane);
    basic::vram_distance_pair_t const paired = hip_alignment_score_pair<rows_k, warp_t>(
        query, entry(3 * task + 1), entry(3 * task + 2), first_radius, second_radius, lane);
    if (lane) return;
    distances[4 * task] = first, distances[4 * task + 1] = second;
    distances[4 * task + 2] = paired.first, distances[4 * task + 3] = paired.second;
}

} // namespace unum::usearch::sequence
