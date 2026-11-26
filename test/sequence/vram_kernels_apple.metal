/**
 *  @file test/sequence/vram_kernels_apple.metal
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief The Apple suite's sequence kernel: alignment candidate pairs scored with the single
 *      scorer twice and the paired scorer once, which @c vram_index.cpp embeds behind the
 *      alignment kit's sources for the device to compile at first use.
 *
 *  @c vram_index.cpp finds the kernel by its host name.
 */
namespace unum::usearch::sequence {

/** A simdgroup per task @c t: lays out sequence @c 3t of @p tape, sequence @c i spanning
 *  @c offsets[i] to @c offsets[i+1], as a query under @p costs in its own @p span_bytes of
 *  @p spans, then scores it against sequences @c 3t+1 and @c 3t+2 under @c radii[2t] and
 *  @c radii[2t+1], into @c distances[4t] onwards: the single scorer's two, then the pair's. */
[[host_name("alignment_pairs_kernel")]] kernel void alignment_pairs_kernel(
    device uchar const* costs [[buffer(0)]], device uchar const* tape [[buffer(1)]],
    device uint const* offsets [[buffer(2)]], device float const* radii [[buffer(3)]],
    device uchar* spans [[buffer(4)]], constant uint& span_bytes [[buffer(5)]], device float* distances [[buffer(6)]],
    uint task [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_k = metal_alignment_rows(metal_alignment_canonical_rungs_count_k - 1);
    device uint const* const bounds = offsets + 3 * task;
    metal_alignment_query_t const query = metal_alignment_prepare(costs, tape + bounds[0], bounds[1] - bounds[0],
                                                                  spans + ulong(task) * span_bytes, lane);
    float const first_radius = radii[2 * task], second_radius = radii[2 * task + 1];
    float const first = metal_alignment_score<rows_k>(query, tape + bounds[1], bounds[2] - bounds[1], first_radius,
                                                      lane, simdgroup);
    float const second = metal_alignment_score<rows_k>(query, tape + bounds[2], bounds[3] - bounds[2], second_radius,
                                                       lane, simdgroup);
    float2 const paired = metal_alignment_score_pair<rows_k>(query, tape + bounds[1], bounds[2] - bounds[1],
                                                             tape + bounds[2], bounds[3] - bounds[2], first_radius,
                                                             second_radius, lane, simdgroup);
    if (lane) return;
    distances[4 * task] = first, distances[4 * task + 1] = second;
    distances[4 * task + 2] = paired.x, distances[4 * task + 3] = paired.y;
}

} // namespace unum::usearch::sequence
