/**
 *  @file include/usearch/shapes/vram_apple_metric.metal
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The Metal kit for the shapes Apple index, the MSL twin of @c vram_nvidia_metric.cuh:
 *      Chamfer and Hausdorff between f16 packs in f32 math, a simdgroup to a shape.
 *
 *  A stored shape is its pack byte for byte, and the simdgroup bodies score two packs where they
 *  lie, stopping once a candidate lies past the walk's radius, so a query is the address of its
 *  own pack and prepares into no span: packs are whole 64-byte lines, so every pack of a tape
 *  starts on one. The whole simdgroup scores one candidate, its lanes splitting the outer cloud's
 *  points, and a threadgroup walks each query while its four simdgroups score a candidate each.
 *  The face refuses f32 packs, which no Metal body here scores to CUDA's f32 guarantee.
 *
 *  The host splices NumKong's @c types.metal and @c shapes/metal.metal ahead of this file.
 */
#include <metal_stdlib>
using namespace metal;

namespace unum::usearch::shapes {

using namespace unum::usearch::utilities;

/** The metric a launch scores under, as the host stores the kernel kind: NumKong's
 *  @c nk_kernel_hausdorffs_packed_k, Chamfer otherwise. */
constant constexpr uint metal_hausdorff_kernel_k = '9';

/** The pack a walk scores against, where its entry lies. */
struct metal_pack_query_t {
    device uchar const* pack;
};

/** Chamfer and Hausdorff between f16 packs in f32 math, a simdgroup to a shape. */
struct metal_f16_shapes_kit_t {
    using query_t = metal_pack_query_t;

    /** One rung, serving every cloud. */
    static constexpr constant uint rungs_k[1] = {0};

    static constexpr uint lanes_per_row(uint) { return metal_simdgroup_width_k; }

    static query_t prepare(device uchar const*, device uchar const* data, uint, device uchar*, uint) {
        query_t query;
        query.pack = data;
        return query;
    }

    /** The gap between the two clouds' largest radii, from their headers alone, bounds Hausdorff;
     *  nothing bounds Chamfer. */
    static float lower_bound(uint metric, thread query_t const& query, device uchar const* data, uint) {
        return metric == metal_hausdorff_kernel_k ? nk_hausdorff_bound_packed_f16_metal_(query.pack, data) : 0.0f;
    }

    template <uint rung_>
    static float score(uint metric, thread query_t const& query, device uchar const* data, uint, float radius,
                       uint lane, uint) {
        return metric == metal_hausdorff_kernel_k
                   ? nk_hausdorff_simdgroup_packed_f16_metal_(query.pack, data, radius, lane)
                   : nk_chamfer_simdgroup_packed_f16_metal_(query.pack, data, radius, lane);
    }
};

} // namespace unum::usearch::shapes
