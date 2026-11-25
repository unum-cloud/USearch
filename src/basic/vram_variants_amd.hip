/**
 *  @file src/basic/vram_variants_amd.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The AMD unit of the kernels every basic face shares, over 32-bit node IDs.
 *
 *  The rebase of moved nodes, the reachability audit with its reconnection, a sweep's count and
 *  scatter, and a split's partition and scatter read the neighbors matrix alone and move values as
 *  opaque bytes, so one unit serves every modality on every AMD class, compiled once per class it
 *  is kept for, over that class's wavefront: 64 lanes on CDNA3 and CDNA4, 32 on CDNA5.
 *
 *  Compiled twice per class: by HIP-Clang device-only, where a host function names every kernel so
 *  the device pass emits them into the class's code object, and host-only, where it names that
 *  code object, linked in beside it, to the faces, which load it at their first @c reserve.
 */
#include <iterator>    // `std::size` - every shared kernel
#include <type_traits> // `std::conditional_t` - the class's wavefront and scans
#include <utility>     // `std::index_sequence` - every shared kernel

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_amd_device.hip" // `hip_warp`, `cdna3_scan_t`, `cdna5_scan_t`
#include "usearch/dense/vram_index.cuh"          // `vram_rebase_links`, the health passes' & splits' bodies
#include "usearch/basic/vram_amd_index.hpp"

#include "lib.hpp" // `gpu_class_t`, `hip_class_k` - the class this unit compiles as

namespace unum::usearch::basic {

#if __HIP_DEVICE_COMPILE__

/** The wavefront and the scans of the class this unit compiles as. */
using hip_class_warp_t = std::conditional_t<hip_class_k == gpu_class_t::amd_cdna5_k, hip_warp<32>, hip_warp<64>>;
using hip_class_scan_t = std::conditional_t<hip_class_k == gpu_class_t::amd_cdna5_k, cdna5_scan_t, cdna3_scan_t>;

/** Every shared AMD basic kernel: @p kernel_ picks the pass, and @p node_id_type_ the IDs of the
 *  neighbors matrix, which the dense design's bodies read as 32-bit words. */
template <vram_kernel_t kernel_, typename node_id_type_>
__global__ void __launch_bounds__(vram_warps_per_block_k* hip_class_warp_t::width_k,
                                  hip_resident_blocks(kernel_, 0) * vram_warps_per_block_k / 4)
    amd_kernel(vram_graph<node_id_type_> graph) {
    using warp_t = hip_class_warp_t;
    using scan_t = hip_class_scan_t;
    if constexpr (kernel_ == vram_kernel_t::rebase_links_k)
        dense::vram_rebase_links<warp_t>(graph.neighbors, graph.first_id, graph.nodes_count, graph.connectivity,
                                         graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel_ == vram_kernel_t::widen_links_k)
        dense::vram_widen_links<warp_t>(graph.neighbors, graph.first_id, graph.nodes_count, graph.connectivity,
                                        graph.saved_neighbors, graph.saved_offsets, graph.degree, graph.source_bound,
                                        graph.shift, graph.malformed);
    else if constexpr (dense::vram_health_kernel(kernel_))
        dense::vram_health_pass<kernel_, warp_t, scan_t>(graph.health);
    else if constexpr (kernel_ == vram_kernel_t::partition_roots_k)
        dense::vram_partition_roots<warp_t>(graph.roots, graph.seed_count, graph.node_parts, graph.frontier,
                                            graph.part_sizes);
    else if constexpr (kernel_ == vram_kernel_t::partition_propose_k)
        dense::vram_partition_propose<warp_t>(graph.neighbors, graph.nodes_count, graph.connectivity, graph.node_parts,
                                              graph.proposals, graph.frontier, graph.part_sizes, graph.part_capacity,
                                              graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_count_k)
        dense::vram_partition_count<warp_t>(graph.node_parts, graph.proposals, graph.nodes_count, graph.round_counts,
                                            graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_accept_k)
        dense::vram_partition_accept<warp_t>(graph.node_parts, graph.proposals, graph.nodes_count, graph.next_frontier,
                                             graph.part_sizes, graph.round_counts, graph.part_capacity, graph.counter,
                                             graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_fill_k)
        dense::vram_partition_fill<warp_t>(graph.node_parts, graph.proposals, graph.nodes_count, graph.next_frontier,
                                           graph.part_sizes, graph.round_counts, graph.part_capacity, graph.counter,
                                           graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_islands_k)
        dense::vram_partition_islands<warp_t>(graph.node_parts, graph.nodes_count, graph.part_sizes, graph.parts_count);
    else if constexpr (kernel_ == vram_kernel_t::split_links_k)
        dense::vram_split_links<warp_t>(graph.neighbors, graph.nodes_count, graph.connectivity, graph.node_parts,
                                        graph.part_capacity, graph.part_neighbors);
    else if constexpr (kernel_ == vram_kernel_t::split_tape_k)
        dense::vram_split_tape<warp_t>(graph.values, graph.value_offsets, graph.nodes_count, graph.node_parts,
                                       graph.part_capacity, graph.part_values, graph.part_offsets);
    else if constexpr (kernel_ == vram_kernel_t::gather_tape_k)
        dense::vram_gather_tape<warp_t, scan_t>(graph.values, graph.value_offsets, graph.listed, graph.listed_count,
                                                graph.gathered_values, graph.gathered_offsets);
}

template <typename node_id_type_, std::size_t... kernels_>
void amd_instantiate_kernels(std::index_sequence<kernels_...>) noexcept {
    (static_cast<void>(&amd_kernel<amd_graph_kernels[kernels_], node_id_type_>), ...);
}

/** Instantiates every shared kernel with @p node_id_type_ IDs. Host-side, since HIP-Clang refuses
 *  a kernel's address in device code, while a host reference still makes the device pass emit it;
 *  only the device pass names it, so no host pass emits launch stubs. */
template <typename node_id_type_>
void amd_instantiate() noexcept {
    amd_instantiate_kernels<node_id_type_>(std::make_index_sequence<std::size(amd_graph_kernels)> {});
}

template void amd_instantiate<u32_t>();

#else

/** The bounds of the code object every basic face loads, linked in as raw bytes, from the table
 *  CMake writes for the class this unit compiles as; the tables of the other classes stay
 *  unreferenced by it. */
extern "C" util::byte_t const* const basic_amd_cdna3_bundles[];
extern "C" util::byte_t const* const basic_amd_cdna4_bundles[];
extern "C" util::byte_t const* const basic_amd_cdna5_bundles[];
template <gpu_class_t gpu_class_>
util::byte_t const* const* basic_bundles() noexcept {
    if constexpr (gpu_class_ == gpu_class_t::amd_cdna5_k) return basic_amd_cdna5_bundles;
    else if constexpr (gpu_class_ == gpu_class_t::amd_cdna4_k) return basic_amd_cdna4_bundles;
    else return basic_amd_cdna3_bundles;
}

template <>
bytes_view_t hip_graph_code_object<numkong_capabilities_of(hip_class_k)>::bytes() noexcept {
    util::byte_t const* const* const bounds = basic_bundles<hip_class_k>();
    return {bounds[0], bounds[1]};
}

#endif

} // namespace unum::usearch::basic
