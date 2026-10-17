/**
 *  @file src/basic/vram_variants_nvidia.cu
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The NVIDIA unit of the kernels every basic face shares, over 32-bit node IDs.
 *
 *  The rebase of absorbed rows, the reachability audit with its reconnection, and a split's
 *  partition and scatter read the neighbors matrix alone and move values as opaque bytes, so one
 *  unit serves every modality on every NVIDIA class.
 *
 *  Compiled twice: by NVCC device-only into the unit's fatbin, and by Clang host-only, where it
 *  names that fatbin, linked in beside it, to the faces, which load it at their first @c reserve.
 */
#include <iterator> // `std::size` - every shared kernel
#include <utility>  // `std::index_sequence` - every shared kernel

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t`
#include "usearch/dense/vram_index.cuh"             // `vram_rebase_links`, the audits' & splits' bodies
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::basic {

/** The unit's fatbin, linked in as raw bytes, which every basic face loads. */
extern "C" byte_t const basic_nvidia_bundle_start[], basic_nvidia_bundle_end[];

bytes_view_t cuda_graph_fatbin_t::bytes() noexcept {
    return {basic_nvidia_bundle_start, static_cast<std::size_t>(basic_nvidia_bundle_end - basic_nvidia_bundle_start)};
}

/** Every shared NVIDIA basic kernel: @p kernel_ picks the pass, and @p node_id_type_ the IDs of the
 *  neighbors matrix, which the dense design's bodies read as 32-bit words. */
template <vram_kernel_t kernel_, typename node_id_type_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, cuda_resident_blocks(kernel_, 0))
    cuda_kernel(vram_graph<node_id_type_> graph) {
    if constexpr (kernel_ == vram_kernel_t::rebase_links_k)
        dense::vram_rebase_links<cuda_warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                              graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel_ == vram_kernel_t::mark_seeds_k)
        dense::vram_mark_seeds<cuda_warp_t>(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel_ == vram_kernel_t::reachability_sweep_k)
        dense::vram_reachability_sweep<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity,
                                                    graph.reached, graph.frontier, graph.next_frontier, graph.counter,
                                                    graph.previous_counter, graph.levels);
    else if constexpr (kernel_ == vram_kernel_t::reachability_islands_k)
        dense::vram_reachability_islands<cuda_warp_t>(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel_ == vram_kernel_t::reached_in_links_k)
        dense::vram_reached_in_links<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                                  graph.in_links);
    else if constexpr (kernel_ == vram_kernel_t::reconnect_island_k)
        dense::vram_reconnect_island<cuda_warp_t>(graph.neighbors, graph.connectivity, graph.island_ids,
                                                  graph.islands_count, graph.reached, graph.rows_count,
                                                  graph.seed_count, graph.in_links, graph.reconnected);
    else if constexpr (kernel_ == vram_kernel_t::partition_seeds_k)
        dense::vram_partition_seeds<cuda_warp_t>(graph.roots, graph.seed_count, graph.labels, graph.frontier,
                                                 graph.part_sizes);
    else if constexpr (kernel_ == vram_kernel_t::partition_propose_k)
        dense::vram_partition_propose<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.labels,
                                                   graph.proposals, graph.frontier, graph.part_sizes,
                                                   graph.part_capacity, graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_tally_k)
        dense::vram_partition_tally<cuda_warp_t>(graph.labels, graph.proposals, graph.rows_count, graph.proposed,
                                                 graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_accept_k)
        dense::vram_partition_accept<cuda_warp_t>(graph.labels, graph.proposals, graph.rows_count, graph.next_frontier,
                                                  graph.part_sizes, graph.proposed, graph.part_capacity, graph.counter,
                                                  graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_fill_k)
        dense::vram_partition_fill<cuda_warp_t>(graph.labels, graph.proposals, graph.rows_count, graph.next_frontier,
                                                graph.part_sizes, graph.proposed, graph.part_capacity, graph.counter,
                                                graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_islands_k)
        dense::vram_partition_islands<cuda_warp_t>(graph.labels, graph.rows_count, graph.part_sizes, graph.parts_count);
    else if constexpr (kernel_ == vram_kernel_t::partition_seams_k)
        dense::vram_partition_seams<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.labels,
                                                 graph.seams);
    else if constexpr (kernel_ == vram_kernel_t::split_links_k)
        dense::vram_split_links<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.labels,
                                             graph.part_capacity, graph.part_neighbors);
    else
        dense::vram_split_tape<cuda_warp_t>(graph.values, graph.value_offsets, graph.rows_count, graph.labels,
                                            graph.part_capacity, graph.part_values, graph.part_offsets);
}

template <typename node_id_type_, std::size_t... kernels_>
__device__ void cuda_instantiate_kernels(std::index_sequence<kernels_...>) noexcept {
    (static_cast<void>(&cuda_kernel<cuda_graph_kernels[kernels_], node_id_type_>), ...);
}

/** Instantiates every shared kernel with @p node_id_type_ IDs. Taking the kernels' addresses in
 *  device code makes NVCC emit them, while a host-only pass emits no launch stub. */
template <typename node_id_type_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_kernels<node_id_type_>(std::make_index_sequence<std::size(cuda_graph_kernels)> {});
}

template __device__ void cuda_instantiate<u32_t>();

} // namespace unum::usearch::basic
