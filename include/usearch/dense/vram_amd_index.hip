/**
 *  @file include/usearch/dense/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index's kernels: one template over the kernel, the capability ceiling, the dtype,
 *      the rung and the attend value lookup.
 *
 *  Included only by the class units, which HIP-Clang compiles device-only, once per class over
 *  every compiled dtype. Each @c amd_kernel instantiation turns the plain records of
 *  @c vram_index.hpp into the shared views of @c vram_index.cuh and runs the shared bodies there
 *  through its schedule's @ref hip_warp, and the host finds it by the symbol @c vram_kernel_symbol
 *  spells.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <utility> // `std::index_sequence` - every rung of one dtype

#include <numkong/types.h>       // `NUMKONG_ARCH_ROCM_`, which the two below test before including it
#include <numkong/cast/simt.cuh> // `nk_cast_unit_simt_` - the import casts' device body
#include <numkong/each/rocm.cuh> // `nk_each_rmscast_e4m3_rocm_vectors_` & kin - the unit-RMS import's device bodies

#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/vram_index.cuh"
#include "usearch/dense/vram_amd_metric.hip"
#include "usearch/dense/vram_amd_index.hpp"

namespace unum::usearch::dense {

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename kernels_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
bool vram_amd_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_, neighbors_allocator_type_,
                    traversals_allocator_type_>::walk_scratch_fits(std::size_t fold_bytes) const noexcept {
    using kit_t = typename quads_ladder<scalar_t, capability_>::kit_t;
    using warp_t = typename schedule_ladder<capability_>::schedule_t::warp_t;
    static_assert(std::is_same_v<typename kit_t::norms_t, norms_t> &&
                  sizeof(typename kit_t::fragment_t) == base_t::fragment_bytes_k);
    static_assert(warp_t::width_k == hip_wave_lanes(capability_));
    return std::size_t {vram_warps_per_block_k} * (row_bytes() + fold_bytes + sizeof(vram_gathered<kit_t, warp_t>)) <=
           device_.shared_memory_per_block_bytes;
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename kernels_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
bool vram_amd_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_, neighbors_allocator_type_,
                    traversals_allocator_type_>::build_scratch_fits() const noexcept {
    using kit_t = typename quads_ladder<scalar_t, capability_>::kit_t;
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    std::size_t const statics = std::size_t {vram_warps_per_block_k} *
                                (vram_cache_rows_k * sizeof(norms_t) +
                                 sizeof(vram_gathered<kit_t, typename schedule_t::warp_t>) +
                                 sizeof(typename schedule_t::shared_t));
    return statics + std::size_t {vram_warps_per_block_k} *
                         max_of(insert_scratch_bytes_per_warp(), merge_scratch_bytes_per_warp()) <=
           device_.shared_memory_per_block_bytes;
}

#pragma region Kernels

/** Every AMD kernel: @p kernel_ picks the body, @p capability_ and @p dtype_ the kit,
 *  @p capability_ alone the schedule and its warp, @p slots_ the rung, and @p lookup_ an attend's
 *  value lookup through @ref amd_value_lookup_of. The blocks @ref hip_resident_blocks seats per
 *  compute unit bound its registers and reach the body, which sizes its load window by them. */
template <vram_kernel_t kernel_, nk_capability_t capability_, nk_dtype_t dtype_, u32_t slots_, u32_t lookup_>
__global__ void __launch_bounds__(vram_warps_per_block_k* hip_wave_lanes(capability_),
                                  hip_waves_per_simd(hip_resident_blocks<capability_, dtype_>(kernel_, slots_)))
    amd_kernel(vram_arguments_t arguments) {
    using kit_t = typename quads_ladder<typename nk::type_for<dtype_>::type, capability_>::kit_t;
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    using warp_t = typename schedule_t::warp_t;
    constexpr u32_t resident_blocks = hip_resident_blocks<capability_, dtype_>(kernel_, slots_);
    vram_graph_t const& graph = arguments.graph;
    if constexpr (kernel_ == vram_kernel_t::insert_k)
        vram_insert<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel_ == vram_kernel_t::back_link_k)
        vram_back_link<kit_t, schedule_t, resident_blocks>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel_ == vram_kernel_t::refine_k)
        vram_refine<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.refine));
    else if constexpr (kernel_ == vram_kernel_t::search_k)
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
            vram_view<kit_t>(arguments.search), vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    else if constexpr (kernel_ == vram_kernel_t::search_indexer_k) {
        // Only the floating dtypes take indexer heads; others export a plain walk never launched.
        if constexpr (floating_scalar<typename kit_t::scalar_t>)
            vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.search),
                                                                       vram_view<kit_t>(arguments.search_sink),
                                                                       vram_view<kit_t, warp_t>(arguments.rescoring));
        else
            vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
                vram_view<kit_t>(arguments.search), vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    }
    else if constexpr (kernel_ == vram_kernel_t::multiply_k)
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.search),
                                                                   vram_view<kit_t, warp_t>(arguments.multiply_sink),
                                                                   vram_no_rescoring {});
    else if constexpr (kernel_ == vram_kernel_t::attend_k) {
        using value_lookup_t = typename amd_value_lookup_of<lookup_>::type;
        vram_attend_t const& attend = arguments.attend;
        vram_rescoring_t const& heads = arguments.rescoring;
        value_lookup_t value_lookup;
        warp_t::copy_bytes(&value_lookup, attend.value_lookup, sizeof(value_lookup_t));
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
            vram_view<kit_t>(arguments.search),
            vram_attend_sink<kit_t, warp_t, value_lookup_t> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                             attend.logit_scale, attend.heads_count, attend.columns},
            vram_heads_rescoring<kit_t, warp_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads),
                                                 nullptr, heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel_ == vram_kernel_t::merge_back_links_k)
        vram_merge_back_links<kit_t, schedule_t, resident_blocks>(vram_view<kit_t>(arguments.refine), graph.bucket_ids,
                                                                  graph.bucket_counts, graph.bucket_width,
                                                                  graph.touched);
    else if constexpr (kernel_ == vram_kernel_t::scatter_back_links_k)
        vram_scatter_back_links<warp_t>(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count,
                                        graph.node_ids, graph.connectivity, graph.bucket_width, graph.bucket_ids,
                                        graph.bucket_counts, graph.touched);
    else if constexpr (kernel_ == vram_kernel_t::derive_norms_k)
        vram_derive_norms<kit_t, warp_t>(vram_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel_ == vram_kernel_t::rebase_links_k)
        vram_rebase_links<warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                  graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel_ == vram_kernel_t::mark_seeds_k)
        vram_mark_seeds<warp_t>(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel_ == vram_kernel_t::reachability_sweep_k)
        vram_reachability_sweep<warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                        graph.frontier, graph.next_frontier, graph.counter, graph.previous_counter,
                                        graph.levels);
    else if constexpr (kernel_ == vram_kernel_t::reachability_islands_k)
        vram_reachability_islands<warp_t>(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel_ == vram_kernel_t::reached_in_links_k)
        vram_reached_in_links<warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                      graph.in_links);
    else if constexpr (kernel_ == vram_kernel_t::reconnect_island_k)
        vram_reconnect_island<warp_t>(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count,
                                      graph.reached, graph.rows_count, graph.seed_count, graph.in_links,
                                      graph.reconnected);
    else if constexpr (kernel_ == vram_kernel_t::cast_k) {
        nk_size_t const threads = static_cast<nk_size_t>(gridDim.x) * blockDim.x;
        for (nk_size_t unit = static_cast<nk_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             unit < arguments.cast.plan.units; unit += threads)
            nk_cast_unit_simt_(&arguments.cast, unit);
    }
    else if constexpr (kernel_ == vram_kernel_t::rmscast_k) {
        // Only narrow floats take a unit-RMS import; every other dtype exports one never launched.
        if constexpr (dtype_ == nk_f16_k) nk_each_rmscast_f16_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e4m3_k) nk_each_rmscast_e4m3_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e5m2_k) nk_each_rmscast_e5m2_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e2m3_k) nk_each_rmscast_e2m3_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e3m2_k) nk_each_rmscast_e3m2_rocm_vectors_(arguments.rmscast);
    }
    else if constexpr (kernel_ == vram_kernel_t::resolve_matches_k)
        vram_resolve_matches<warp_t>(arguments.matches.ids, arguments.matches.distances, arguments.matches.walked,
                                     arguments.matches.keys, arguments.matches.queries_count,
                                     vram_view(arguments.matches.ranking));
    else if constexpr (kernel_ == vram_kernel_t::merge_matches_k)
        vram_merge_matches<warp_t>(vram_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                   arguments.matches.shards, arguments.matches.queries_count,
                                   vram_view(arguments.matches.ranking));
    else
        vram_merge_attention<warp_t>(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
                                     arguments.attend.columns, arguments.attend.shards);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, std::size_t... rungs_>
void amd_instantiate_rungs(std::index_sequence<rungs_...>) {
    (static_cast<void>(&amd_kernel<amd_rungs[rungs_].kernel, capability_, dtype_, amd_rungs[rungs_].slots, 0>), ...);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, u32_t lookup_, std::size_t... rungs_>
void amd_instantiate_attends(std::index_sequence<rungs_...>) {
    (static_cast<void>(&amd_kernel<vram_kernel_t::attend_k, capability_, dtype_, amd_walk_rungs[rungs_], lookup_>),
     ...);
}

/** Instantiates every kernel one dtype exports under one ceiling, and its attends over each of
 *  @p lookups_; a unit compiling a code object names it once per compiled dtype. Host-side, since
 *  HIP-Clang refuses a kernel's address in device code, while a host reference still makes the
 *  device pass emit it. Only a device-only entry names it: on a host pass the addresses would emit
 *  launch stubs and a fat-binary registration the library never links, since it loads the class's
 *  code object itself. */
template <nk_capability_t capability_, nk_dtype_t dtype_, u32_t... lookups_>
void amd_instantiate() {
    amd_instantiate_rungs<capability_, dtype_>(std::make_index_sequence<sizeof(amd_rungs) / sizeof(amd_rungs[0])> {});
    (amd_instantiate_attends<capability_, dtype_, lookups_>(
         std::make_index_sequence<sizeof(amd_walk_rungs) / sizeof(amd_walk_rungs[0])> {}),
     ...);
}

#pragma endregion Kernels

} // namespace unum::usearch::dense
