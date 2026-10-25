/**
 *  @file include/usearch/dense/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index's kernels: @c amd_kernel for the walks basic graphs share and
 *      @c amd_dense_kernel for the rest, both over the kernel, the capability ceiling, the dtype,
 *      the rung and the attend value lookup.
 *
 *  Included by the per-dtype kernel units, which HIP-Clang compiles device-only once per class,
 *  and by the class units host-only. Each instantiation turns the plain records of
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

template <nk_dtype_t vector_dtype_, typename kernels_type_, typename cache_hints_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
bool vram_amd_kernels<vector_dtype_, kernels_type_, cache_hints_type_, vectors_allocator_type_,
                      neighbors_allocator_type_, traversals_allocator_type_>::walk_scratch_fits(core_t const& core,
                                                                                                std::size_t fold_bytes)
    const noexcept {
    using kit_t = typename quads_ladder<scalar_t, capability_k>::kit_t;
    using warp_t = typename schedule_ladder<capability_k>::schedule_t::warp_t;
    static_assert(std::is_same_v<typename kit_t::norms_t, norms_t> &&
                  sizeof(typename kit_t::fragment_t) == core_t::fragment_bytes_k);
    static_assert(warp_t::width_k == hip_wave_lanes(capability_k));
    return std::size_t {vram_warps_per_block_k} *
               (core.row_bytes() + fold_bytes + sizeof(vram_gathered<kit_t, warp_t>)) <=
           core.device_.shared_memory_per_block_bytes;
}

template <nk_dtype_t vector_dtype_, typename kernels_type_, typename cache_hints_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
bool vram_amd_kernels<vector_dtype_, kernels_type_, cache_hints_type_, vectors_allocator_type_,
                      neighbors_allocator_type_,
                      traversals_allocator_type_>::build_scratch_fits(core_t const& core,
                                                                      u32_t scratch_bytes_per_warp) const noexcept {
    using kit_t = typename quads_ladder<scalar_t, capability_k>::kit_t;
    using schedule_t = typename schedule_ladder<capability_k>::schedule_t;
    std::size_t const statics = std::size_t {vram_warps_per_block_k} *
                                (sizeof(vram_gathered<kit_t, typename schedule_t::warp_t>) +
                                 sizeof(typename schedule_t::shared_t));
    std::size_t const block_bytes = statics + std::size_t {vram_warps_per_block_k} * scratch_bytes_per_warp;
    // Every resident block of a builder takes its own scratch, so the most any builder keeps
    // resident must fit on one compute unit, or the launch keeps fewer blocks resident than its grid
    // and arena were sized for.
    std::size_t const resident_blocks = max_of(resident_blocks_(vram_kernel_t::insert_k, 0),
                                               resident_blocks_(vram_kernel_t::refine_k, 0));
    return block_bytes <= core.device_.shared_memory_per_block_bytes &&
           resident_blocks * block_bytes <= core.device_.shared_memory_per_multiprocessor_bytes;
}

#pragma region Kernels

/** Every AMD kernel of a walk basic graphs share: @p kernel_ picks the body, @p capability_ and
 *  @p dtype_ the kit, @p capability_ alone the schedule and its warp, @p slots_ the rung, and
 *  @p lookup_type_ is @c void, as @c amd_dense_kernel takes it. The blocks
 *  @ref hip_resident_blocks keeps per compute unit bound its registers and reach the body, which
 *  sizes its load window by them. */
template <vram_kernel_t kernel_, nk_capability_t capability_, nk_dtype_t dtype_, u32_t slots_, typename lookup_type_>
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
    else if constexpr (kernel_ == vram_kernel_t::rebase_links_k)
        vram_rebase_links<warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                  graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel_ == vram_kernel_t::mark_landmarks_k)
        vram_mark_landmarks<warp_t>(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
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
}

/** Every AMD kernel only dense indexes launch, alike in its arguments to @c amd_kernel, with
 *  @p lookup_type_ an attend's value lookup record and @c void for every other kernel. */
template <vram_dense_kernel_t kernel_, nk_capability_t capability_, nk_dtype_t dtype_, u32_t slots_,
          typename lookup_type_>
__global__ void __launch_bounds__(vram_warps_per_block_k* hip_wave_lanes(capability_),
                                  hip_waves_per_simd(hip_resident_blocks<capability_, dtype_>(kernel_, slots_)))
    amd_dense_kernel(vram_arguments_t arguments) {
    using kit_t = typename quads_ladder<typename nk::type_for<dtype_>::type, capability_>::kit_t;
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    using warp_t = typename schedule_t::warp_t;
    constexpr u32_t resident_blocks = hip_resident_blocks<capability_, dtype_>(kernel_, slots_);
    vram_graph_t const& graph = arguments.graph;
    if constexpr (kernel_ == vram_dense_kernel_t::percolate_k)
        vram_percolate<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.refine));
    else if constexpr (kernel_ == vram_dense_kernel_t::search_indexer_k) {
        // Only the floating dtypes take indexer heads; others export a plain walk never launched.
        if constexpr (floating_scalar<typename kit_t::scalar_t>)
            vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.search),
                                                                       vram_view<kit_t>(arguments.search_sink),
                                                                       vram_view<kit_t, warp_t>(arguments.rescoring));
        else
            vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
                vram_view<kit_t>(arguments.search), vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    }
    else if constexpr (kernel_ == vram_dense_kernel_t::multiply_k)
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.search),
                                                                   vram_view<kit_t, warp_t>(arguments.multiply_sink),
                                                                   vram_no_rescoring {});
    else if constexpr (kernel_ == vram_dense_kernel_t::attend_k) {
        vram_attend_t const& attend = arguments.attend;
        vram_rescoring_t const& heads = arguments.rescoring;
        lookup_type_ value_lookup;
        warp_t::copy_bytes(&value_lookup, attend.value_lookup, sizeof(lookup_type_));
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
            vram_view<kit_t>(arguments.search),
            vram_attend_sink<kit_t, warp_t, lookup_type_> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                           attend.logit_scale, attend.heads_count, attend.columns},
            vram_heads_rescoring<kit_t, warp_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads),
                                                 nullptr, heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel_ == vram_dense_kernel_t::merge_back_links_k)
        vram_merge_back_links<kit_t, schedule_t, resident_blocks>(vram_view<kit_t>(arguments.refine));
    else if constexpr (kernel_ == vram_dense_kernel_t::fold_k)
        vram_fold<kit_t, schedule_t, resident_blocks>(vram_view<kit_t>(arguments.fold));
    else if constexpr (kernel_ == vram_dense_kernel_t::derive_norms_k)
        vram_derive_norms<kit_t, warp_t>(vram_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel_ == vram_dense_kernel_t::cast_k) {
        nk_size_t const threads = static_cast<nk_size_t>(gridDim.x) * blockDim.x;
        for (nk_size_t unit = static_cast<nk_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             unit < arguments.cast.plan.units; unit += threads)
            nk_cast_unit_simt_(&arguments.cast, unit);
    }
    else if constexpr (kernel_ == vram_dense_kernel_t::rmscast_k) {
        // Only narrow floats take a unit-RMS import; every other dtype exports one never launched.
        if constexpr (dtype_ == nk_f16_k) nk_each_rmscast_f16_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e4m3_k) nk_each_rmscast_e4m3_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e5m2_k) nk_each_rmscast_e5m2_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e2m3_k) nk_each_rmscast_e2m3_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e3m2_k) nk_each_rmscast_e3m2_rocm_vectors_(arguments.rmscast);
    }
    else if constexpr (kernel_ == vram_dense_kernel_t::resolve_matches_k)
        vram_resolve_matches<warp_t>(arguments.matches.ids, arguments.matches.distances, arguments.matches.walked,
                                     arguments.matches.keys, arguments.matches.queries_count,
                                     vram_view(arguments.matches.ranking));
    else if constexpr (kernel_ == vram_dense_kernel_t::merge_matches_k)
        vram_merge_matches<warp_t>(vram_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                   arguments.matches.shards, arguments.matches.queries_count,
                                   vram_view(arguments.matches.ranking));
    else if constexpr (kernel_ == vram_dense_kernel_t::merge_attention_k)
        vram_merge_attention<warp_t>(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
                                     arguments.attend.columns, arguments.attend.shards);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, std::size_t... rungs_>
void amd_instantiate_rungs(std::index_sequence<rungs_...>) {
    (static_cast<void>(&amd_kernel<amd_rungs[rungs_].kernel, capability_, dtype_, amd_rungs[rungs_].slots, void>), ...);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, std::size_t... rungs_>
void amd_instantiate_dense_rungs(std::index_sequence<rungs_...>) {
    (static_cast<void>(
         &amd_dense_kernel<amd_dense_rungs[rungs_].kernel, capability_, dtype_, amd_dense_rungs[rungs_].slots, void>),
     ...);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, typename lookup_type_, std::size_t... rungs_>
void amd_instantiate_attends(std::index_sequence<rungs_...>) {
    (static_cast<void>(
         &amd_dense_kernel<vram_dense_kernel_t::attend_k, capability_, dtype_, amd_walk_rungs[rungs_], lookup_type_>),
     ...);
}

/** Instantiates every kernel one dtype exports under one ceiling, and its attends over each of
 *  @p lookup_types_; a unit compiling a code object names it once per compiled dtype. Host-side,
 *  since HIP-Clang refuses a kernel's address in device code, while a host reference still makes
 *  the device pass emit it. Only a device-only entry names it: on a host pass the addresses would
 *  emit launch stubs and a fat-binary registration the library never links, since it loads each
 *  code object itself. */
template <nk_capability_t capability_, nk_dtype_t dtype_, typename... lookup_types_>
void amd_instantiate() {
    amd_instantiate_rungs<capability_, dtype_>(std::make_index_sequence<sizeof(amd_rungs) / sizeof(amd_rungs[0])> {});
    amd_instantiate_dense_rungs<capability_, dtype_>(
        std::make_index_sequence<sizeof(amd_dense_rungs) / sizeof(amd_dense_rungs[0])> {});
    (amd_instantiate_attends<capability_, dtype_, lookup_types_>(
         std::make_index_sequence<sizeof(amd_walk_rungs) / sizeof(amd_walk_rungs[0])> {}),
     ...);
}

#pragma endregion Kernels

} // namespace unum::usearch::dense
