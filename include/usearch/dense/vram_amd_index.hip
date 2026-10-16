/**
 *  @file include/usearch/dense/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index's kernels: one template over the kernel, the class, the dtype, the rung and
 *      the attend value lookup.
 *
 *  Included only by the class units, which HIP-Clang compiles device-only, once per class over
 *  every compiled dtype. Each @c amd_kernel instantiation turns the plain records of
 *  @c vram_index.hpp into the shared views of @c vram_index.cuh and runs the shared bodies there
 *  through the class's @ref hip_warp, and the host finds it by the mangled name @c amd_kernel_name
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

/** The kit @c quads_chain picks for @p dtype_ under the ceiling @p gpu_class_ grants, checked
 *  against the host-visible half the face sizes its launches by. */
template <u32_t gpu_class_, u32_t dtype_>
struct amd_kit_of {
    using traits_t =
        amd_kit_traits<static_cast<nk_dtype_t>(dtype_), amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>;
    using type = typename quads_chain<typename traits_t::scalar_t,
                                      amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>::kit_t;
    static_assert(std::is_same_v<typename traits_t::scalar_t, typename type::scalar_t> &&
                  sizeof(typename type::fragment_t) == 16 &&
                  std::is_same_v<typename traits_t::dot_t, typename type::dot_t> &&
                  std::is_same_v<typename traits_t::norms_t, typename type::norms_t>);
    static_assert(traits_t::picked_k == type::capability_k);
    static_assert(traits_t::wave_lanes_k == type::warp_t::width_k);
    static_assert(traits_t::gathered_bytes_k == sizeof(vram_gathered<type>));
};

#pragma region Kernels

/** Every AMD kernel: @p kernel_ picks the body, @p gpu_class_ and @p dtype_ the kit, @p gpu_class_
 *  alone the schedule, @p slots_ the rung, and @p lookup_ an attend's value lookup through
 *  @ref amd_value_lookup_of. */
template <u32_t kernel_, u32_t gpu_class_, u32_t dtype_, u32_t slots_, u32_t lookup_>
__global__ void __launch_bounds__(vram_warps_per_block_k* hip_wave_lanes(static_cast<gpu_class_t>(gpu_class_)))
    amd_kernel(vram_arguments_t arguments) {
    using warp_t = hip_warp_of<amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>;
    using kit_t = typename amd_kit_of<gpu_class_, dtype_>::type;
    static_assert(std::is_same_v<typename kit_t::warp_t, warp_t>);
    using schedule_t = rocm_schedule_of<amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>;
    constexpr amd_kernel_t kernel = static_cast<amd_kernel_t>(kernel_);
    vram_graph_t const& graph = arguments.graph;
    if constexpr (kernel == amd_kernel_t::insert_k)
        vram_insert<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel == amd_kernel_t::back_link_k)
        vram_back_link<kit_t, schedule_t>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel == amd_kernel_t::refine_k)
        vram_refine<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.refine));
    else if constexpr (kernel == amd_kernel_t::search_k)
        vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                  vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    else if constexpr (kernel == amd_kernel_t::search_indexer_k) {
        // Only the floating dtypes take indexer heads; others export a plain walk never launched.
        if constexpr (floating_scalar<typename kit_t::scalar_t>)
            vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                      vram_view<kit_t>(arguments.search_sink),
                                                      vram_view<kit_t>(arguments.rescoring));
        else
            vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                      vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    }
    else if constexpr (kernel == amd_kernel_t::multiply_k)
        vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                  vram_view<kit_t>(arguments.multiply_sink), vram_no_rescoring {});
    else if constexpr (kernel == amd_kernel_t::attend_k) {
        using value_lookup_t = typename amd_value_lookup_of<lookup_>::type;
        vram_attend_t const& attend = arguments.attend;
        vram_rescoring_t const& heads = arguments.rescoring;
        value_lookup_t value_lookup;
        warp_t::copy_bytes(&value_lookup, attend.value_lookup, sizeof(value_lookup_t));
        vram_walk_rung<kit_t, schedule_t, slots_>(
            vram_view<kit_t>(arguments.search),
            vram_attend_sink<kit_t, value_lookup_t> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                     attend.logit_scale, attend.heads_count, attend.columns},
            vram_heads_rescoring<kit_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads), nullptr,
                                         heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel == amd_kernel_t::merge_back_links_k)
        vram_merge_back_links<kit_t, schedule_t>(vram_view<kit_t>(arguments.refine), graph.bucket_ids,
                                                 graph.bucket_counts, graph.bucket_width, graph.touched);
    else if constexpr (kernel == amd_kernel_t::scatter_back_links_k)
        vram_scatter_back_links<warp_t>(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count,
                                        graph.node_ids, graph.connectivity, graph.bucket_width, graph.bucket_ids,
                                        graph.bucket_counts, graph.touched);
    else if constexpr (kernel == amd_kernel_t::derive_norms_k)
        vram_derive_norms<kit_t>(vram_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel == amd_kernel_t::rebase_links_k)
        vram_rebase_links<warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                  graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel == amd_kernel_t::mark_seeds_k)
        vram_mark_seeds<warp_t>(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel == amd_kernel_t::reachability_sweep_k)
        vram_reachability_sweep<warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                        graph.frontier, graph.next_frontier, graph.counter, graph.previous_counter,
                                        graph.levels);
    else if constexpr (kernel == amd_kernel_t::reachability_islands_k)
        vram_reachability_islands<warp_t>(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel == amd_kernel_t::reached_in_links_k)
        vram_reached_in_links<warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                      graph.in_links);
    else if constexpr (kernel == amd_kernel_t::reconnect_island_k)
        vram_reconnect_island<warp_t>(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count,
                                      graph.reached, graph.rows_count, graph.seed_count, graph.in_links,
                                      graph.reconnected);
    else if constexpr (kernel == amd_kernel_t::cast_k) {
        nk_size_t const threads = static_cast<nk_size_t>(gridDim.x) * blockDim.x;
        for (nk_size_t unit = static_cast<nk_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             unit < arguments.cast.plan.units; unit += threads)
            nk_cast_unit_simt_(&arguments.cast, unit);
    }
    else if constexpr (kernel == amd_kernel_t::rmscast_k) {
        // Only narrow floats take a unit-RMS import; every other dtype exports one never launched.
        if constexpr (dtype_ == nk_f16_k) nk_each_rmscast_f16_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e4m3_k) nk_each_rmscast_e4m3_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e5m2_k) nk_each_rmscast_e5m2_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e2m3_k) nk_each_rmscast_e2m3_rocm_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e3m2_k) nk_each_rmscast_e3m2_rocm_vectors_(arguments.rmscast);
    }
    else if constexpr (kernel == amd_kernel_t::resolve_matches_k)
        vram_resolve_matches<warp_t>(arguments.matches.ids, arguments.matches.distances, arguments.matches.walked,
                                     arguments.matches.keys, arguments.matches.queries_count,
                                     vram_view(arguments.matches.ranking));
    else if constexpr (kernel == amd_kernel_t::merge_matches_k)
        vram_merge_matches<warp_t>(vram_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                   arguments.matches.shards, arguments.matches.queries_count,
                                   vram_view(arguments.matches.ranking));
    else
        vram_merge_attention<warp_t>(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
                                     arguments.attend.columns, arguments.attend.shards);
}

template <u32_t gpu_class_, u32_t dtype_, std::size_t... rungs_>
void amd_instantiate_rungs(std::index_sequence<rungs_...>) {
    (static_cast<void>(
         &amd_kernel<static_cast<u32_t>(amd_rungs[rungs_].kernel), gpu_class_, dtype_, amd_rungs[rungs_].slots, 0>),
     ...);
}

template <u32_t gpu_class_, u32_t dtype_, u32_t lookup_, std::size_t... rungs_>
void amd_instantiate_attends(std::index_sequence<rungs_...>) {
    (static_cast<void>(
         &amd_kernel<static_cast<u32_t>(amd_kernel_t::attend_k), gpu_class_, dtype_, amd_walk_rungs[rungs_], lookup_>),
     ...);
}

/** Instantiates every kernel one dtype exports on one class, and its attends over each of
 *  @p lookups_; a unit compiling a code object names it once per compiled dtype. Host-side, since
 *  HIP-Clang refuses a kernel's address in device code, while a host reference still makes the
 *  device pass emit it. Only the device pass takes the addresses: on the host pass they would emit
 *  launch stubs and a fat-binary registration the library never links, since it loads the class's
 *  code object itself. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_, u32_t... lookups_>
void amd_instantiate() {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr u32_t gpu_class = static_cast<u32_t>(gpu_class_), dtype = static_cast<u32_t>(dtype_);
    amd_instantiate_rungs<gpu_class, dtype>(std::make_index_sequence<sizeof(amd_rungs) / sizeof(amd_rungs[0])> {});
    (amd_instantiate_attends<gpu_class, dtype, lookups_>(
         std::make_index_sequence<sizeof(amd_walk_rungs) / sizeof(amd_walk_rungs[0])> {}),
     ...);
#endif
}

#pragma endregion Kernels

} // namespace unum::usearch::dense
