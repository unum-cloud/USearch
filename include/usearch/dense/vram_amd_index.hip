/**
 *  @file include/usearch/dense/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index's kernels: one template over the kernel, the class, the dtype and the rung.
 *
 *  Included only by the class units, which HIP-Clang compiles device-only, once per class over
 *  every compiled dtype. Each @c amd_kernel instantiation turns the plain records of
 *  @c vram_amd_index.hpp into the shared views of @c vram_index.cuh and runs the shared bodies
 *  through @ref hip_warp_t, and the host finds it by the mangled name @c amd_kernel_name
 *  spells.
 */
#pragma once

#include <cstddef>     // `std::size_t`
#include <type_traits> // `std::conditional_t` - a builder's frontier by its rung
#include <utility>     // `std::index_sequence` - every rung of one dtype

#include <numkong/types.h>       // `NUMKONG_ARCH_ROCM_`, which the two below test before including it
#include <numkong/cast/simt.cuh> // `nk_cast_unit_simt_` - the import casts' device body
#include <numkong/each/rocm.cuh> // `nk_each_rmscast_e4m3_rocm_vectors_` & kin - the unit-RMS import's device bodies

#include "usearch/dense/vram_queue.cuh"
#include "usearch/dense/vram_index.cuh"
#include "usearch/dense/vram_amd_metric.hip"
#include "usearch/dense/vram_amd_index.hpp"

namespace unum::usearch::dense {

/** The kit of one dtype on one class, under the ceiling the class grants. */
template <u32_t gpu_class_, u32_t dtype_>
using amd_kit = typename quads_chain<typename nk::type_for<static_cast<nk_dtype_t>(dtype_)>::type,
                                     amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>::kit_t;

#pragma region Views

/** The shared views and sinks, field by field from the host's records. */
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_flat_view<kit_> amd_view(amd_flat_storage_t const& storage) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t*>(storage.vectors),
            reinterpret_cast<typename kit_::norms_t*>(storage.norms),
            storage.neighbors,
            storage.connectivity,
            storage.row_fragments,
            storage.lanes_per_row,
            storage.steps_per_row,
            storage.row_fragments_magic,
            storage.kernel};
}
__device__ __forceinline__ vram_arena_shape_t amd_view(amd_arena_shape_t const& shape) noexcept {
    return {shape.visited_slots, shape.visited_hash_shift, shape.tail_entries, shape.cache_bytes};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_build_view<kit_> amd_view(amd_build_t const& build) noexcept {
    return {amd_view<kit_>(build.storage),
            reinterpret_cast<typename kit_::fragment_t const*>(build.staging),
            build.end_id,
            build.scratch_bytes_per_warp,
            build.first_id,
            build.arena,
            amd_view(build.arena_shape),
            build.expansion,
            build.epoch_base,
            build.published_rows,
            build.telemetry,
            build.assigned_rows};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_search_view<kit_> amd_view(amd_search_t const& search) noexcept {
    return {amd_view<kit_>(search.storage),
            reinterpret_cast<typename kit_::fragment_t const*>(search.queries),
            search.queries_count,
            search.sampling_bound,
            search.scratch_bytes_per_warp,
            search.expansion,
            search.arena,
            amd_view(search.arena_shape),
            search.epoch_base,
            search.telemetry,
            search.assigned_queries};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_refine_view<kit_> amd_view(amd_refine_t const& refine) noexcept {
    return {amd_view<kit_>(refine.storage),
            refine.shadow_neighbors,
            refine.end_id,
            refine.scratch_bytes_per_warp,
            refine.first_id,
            refine.arena,
            amd_view(refine.arena_shape),
            refine.expansion,
            refine.epoch_base,
            refine.telemetry,
            refine.assigned_rows,
            refine.node_ids,
            refine.sampling_bound,
            refine.first_guest};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_search_sink<kit_> amd_view(amd_search_sink_t const& sink) noexcept {
    return {sink.result_ids, sink.result_distances, sink.count, sink.kernel};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_multiply_sink<kit_> amd_view(amd_multiply_sink_t const& sink) noexcept {
    return {sink.ids, reinterpret_cast<typename kit_::dot_t*>(sink.products), sink.counts, sink.wanted};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_indexer_rescoring<kit_> amd_view(amd_rescoring_t const& rescoring) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t const*>(rescoring.heads), rescoring.weights,
            rescoring.heads_count, rescoring.row_fragments};
}
__device__ __forceinline__ vram_ranking_view amd_view(amd_ranking_t const& ranking) noexcept {
    return {ranking.keys,          ranking.scores, ranking.counts,    ranking.keys_stride,
            ranking.scores_stride, ranking.width,  ranking.key_bytes, ranking.score_bytes};
}

#pragma endregion Views

#pragma region Bodies

__device__ __forceinline__ bool amd_bitset_contains(hip_bitset_t bitset, u32_t row) noexcept {
    return (hip_warp_t::load_relaxed(bitset.words[row >> 5]) >> (row & 31u)) & 1u;
}

/** Sets the bit of @p row, answering whether this call was the one that set it. */
__device__ __forceinline__ bool amd_bitset_insert(hip_bitset_t bitset, u32_t row) noexcept {
    u32_t const bit = 1u << (row & 31u);
    return !(__hip_atomic_fetch_or(&bitset.words[row >> 5], bit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & bit);
}

/** One wavefront inserts one node, as the NVIDIA insertion kernel does. */
template <vram_quads_kit kit_, vram_schedule schedule_, u32_t pool_slots_>
__device__ __forceinline__ void amd_insert(vram_build_view<kit_> const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_vector_cache<typename kit_::fragment_t> const cache {query_span, view.storage.row_fragments};
    schedule_ schedule {schedule_states[warp], lane};
    vram_selection_scratch_t const selection = vram_selection_scratch_t::carve(cache, view.storage.connectivity);
    using frontier_t = std::conditional_t<pool_slots_ == 0, vram_arena_frontier<hip_warp_t>,
                                          vram_register_frontier<pool_slots_, hip_warp_t>>;
    while (true) {
        u32_t new_node_id = lane == 0 ? hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        new_node_id = view.first_id + hip_warp_t::shuffle(new_node_id, 0);
        if (new_node_id >= view.end_id) return;

        vram_warp_context<frontier_t> context;
        if constexpr (pool_slots_ == 0) {
            context.frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
            context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
            context.frontier.expansion_target_ = view.expansion;
        }
        context.random_generator = vram_lane_generator<hip_warp_t>(new_node_id, lane);
        vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                              view.epoch_base + (new_node_id - view.first_id) + 1};
        vram_insert_one<kit_>( //
            view, new_node_id, visited, query_span, cache, schedule, selection, cache_norms[warp], gathered[warp],
            context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<hip_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
    }
}

/** One wavefront refines one node, assigning itself node IDs from the pass counter. */
template <vram_quads_kit kit_, vram_schedule schedule_, u32_t pool_slots_>
__device__ __forceinline__ void amd_refine(vram_refine_view<kit_> const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_vector_cache<typename kit_::fragment_t> const cache {query_span, view.storage.row_fragments};
    schedule_ schedule {schedule_states[warp], lane};
    vram_selection_scratch_t const selection = vram_selection_scratch_t::carve(cache, view.storage.connectivity);
    using frontier_t = std::conditional_t<pool_slots_ == 0, vram_arena_frontier<hip_warp_t>,
                                          vram_register_frontier<pool_slots_, hip_warp_t>>;
    while (true) {
        u32_t offset = lane == 0 ? hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        offset = hip_warp_t::shuffle(offset, 0);
        // Under an ID array, end_id bounds it; the assigned offset indexes it.
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_ids ? view.node_ids[offset] : view.first_id + offset;

        vram_warp_context<frontier_t> context;
        if constexpr (pool_slots_ == 0) {
            context.frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
            context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
            context.frontier.expansion_target_ = view.expansion;
        }
        context.random_generator = vram_lane_generator<hip_warp_t>(node_id, lane);
        vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                              view.epoch_base + offset + 1};
        vram_refine_one<kit_>( //
            view, node_id, visited, query_span, cache, schedule, selection, cache_norms[warp], gathered[warp], context,
            lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<hip_warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** One wavefront per query over the register frontier, each walk scored under @p rescorings and
 *  written out through @p sink; wavefronts take queries from the batch counter, as builders take
 *  rows. */
template <vram_quads_kit kit_, u32_t pool_slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void amd_walk(vram_search_view<kit_> const& view, sink_ const& sink,
                                         rescoring_ const& rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    while (true) {
        u32_t query_index = slice.lane == 0 ? hip_warp_t::fetch_add_relaxed(*view.assigned_queries, 1u) : 0;
        query_index = hip_warp_t::shuffle(query_index, 0);
        if (query_index >= view.queries_count) return;
        vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                              view.epoch_base + query_index + 1};
        vram_walk_one<kit_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, sink, rescorings, query_index, query_span, visited,
            vram_register_frontier<pool_slots_, hip_warp_t> {}, gathered[slice.warp], slice.lane);
    }
}

/** One wavefront per query past the register ladder, over the arena frontier. */
template <vram_quads_kit kit_, typename sink_, typename rescoring_>
__device__ __forceinline__ void amd_walk_wide(vram_search_view<kit_> const& view, sink_ const& sink,
                                              rescoring_ const& rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_arena_frontier<hip_warp_t> frontier;
    frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    while (true) {
        u32_t query_index = slice.lane == 0 ? hip_warp_t::fetch_add_relaxed(*view.assigned_queries, 1u) : 0;
        query_index = hip_warp_t::shuffle(query_index, 0);
        if (query_index >= view.queries_count) return;
        vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                              view.epoch_base + query_index + 1};
        vram_walk_one<kit_, 4u>(view, sink, rescorings, query_index, query_span, visited, frontier,
                                gathered[slice.warp], slice.lane);
    }
}

/** One wavefront merges one touched node's forward and reverse candidates into its final
 *  neighbors. */
template <vram_quads_kit kit_, vram_schedule schedule_>
__device__ __forceinline__ void amd_merge_links(vram_refine_view<kit_> const& view, u32_t const* bucket_ids,
                                                u32_t const* bucket_counts, u32_t bucket_width, hip_bitset_t touched) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_vector_cache<typename kit_::fragment_t> const cache {
        reinterpret_cast<typename kit_::fragment_t*>(slice.scratch), view.storage.row_fragments};
    schedule_ schedule {schedule_states[warp], lane};
    while (true) {
        u32_t node_id = lane == 0 ? hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        node_id = view.first_id + hip_warp_t::shuffle(node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the walk neither re-selected nor picked keeps its neighbors: re-judging it could
        // only drop links.
        if (!amd_bitset_contains(touched, node_id)) continue;

        vram_telemetry_t telemetry;
        vram_merge_links_one<kit_>( //
            view, bucket_ids, bucket_counts, bucket_width, node_id, cache, schedule, cache_norms[warp], gathered[warp],
            telemetry, lane);
        telemetry.template flush<hip_warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** Scatters the walked rows' shadow neighbors in reverse, marking every row the merge
 *  re-selects. */
__device__ __forceinline__ void amd_scatter_links(u32_t const* shadow_neighbors, u32_t rows_count, u32_t first_id,
                                                  u32_t walk_count, u32_t const* node_ids, u32_t connectivity,
                                                  u32_t bucket_width, u32_t* bucket_ids, u32_t* bucket_counts,
                                                  hip_bitset_t touched) {
    vram_grid_warp_t const warp = vram_grid_warp<hip_warp_t>();
    for (u32_t offset = warp.index; offset < walk_count; offset += warp.count) {
        u32_t const row = node_ids ? node_ids[offset] : first_id + offset;
        if (warp.lane == 0) amd_bitset_insert(touched, row);
        if (node_ids) continue;
        vram_for_each_link<hip_warp_t>(
            shadow_neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
                amd_bitset_insert(touched, target);
                u32_t const landing = hip_warp_t::fetch_add_relaxed(bucket_counts[target], 1u);
                u32_t* const bucket = vram_row(bucket_ids, target, bucket_width);
                if (landing < bucket_width) bucket[landing] = row;
            });
    }
}

/** Rebuilds the norms column for a run of rows, one wavefront per row. */
template <vram_quads_kit kit_>
__device__ __forceinline__ void amd_derive_norms(vram_flat_view<kit_> const& storage, u32_t first_row,
                                                 u32_t rows_count) {
    vram_grid_warp_t const warp = vram_grid_warp<hip_warp_t>();
    for (u32_t row = first_row + warp.index; row < first_row + rows_count; row += warp.count) {
        typename kit_::norms_accumulator_t accumulator {};
        for (u32_t fragment = warp.lane; fragment < storage.row_fragments; fragment += hip_warp_t::width_k)
            accumulator = kit_::accumulate_norms(storage.vector_for(row, fragment), accumulator);
        accumulator = vram_reduce_norms<hip_warp_t>(accumulator);
        if (warp.lane == 0) storage.norm_ref_for(row) = kit_::partials_to_norms(accumulator);
    }
}

/** Marks the stride-spread landmarks reached and in the first frontier. */
__device__ __forceinline__ void amd_mark_seeds(hip_bitset_t reached, hip_bitset_t frontier, u32_t rows_count,
                                               u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    amd_bitset_insert(reached, row);
    amd_bitset_insert(frontier, row);
}

/** One level of forward reachability from the frontier; a level whose @p expanded_before reads zero
 *  was queued past convergence and has nothing to expand. */
__device__ __forceinline__ void amd_reachability_sweep(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                       hip_bitset_t reached, hip_bitset_t frontier,
                                                       hip_bitset_t next_frontier, u32_t* expanded,
                                                       u32_t const* expanded_before) {
    // The level before landed ahead of this launch, so every thread reads one settled word.
    if (expanded_before && !*expanded_before) return;
    vram_grid_warp_t const warp = vram_grid_warp<hip_warp_t>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!amd_bitset_contains(frontier, row)) continue;
        if (warp.lane == 0) hip_warp_t::fetch_add_relaxed(*expanded, 1u);
        vram_for_each_link<hip_warp_t>(neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
            if (amd_bitset_insert(reached, target)) amd_bitset_insert(next_frontier, target);
        });
    }
}

/** Counts and collects the rows the sweep never reached. */
__device__ __forceinline__ void amd_reachability_islands(hip_bitset_t reached, u32_t rows_count, u32_t* island_ids,
                                                         u32_t* islands_cursor) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count || amd_bitset_contains(reached, row)) return;
    u32_t const slot = hip_warp_t::fetch_add_relaxed(*islands_cursor, 1u);
    if (island_ids) island_ids[slot] = row;
}

/** Counts, for every row, the reached rows pointing at it. */
__device__ __forceinline__ void amd_reached_in_links(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                     hip_bitset_t reached, u32_t* in_links) {
    vram_grid_warp_t const warp = vram_grid_warp<hip_warp_t>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!amd_bitset_contains(reached, row)) continue;
        vram_for_each_link<hip_warp_t>(neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
            hip_warp_t::fetch_add_relaxed(in_links[target], 1u);
        });
    }
}

/** Reconnects each island into a reached pick or the landmark covering it. */
__device__ __forceinline__ void amd_reconnect_island(u32_t* neighbors, u32_t connectivity, u32_t const* island_ids,
                                                     u32_t islands_count, hip_bitset_t reached, u32_t rows_count,
                                                     u32_t seed_count, u32_t* in_links, u32_t* reconnected) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const spread = heal_spread(connectivity);
    auto const fetch_sub = [](u32_t& word, u32_t value) noexcept {
        return __hip_atomic_fetch_sub(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    };

    // Whether the island now sits in the target's row, landing it there when a slot allows.
    auto const land = [&](u32_t target) noexcept {
        u32_t* const target_row = vram_row(neighbors, target, connectivity);
        auto const claim = [&](u32_t slot, u32_t resident) noexcept {
            if (!__hip_atomic_compare_exchange_strong(&target_row[slot], &resident, island, __ATOMIC_RELAXED,
                                                      __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                return false;
            hip_warp_t::fetch_add_relaxed(*reconnected, 1u);
            return true;
        };
        for (u32_t slot = 0; slot != connectivity; ++slot) {
            u32_t const resident = hip_warp_t::load_relaxed(target_row[slot]);
            if (resident == island) return true;
            if (resident != node_id_unknown_k && resident != target) continue;
            if (claim(slot, resident)) return true;
        }
        for (u32_t attempt = 0; attempt != spread; ++attempt) {
            u32_t const landing = connectivity - 1 - (island_index + attempt) % spread;
            u32_t const resident = hip_warp_t::load_relaxed(target_row[landing]);
            if (resident == island) return true;
            if (resident != node_id_unknown_k && !amd_bitset_contains(reached, resident)) continue;
            bool const evicting = resident != node_id_unknown_k && resident != target;
            if (evicting && fetch_sub(in_links[resident], 1u) <= 1u) {
                hip_warp_t::fetch_add_relaxed(in_links[resident], 1u);
                continue;
            }
            if (claim(landing, resident)) return true;
            if (evicting) hip_warp_t::fetch_add_relaxed(in_links[resident], 1u);
        }
        return false;
    };

    u32_t const* const island_row = vram_row(neighbors, island, connectivity);
    for (u32_t slot = 0; slot != connectivity; ++slot) {
        u32_t const pick = island_row[slot];
        if (pick != node_id_unknown_k && pick != island && amd_bitset_contains(reached, pick) && land(pick)) return;
    }
    if (!seed_count) return;
    u32_t const seed = static_cast<u32_t>(static_cast<u64_t>(island) * seed_count / rows_count);
    u32_t const landmark = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    if (landmark != island) land(landmark);
}

#pragma endregion Bodies

#pragma region Kernels

/** Runs one walk family on its rung, the arena frontier at slot zero. */
template <vram_quads_kit kit_, u32_t slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void amd_walk_rung(vram_search_view<kit_> const& view, sink_ const& sink,
                                              rescoring_ const& rescorings) {
    if constexpr (slots_ == 0) amd_walk_wide<kit_>(view, sink, rescorings);
    else amd_walk<kit_, slots_>(view, sink, rescorings);
}

/** Every AMD kernel: @p kernel_ picks the body, @p gpu_class_ and @p dtype_ the kit, @p gpu_class_
 *  alone the staging schedule, @p slots_ the rung. */
template <u32_t kernel_, u32_t gpu_class_, u32_t dtype_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k* hip_warp_t::width_k) amd_kernel(amd_arguments_t arguments) {
    using kit_t = amd_kit<gpu_class_, dtype_>;
    using schedule_t = hip_schedule_chain<amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>;
    constexpr amd_kernel_t kernel = static_cast<amd_kernel_t>(kernel_);
    amd_graph_t const& graph = arguments.graph;
    if constexpr (kernel == amd_kernel_t::insert_k)
        amd_insert<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.build));
    else if constexpr (kernel == amd_kernel_t::refine_k)
        amd_refine<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.refine));
    else if constexpr (kernel == amd_kernel_t::search_k)
        amd_walk_rung<kit_t, slots_>(amd_view<kit_t>(arguments.search), amd_view<kit_t>(arguments.search_sink),
                                     vram_no_rescoring {});
    else if constexpr (kernel == amd_kernel_t::search_indexer_k)
        amd_walk_rung<kit_t, slots_>(amd_view<kit_t>(arguments.search), amd_view<kit_t>(arguments.search_sink),
                                     amd_view<kit_t>(arguments.rescoring));
    else if constexpr (kernel == amd_kernel_t::multiply_k)
        amd_walk_rung<kit_t, slots_>(amd_view<kit_t>(arguments.search), amd_view<kit_t>(arguments.multiply_sink),
                                     vram_no_rescoring {});
    else if constexpr (kernel == amd_kernel_t::merge_links_k)
        amd_merge_links<kit_t, schedule_t>(amd_view<kit_t>(arguments.refine), graph.bucket_ids, graph.bucket_counts,
                                           graph.bucket_width, graph.touched);
    else if constexpr (kernel == amd_kernel_t::scatter_links_k)
        amd_scatter_links(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count, graph.node_ids,
                          graph.connectivity, graph.bucket_width, graph.bucket_ids, graph.bucket_counts, graph.touched);
    else if constexpr (kernel == amd_kernel_t::derive_norms_k)
        amd_derive_norms<kit_t>(amd_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel == amd_kernel_t::mark_seeds_k)
        amd_mark_seeds(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel == amd_kernel_t::reachability_sweep_k)
        amd_reachability_sweep(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached, graph.frontier,
                               graph.next_frontier, graph.counter, graph.previous_counter);
    else if constexpr (kernel == amd_kernel_t::reachability_islands_k)
        amd_reachability_islands(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel == amd_kernel_t::reached_in_links_k)
        amd_reached_in_links(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached, graph.in_links);
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
        vram_resolve_matches<hip_warp_t>(arguments.matches.ids, arguments.matches.distances, arguments.matches.walked,
                                         arguments.matches.keys, arguments.matches.queries_count,
                                         amd_view(arguments.matches.ranking));
    else if constexpr (kernel == amd_kernel_t::merge_matches_k)
        vram_merge_matches<hip_warp_t>(amd_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                       arguments.matches.shards, arguments.matches.queries_count,
                                       amd_view(arguments.matches.ranking));
    else
        amd_reconnect_island(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count, graph.reached,
                             graph.rows_count, graph.seed_count, graph.in_links, graph.reconnected);
}

template <u32_t gpu_class_, u32_t dtype_, std::size_t... rungs_>
void amd_instantiate_rungs(std::index_sequence<rungs_...>) {
    (static_cast<void>(
         &amd_kernel<static_cast<u32_t>(amd_rungs[rungs_].kernel), gpu_class_, dtype_, amd_rungs[rungs_].slots>),
     ...);
}

/** Instantiates every kernel one dtype exports on one class; a class unit names it once per
 *  compiled dtype. Host-side, since HIP-Clang refuses a kernel's address in device code, while a
 *  host reference still makes the device pass emit it. Only the device pass takes the addresses:
 *  on the host pass they would emit launch stubs and a fat-binary registration the library never
 *  links, since it loads the class's code object itself. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
void amd_instantiate() {
#if defined(__HIP_DEVICE_COMPILE__)
    amd_instantiate_rungs<static_cast<u32_t>(gpu_class_), static_cast<u32_t>(dtype_)>(
        std::make_index_sequence<sizeof(amd_rungs) / sizeof(amd_rungs[0])> {});
#endif
}

#pragma endregion Kernels

} // namespace unum::usearch::dense
