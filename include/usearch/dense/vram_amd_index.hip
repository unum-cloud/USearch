/**
 *  @file include/usearch/dense/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index's kernels: one template over the kernel, the class, the dtype and the rung.
 *
 *  Included only by the class units, which HIP-Clang compiles device-only, once per class over
 *  every compiled dtype. Each @c dense_amd_kernel instantiation turns the plain records of
 *  @c vram_amd_index.hpp into the shared views of @c vram_index.cuh and runs the shared bodies
 *  through @ref hip_warp_t, and the host finds it by the mangled name @c dense_amd_kernel_name
 *  spells.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <utility> // `std::index_sequence` - every rung of one dtype

#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/vram_index.cuh"
#include "usearch/dense/vram_amd_metric.hip"
#include "usearch/dense/vram_amd_index.hpp"

namespace unum {
namespace usearch {

/** The kit of one dtype on one class, under the ceiling the class grants. */
template <u32_t gpu_class_, u32_t dtype_>
using dense_amd_kit =
    typename dense_quads_chain<typename nk::type_for<static_cast<nk_dtype_t>(dtype_)>::type,
                               dense_amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>::kit_t;

#pragma region Views

/** The shared views and sinks, field by field from the host's records. */
template <typename kit_>
__device__ __forceinline__ dense_vram_planes_view<kit_> dense_amd_view(dense_amd_planes_t const& planes) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t*>(planes.vectors),
            reinterpret_cast<typename kit_::norms_t*>(planes.norms),
            planes.neighbors,
            planes.connectivity,
            planes.row_fragments,
            planes.lanes_per_row,
            planes.steps_per_row,
            planes.row_fragments_magic,
            planes.kernel};
}
__device__ __forceinline__ dense_vram_arena_shape_t dense_amd_view(dense_amd_arena_shape_t const& shape) noexcept {
    return {shape.visited_slots, shape.visited_hash_shift, shape.tail_entries};
}
template <typename kit_>
__device__ __forceinline__ dense_vram_build_view<kit_> dense_amd_view(dense_amd_build_t const& build) noexcept {
    return {dense_amd_view<kit_>(build.planes),
            reinterpret_cast<typename kit_::fragment_t const*>(build.staging),
            build.end_id,
            build.scratch_bytes_per_warp,
            build.first_id,
            build.arena,
            dense_amd_view(build.arena_shape),
            build.epoch_base,
            build.published_rows,
            build.telemetry,
            build.assigned_rows};
}
template <typename kit_>
__device__ __forceinline__ dense_vram_search_view<kit_> dense_amd_view(dense_amd_search_t const& search) noexcept {
    return {dense_amd_view<kit_>(search.planes),
            reinterpret_cast<typename kit_::fragment_t const*>(search.queries),
            search.queries_count,
            search.sampling_bound,
            search.scratch_bytes_per_warp,
            search.expansion,
            search.arena,
            dense_amd_view(search.arena_shape),
            search.epoch_base,
            search.telemetry};
}
template <typename kit_>
__device__ __forceinline__ dense_vram_refine_view<kit_> dense_amd_view(dense_amd_refine_t const& refine) noexcept {
    return {dense_amd_view<kit_>(refine.planes),
            refine.shadow_neighbors,
            refine.end_id,
            refine.scratch_bytes_per_warp,
            refine.first_id,
            refine.arena,
            dense_amd_view(refine.arena_shape),
            refine.epoch_base,
            refine.telemetry,
            refine.assigned_rows,
            refine.node_ids,
            refine.sampling_bound};
}
template <typename kit_>
__device__ __forceinline__ dense_vram_search_sink<kit_> dense_amd_view(dense_amd_search_sink_t const& sink) noexcept {
    return {sink.result_ids, sink.result_distances, sink.count, sink.kernel};
}
template <typename kit_>
__device__ __forceinline__ dense_vram_multiply_sink<kit_> dense_amd_view(
    dense_amd_multiply_sink_t const& sink) noexcept {
    return {sink.ids, reinterpret_cast<typename kit_::dot_t*>(sink.products), sink.counts, sink.wanted};
}
template <typename kit_>
__device__ __forceinline__ dense_vram_indexer_rescoring<kit_> dense_amd_view(
    dense_amd_rescoring_t const& rescoring) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t const*>(rescoring.heads), rescoring.weights,
            rescoring.heads_count, rescoring.row_fragments};
}

#pragma endregion Views

#pragma region Bodies

__device__ __forceinline__ bool dense_amd_bitset_contains(dense_hip_bitset_t bitset, u32_t row) noexcept {
    return (hip_warp_t::load_relaxed(bitset.words[row >> 5]) >> (row & 31u)) & 1u;
}

/** Sets the bit of @p row, answering whether this call was the one that set it. */
__device__ __forceinline__ bool dense_amd_bitset_insert(dense_hip_bitset_t bitset, u32_t row) noexcept {
    u32_t const bit = 1u << (row & 31u);
    return !(__hip_atomic_fetch_or(&bitset.words[row >> 5], bit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & bit);
}

/** The calling wavefront's place in a launch: its global index, its lane, and the wavefronts in the
 *  grid. */
struct dense_amd_wave_t {
    std::size_t index;
    int lane;
    std::size_t count;
};
__device__ __forceinline__ dense_amd_wave_t dense_amd_wave() noexcept {
    return {(blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / hip_warp_t::width_k,
            static_cast<int>(threadIdx.x % hip_warp_t::width_k),
            gridDim.x * static_cast<std::size_t>(blockDim.x) / hip_warp_t::width_k};
}

/** One wavefront inserts one node, as the NVIDIA insertion kernel does. */
template <typename kit_, u32_t pool_slots_>
__device__ __forceinline__ void dense_amd_insert(dense_vram_build_view<kit_> const& view) {
    using kit_t = kit_;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_t::norms_t cache_norms[dense_vram_warps_per_block_k][dense_vram_cache_rows_k];
    __shared__ dense_vram_gathered<kit_t> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_vram_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_vram_selection_scratch_t const selection = dense_vram_selection_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_vram_cache_rows_k);
    while (true) {
        u32_t new_node_id = 0;
        if (lane == 0) new_node_id = view.first_id + hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u);
        new_node_id = hip_warp_t::shuffle(new_node_id, 0);
        if (new_node_id >= view.end_id) return;
        while (true) { // Admission ramp - the only bootstrap throttle.
            u32_t const published = hip_warp_t::load_relaxed(*view.published_rows);
            if (new_node_id < published + max_of(published, admission_ramp_floor_k)) break;
            hip_warp_t::sleep(2048);
        }
        dense_vram_warp_context<dense_vram_register_frontier<pool_slots_, hip_warp_t>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(new_node_id) * 33 + lane)};
        dense_vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                    view.epoch_base + (new_node_id - view.first_id) + 1};
        dense_vram_insert_one<kit_t>( //
            view, new_node_id, visited, query_span, cache, selection, cache_norms[warp], gathered[warp], context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<hip_warp_t, dense_vram_traversal_t::inserting_k>(view.telemetry, lane);
    }
}

/** One wavefront refines one node, assigning itself node IDs from the pass counter. */
template <typename kit_, u32_t pool_slots_>
__device__ __forceinline__ void dense_amd_refine(dense_vram_refine_view<kit_> const& view) {
    using kit_t = kit_;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_t::norms_t cache_norms[dense_vram_warps_per_block_k][dense_vram_cache_rows_k];
    __shared__ dense_vram_gathered<kit_t> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_vram_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_vram_selection_scratch_t const selection = dense_vram_selection_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_vram_cache_rows_k);
    while (true) {
        u32_t offset = 0;
        if (lane == 0) offset = hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u);
        offset = hip_warp_t::shuffle(offset, 0);
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_ids ? view.node_ids[offset] : view.first_id + offset;
        dense_vram_warp_context<dense_vram_register_frontier<pool_slots_, hip_warp_t>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(node_id) * 33 + lane)};
        dense_vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                    view.epoch_base + offset + 1};
        dense_vram_refine_one<kit_t>( //
            view, node_id, visited, query_span, cache, selection, cache_norms[warp], gathered[warp], context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<hip_warp_t, dense_vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** One wavefront per query over the register frontier, each walk scored under @p rescorings and
 *  written out through @p sink. */
template <typename kit_, u32_t pool_slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void dense_amd_walk(dense_vram_search_view<kit_> const& view, sink_ const& sink,
                                               rescoring_ const& rescorings) {
    using kit_t = kit_;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ dense_vram_gathered<kit_t> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    for (u32_t query_index = static_cast<u32_t>(slice.warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_vram_warps_per_block_k) {
        dense_vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                    view.epoch_base + query_index + 1};
        dense_vram_walk_one<kit_t, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, sink, rescorings, query_index, query_span, visited,
            dense_vram_register_frontier<pool_slots_, hip_warp_t> {}, gathered[slice.warp], slice.lane);
    }
}

/** One wavefront per query past the register ladder, over the arena frontier. */
template <typename kit_, typename sink_, typename rescoring_>
__device__ __forceinline__ void dense_amd_walk_wide(dense_vram_search_view<kit_> const& view, sink_ const& sink,
                                                    rescoring_ const& rescorings) {
    using kit_t = kit_;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ dense_vram_gathered<kit_t> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_vram_arena_frontier<hip_warp_t> frontier;
    frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    for (u32_t query_index = static_cast<u32_t>(slice.warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_vram_warps_per_block_k) {
        dense_vram_visited_set<hip_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                    view.epoch_base + query_index + 1};
        dense_vram_walk_one<kit_t, 4u>(view, sink, rescorings, query_index, query_span, visited, frontier,
                                       gathered[slice.warp], slice.lane);
    }
}

/** Hands @p callback every neighbor of @p row the calling lane owns - its lane-strided slots, past
 *  sentinels, self-loops and IDs at or past @p rows_count. */
template <typename callback_type_>
__device__ __forceinline__ void dense_amd_for_each_link(u32_t const* neighbors, u32_t connectivity, u32_t rows_count,
                                                        u32_t row, int lane, callback_type_&& callback) noexcept {
    u32_t const* const slots = neighbors + static_cast<std::size_t>(row) * connectivity;
    for (u32_t slot = static_cast<u32_t>(lane); slot < connectivity; slot += hip_warp_t::width_k)
        if (u32_t const target = slots[slot]; target != dense_node_id_unknown_k && target != row && target < rows_count)
            callback(target);
}

/** Selects one node's final neighbors from its forward picks and its reverse bucket, as the NVIDIA
 *  merge does: the diversity heuristic over the union ascending by distance, backfilled with the
 *  nearest pruned ones. */
template <typename kit_>
__device__ __forceinline__ void dense_amd_merge_links_one( //
    dense_vram_refine_view<kit_> const& view, u32_t const* bucket_ids, u32_t const* bucket_counts, u32_t bucket_width,
    dense_node_id_t node_id, dense_vram_vector_cache<typename kit_::fragment_t> cache,
    dense_vram_scored_node_t* sorted_span, u32_t* kept_positions,
    typename kit_::norms_t (&cache_norms)[dense_vram_cache_rows_k], dense_vram_gathered<kit_>& gathered,
    dense_vram_telemetry_t& telemetry, int lane) noexcept {
    using kit_t = kit_;
    using mask_t = typename hip_warp_t::mask_t;
    constexpr u32_t lanes_k = hip_warp_t::width_k;
    dense_vram_planes_view<kit_t> const& planes = view.planes;
    constexpr u32_t query_row_k = dense_vram_cache_rows_k - 1;
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += lanes_k)
        cache.at(query_row_k, fragment) = planes.vector_for(node_id, fragment);
    typename kit_t::norms_t const query_state = planes.norm_for(node_id);
    hip_warp_t::sync();

    u32_t candidates_count = 0;
    u32_t const* const shadow_slots = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += lanes_k) {
        u32_t const slot = slot_base + lane;
        u32_t const forward_id = slot < planes.connectivity ? shadow_slots[slot] : dense_node_id_unknown_k;
        bool const usable = forward_id != dense_node_id_unknown_k && forward_id != node_id;
        mask_t const usable_mask = hip_warp_t::ballot(usable);
        u32_t const ordinal = hip_warp_t::count(usable_mask & hip_warp_t::below(lane));
        if (usable) gathered.ids[candidates_count + ordinal] = forward_id;
        candidates_count += hip_warp_t::count(usable_mask);
        hip_warp_t::sync();
    }
    u32_t const forward_count = candidates_count;

    u32_t const landed = min_of(bucket_counts[node_id], bucket_width);
    for (u32_t slot_base = 0; slot_base < landed && candidates_count < query_row_k; slot_base += lanes_k) {
        u32_t const slot = slot_base + lane;
        u32_t const reverse_id = slot < landed ? bucket_ids[static_cast<std::size_t>(node_id) * bucket_width + slot]
                                               : dense_node_id_unknown_k;
        bool usable = reverse_id != dense_node_id_unknown_k && reverse_id != node_id;
        for (u32_t forward_index = 0; usable && forward_index != forward_count; ++forward_index)
            usable = gathered.ids[forward_index] != reverse_id;
        mask_t const usable_mask = hip_warp_t::ballot(usable);
        u32_t const ordinal = hip_warp_t::count(usable_mask & hip_warp_t::below(lane));
        if (usable && candidates_count + ordinal < query_row_k) gathered.ids[candidates_count + ordinal] = reverse_id;
        candidates_count = min_of(candidates_count + hip_warp_t::count(usable_mask), query_row_k);
        hip_warp_t::sync();
    }

    telemetry.scored += candidates_count;
    dense_vram_sorted_buffer<hip_warp_t> sorted_candidates {sorted_span, 0, query_row_k};
    for (u32_t base = 0; base < candidates_count; base += lanes_k) {
        u32_t const batch_count = min_of(candidates_count - base, lanes_k);
        dense_vram_scored_node_t const scored = dense_vram_score_batch<kit_t>( //
            planes, cache.row(query_row_k), query_state, gathered.ids + base, batch_count, gathered, lane);
        dense_vram_scored_node_t const sorted_batch = dense_sort_lanes<hip_warp_t>(
            static_cast<u32_t>(lane) < batch_count ? dense_vram_scored_node_t::make(scored.distance(), base + lane)
                                                   : dense_vram_scored_node_t::sentinel(),
            lane);
        sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
    }

    u32_t const candidate_words = candidates_count * planes.row_fragments;
    for (u32_t word = lane; word < candidate_words; word += lanes_k) {
        u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
        hip_warp_t::copy_async(&cache.at(cache_row, fragment),
                               planes.vector_span_for(gathered.ids[cache_row]) + fragment);
    }
    hip_warp_t::copy_wait();
    for (u32_t cache_row = lane; cache_row < candidates_count; cache_row += lanes_k)
        cache_norms[cache_row] = planes.norm_for(gathered.ids[cache_row]);
    hip_warp_t::sync();

    u32_t refined_count = 0;
    for (u32_t merge_index = 0; merge_index != sorted_candidates.size_ && refined_count != planes.connectivity;
         ++merge_index) {
        dense_vram_scored_node_t const entry = sorted_candidates.at(merge_index);
        f32_t const to_query = entry.distance();
        u32_t const candidate_position = entry.id();
        bool diverse = true;
        for (u32_t base = 0; base < refined_count && diverse; base += lanes_k) {
            bool lane_diverse = true;
            if (base + lane < refined_count) {
                u32_t const kept_position = kept_positions[base + lane];
                typename kit_t::dot_t pair_accumulator {};
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    pair_accumulator = kit_t::accumulate_dot(cache.at(candidate_position, fragment),
                                                             cache.at(kept_position, fragment), pair_accumulator);
                f32_t const between = kit_t::distance(planes.kernel, pair_accumulator, cache_norms[candidate_position],
                                                      cache_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++telemetry.scored;
            }
            diverse = hip_warp_t::all(lane_diverse);
        }
        if (diverse) {
            if (lane == 0) kept_positions[refined_count] = candidate_position;
            ++refined_count;
            hip_warp_t::sync();
        }
    }

    u32_t const diverse_count = refined_count;
    for (u32_t merge_index = 0, matched = 0;
         merge_index != sorted_candidates.size_ && refined_count != planes.connectivity; ++merge_index) {
        u32_t const candidate_position = sorted_candidates.at(merge_index).id();
        if (matched != diverse_count && kept_positions[matched] == candidate_position) {
            ++matched;
            continue;
        }
        if (lane == 0) kept_positions[refined_count] = candidate_position;
        ++refined_count;
        hip_warp_t::sync();
    }

    hip_warp_t::sync();
    u32_t* const live_slots = planes.neighbors_for(node_id);
    mask_t changed_mask = 0;
    for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += lanes_k) {
        u32_t const slot = slot_base + lane;
        u32_t const value = slot < refined_count ? gathered.ids[kept_positions[slot]]
                            : slot == 0          ? node_id
                                                 : dense_node_id_unknown_k;
        changed_mask |= hip_warp_t::ballot(slot < planes.connectivity && live_slots[slot] != value);
        if (slot < planes.connectivity) live_slots[slot] = value;
    }
    telemetry.changed_nodes += changed_mask != 0;
    hip_warp_t::sync();
}

/** One wavefront merges one touched node's forward and reverse candidates into its final
 *  neighbors. */
template <typename kit_>
__device__ __forceinline__ void dense_amd_merge_links(dense_vram_refine_view<kit_> const& view, u32_t const* bucket_ids,
                                                      u32_t const* bucket_counts, u32_t bucket_width,
                                                      dense_hip_bitset_t touched) {
    using kit_t = kit_;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_t::norms_t cache_norms[dense_vram_warps_per_block_k][dense_vram_cache_rows_k];
    __shared__ dense_vram_gathered<kit_t> gathered[dense_vram_warps_per_block_k];
    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    dense_vram_vector_cache<typename kit_::fragment_t> const cache {
        reinterpret_cast<typename kit_::fragment_t*>(slice.scratch), view.planes.row_fragments};
    // Past the candidate cache: the ascending scored span, then the kept-row indices.
    std::size_t const cache_bytes = static_cast<std::size_t>(view.planes.row_fragments) *
                                    sizeof(typename kit_::fragment_t) * dense_vram_cache_rows_k;
    dense_vram_scored_node_t* const sorted_span = reinterpret_cast<dense_vram_scored_node_t*>(slice.scratch +
                                                                                              cache_bytes);
    u32_t* const kept_positions = reinterpret_cast<u32_t*>(slice.scratch + cache_bytes +
                                                           sizeof(dense_vram_scored_node_t) * dense_vram_cache_rows_k);
    while (true) {
        u32_t node_id = 0;
        if (lane == 0) node_id = view.first_id + hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u);
        node_id = hip_warp_t::shuffle(node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the walk neither re-selected nor picked keeps its neighbors.
        if (!dense_amd_bitset_contains(touched, node_id)) continue;
        dense_vram_telemetry_t telemetry;
        dense_amd_merge_links_one(view, bucket_ids, bucket_counts, bucket_width, node_id, cache, sorted_span,
                                  kept_positions, cache_norms[warp], gathered[warp], telemetry, lane);
        telemetry.template flush<hip_warp_t, dense_vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** Scatters the walked rows' shadow neighbors in reverse, marking every row the merge
 *  re-selects. */
__device__ __forceinline__ void dense_amd_scatter_links(u32_t const* shadow_neighbors, u32_t rows_count, u32_t first_id,
                                                        u32_t walk_count, u32_t const* node_ids, u32_t connectivity,
                                                        u32_t bucket_width, u32_t* bucket_ids, u32_t* bucket_counts,
                                                        dense_hip_bitset_t touched) {
    dense_amd_wave_t const wave = dense_amd_wave();
    for (std::size_t offset = wave.index; offset < walk_count; offset += wave.count) {
        u32_t const row = node_ids ? node_ids[offset] : first_id + static_cast<u32_t>(offset);
        if (wave.lane == 0) dense_amd_bitset_insert(touched, row);
        if (node_ids) continue;
        dense_amd_for_each_link(shadow_neighbors, connectivity, rows_count, row, wave.lane, [&](u32_t target) noexcept {
            dense_amd_bitset_insert(touched, target);
            u32_t const landing = hip_warp_t::fetch_add_relaxed(bucket_counts[target], 1u);
            if (landing < bucket_width) bucket_ids[static_cast<std::size_t>(target) * bucket_width + landing] = row;
        });
    }
}

/** Rebuilds the norms plane for a run of rows, one wavefront per row. */
template <typename kit_>
__device__ __forceinline__ void dense_amd_derive_norms(dense_vram_planes_view<kit_> const& planes, u32_t first_row,
                                                       u32_t rows_count) {
    dense_amd_wave_t const wave = dense_amd_wave();
    for (std::size_t row = first_row + wave.index; row < first_row + rows_count; row += wave.count) {
        typename kit_::norms_accumulator_t accumulator {};
        for (u32_t fragment = wave.lane; fragment < planes.row_fragments; fragment += hip_warp_t::width_k)
            accumulator = kit_::accumulate_norms(planes.vector_for(static_cast<dense_node_id_t>(row), fragment),
                                                 accumulator);
        accumulator = dense_vram_reduce_norms<hip_warp_t>(accumulator);
        if (wave.lane == 0)
            planes.norm_ref_for(static_cast<dense_node_id_t>(row)) = kit_::partials_to_norms(accumulator);
    }
}

/** Marks the stride-spread landmarks reached and in the first frontier. */
__device__ __forceinline__ void dense_amd_mark_seeds(dense_hip_bitset_t reached, dense_hip_bitset_t frontier,
                                                     u32_t rows_count, u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    dense_amd_bitset_insert(reached, row);
    dense_amd_bitset_insert(frontier, row);
}

/** One level of forward reachability from the frontier. */
__device__ __forceinline__ void dense_amd_reachability_sweep(u32_t const* neighbors, u32_t rows_count,
                                                             u32_t connectivity, dense_hip_bitset_t reached,
                                                             dense_hip_bitset_t frontier,
                                                             dense_hip_bitset_t next_frontier, u32_t* expanded) {
    dense_amd_wave_t const wave = dense_amd_wave();
    for (std::size_t row = wave.index; row < rows_count; row += wave.count) {
        if (!dense_amd_bitset_contains(frontier, static_cast<u32_t>(row))) continue;
        if (wave.lane == 0) hip_warp_t::fetch_add_relaxed(*expanded, 1u);
        dense_amd_for_each_link(neighbors, connectivity, rows_count, static_cast<u32_t>(row), wave.lane,
                                [&](u32_t target) noexcept {
                                    if (dense_amd_bitset_insert(reached, target))
                                        dense_amd_bitset_insert(next_frontier, target);
                                });
    }
}

/** Counts and collects the rows the sweep never reached. */
__device__ __forceinline__ void dense_amd_reachability_islands(dense_hip_bitset_t reached, u32_t rows_count,
                                                               u32_t* island_ids, u32_t* islands_cursor) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count || dense_amd_bitset_contains(reached, row)) return;
    u32_t const slot = hip_warp_t::fetch_add_relaxed(*islands_cursor, 1u);
    if (island_ids) island_ids[slot] = row;
}

/** Counts, for every row, the reached rows pointing at it. */
__device__ __forceinline__ void dense_amd_reached_in_links(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                           dense_hip_bitset_t reached, u32_t* in_links) {
    dense_amd_wave_t const wave = dense_amd_wave();
    for (std::size_t row = wave.index; row < rows_count; row += wave.count) {
        if (!dense_amd_bitset_contains(reached, static_cast<u32_t>(row))) continue;
        dense_amd_for_each_link(neighbors, connectivity, rows_count, static_cast<u32_t>(row), wave.lane,
                                [&](u32_t target) noexcept { hip_warp_t::fetch_add_relaxed(in_links[target], 1u); });
    }
}

/** Reconnects each island into a reached pick or the landmark covering it. */
__device__ __forceinline__ void dense_amd_reconnect_island(u32_t* neighbors, u32_t connectivity,
                                                           u32_t const* island_ids, u32_t islands_count,
                                                           dense_hip_bitset_t reached, u32_t rows_count,
                                                           u32_t seed_count, u32_t* in_links, u32_t* reconnected) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const spread = dense_heal_spread(connectivity);
    auto const fetch_sub = [](u32_t& word, u32_t value) noexcept {
        return __hip_atomic_fetch_sub(&word, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    };

    // Whether the island now sits in the target's row, landing it there when a slot allows.
    auto const land = [&](u32_t target) noexcept {
        u32_t* const target_row = neighbors + static_cast<std::size_t>(target) * connectivity;
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
            if (resident != dense_node_id_unknown_k && resident != target) continue;
            if (claim(slot, resident)) return true;
        }
        for (u32_t attempt = 0; attempt != spread; ++attempt) {
            u32_t const landing = connectivity - 1 - (island_index + attempt) % spread;
            u32_t const resident = hip_warp_t::load_relaxed(target_row[landing]);
            if (resident == island) return true;
            if (resident != dense_node_id_unknown_k && !dense_amd_bitset_contains(reached, resident)) continue;
            bool const evicting = resident != dense_node_id_unknown_k && resident != target;
            if (evicting && fetch_sub(in_links[resident], 1u) <= 1u) {
                hip_warp_t::fetch_add_relaxed(in_links[resident], 1u);
                continue;
            }
            if (claim(landing, resident)) return true;
            if (evicting) hip_warp_t::fetch_add_relaxed(in_links[resident], 1u);
        }
        return false;
    };

    u32_t const* const island_row = neighbors + static_cast<std::size_t>(island) * connectivity;
    for (u32_t slot = 0; slot != connectivity; ++slot) {
        u32_t const pick = island_row[slot];
        if (pick != dense_node_id_unknown_k && pick != island && dense_amd_bitset_contains(reached, pick) && land(pick))
            return;
    }
    if (!seed_count) return;
    u32_t const seed = static_cast<u32_t>(static_cast<u64_t>(island) * seed_count / rows_count);
    u32_t const landmark = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    if (landmark != island) land(landmark);
}

#pragma endregion Bodies

#pragma region Kernels

/** Runs one walk family on its rung, the arena frontier at slot zero. */
template <typename kit_, u32_t slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void dense_amd_walk_rung(dense_vram_search_view<kit_> const& view, sink_ const& sink,
                                                    rescoring_ const& rescorings) {
    if constexpr (slots_ == 0) dense_amd_walk_wide<kit_>(view, sink, rescorings);
    else dense_amd_walk<kit_, slots_>(view, sink, rescorings);
}

/** Every AMD kernel: @p kernel_ picks the body, @p gpu_class_ and @p dtype_ the kit, @p slots_ the
 *  rung. */
template <u32_t kernel_, u32_t gpu_class_, u32_t dtype_, u32_t slots_>
__global__ void __launch_bounds__(dense_vram_warps_per_block_k* hip_warp_t::width_k)
    dense_amd_kernel(dense_amd_arguments_t arguments) {
    using kit_t = dense_amd_kit<gpu_class_, dtype_>;
    constexpr dense_amd_kernel_t kernel = static_cast<dense_amd_kernel_t>(kernel_);
    dense_amd_graph_t const& graph = arguments.graph;
    if constexpr (kernel == dense_amd_kernel_t::insert_k)
        dense_amd_insert<kit_t, slots_>(dense_amd_view<kit_t>(arguments.build));
    else if constexpr (kernel == dense_amd_kernel_t::refine_k)
        dense_amd_refine<kit_t, slots_>(dense_amd_view<kit_t>(arguments.refine));
    else if constexpr (kernel == dense_amd_kernel_t::search_k)
        dense_amd_walk_rung<kit_t, slots_>(dense_amd_view<kit_t>(arguments.search),
                                           dense_amd_view<kit_t>(arguments.search_sink), dense_vram_no_rescoring {});
    else if constexpr (kernel == dense_amd_kernel_t::search_indexer_k)
        dense_amd_walk_rung<kit_t, slots_>(dense_amd_view<kit_t>(arguments.search),
                                           dense_amd_view<kit_t>(arguments.search_sink),
                                           dense_amd_view<kit_t>(arguments.rescoring));
    else if constexpr (kernel == dense_amd_kernel_t::multiply_k)
        dense_amd_walk_rung<kit_t, slots_>(dense_amd_view<kit_t>(arguments.search),
                                           dense_amd_view<kit_t>(arguments.multiply_sink), dense_vram_no_rescoring {});
    else if constexpr (kernel == dense_amd_kernel_t::merge_links_k)
        dense_amd_merge_links<kit_t>(dense_amd_view<kit_t>(arguments.refine), graph.bucket_ids, graph.bucket_counts,
                                     graph.bucket_width, graph.touched);
    else if constexpr (kernel == dense_amd_kernel_t::scatter_links_k)
        dense_amd_scatter_links(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count, graph.node_ids,
                                graph.connectivity, graph.bucket_width, graph.bucket_ids, graph.bucket_counts,
                                graph.touched);
    else if constexpr (kernel == dense_amd_kernel_t::derive_norms_k)
        dense_amd_derive_norms<kit_t>(dense_amd_view<kit_t>(graph.planes), graph.first_id, graph.rows_count);
    else if constexpr (kernel == dense_amd_kernel_t::mark_seeds_k)
        dense_amd_mark_seeds(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel == dense_amd_kernel_t::reachability_sweep_k)
        dense_amd_reachability_sweep(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                     graph.frontier, graph.next_frontier, graph.counter);
    else if constexpr (kernel == dense_amd_kernel_t::reachability_islands_k)
        dense_amd_reachability_islands(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel == dense_amd_kernel_t::reached_in_links_k)
        dense_amd_reached_in_links(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                   graph.in_links);
    else
        dense_amd_reconnect_island(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count,
                                   graph.reached, graph.rows_count, graph.seed_count, graph.in_links,
                                   graph.reconnected);
}

template <u32_t gpu_class_, u32_t dtype_, std::size_t... rungs_>
__device__ void dense_amd_instantiate_rungs(std::index_sequence<rungs_...>) {
    (static_cast<void>(&dense_amd_kernel<static_cast<u32_t>(dense_amd_rungs[rungs_].kernel), gpu_class_, dtype_,
                                         dense_amd_rungs[rungs_].slots>),
     ...);
}

/** Instantiates every kernel one dtype exports on one class; a class unit names it once per
 *  compiled dtype. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
__device__ void dense_amd_instantiate() {
    dense_amd_instantiate_rungs<static_cast<u32_t>(gpu_class_), static_cast<u32_t>(dtype_)>(
        std::make_index_sequence<sizeof(dense_amd_rungs) / sizeof(dense_amd_rungs[0])> {});
}

#pragma endregion Kernels

} // namespace usearch
} // namespace unum
