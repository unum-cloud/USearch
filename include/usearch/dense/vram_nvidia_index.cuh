/**
 *  @file include/usearch/dense/vram_nvidia_index.cuh
 *  @author Ash Vardanian
 *  @date July 18, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  The kernel side of the index: the kernels, their device helpers, and the out-of-class
 *  definitions of the @c launch_*_ members @c vram_nvidia_index.hpp declares. Only the NVIDIA
 *  variant units include it, so their explicit instantiations carry every kernel.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy`

#include <limits>      // `std::numeric_limits` - the running maximum's identity
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::is_same_v` - which rescoring a search walks under

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.hpp"
#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_index.cuh"
#include "usearch/dense/vram_nvidia_index.hpp"
#include "usearch/dense/vram_nvidia_metric.cuh"

namespace unum {
namespace usearch {

static_assert(sizeof(dense_vram_scored_node_t) == dense_cuda_scored_node_bytes_k);
static_assert(dense_cuda_widest_pool_k == dense_vram_sorted_pool<16, cuda_lanes_t>::capacity_k);
static_assert(dense_cuda_ranked_head_k == dense_vram_sorted_pool<2, cuda_lanes_t>::capacity_k);
static_assert(sizeof(dense_cuda_arena_shape_t) == sizeof(dense_vram_arena_shape_t));

/** The quads kit @c dense_quads_chain picks for @p vector_dtype_ under @p capability_, checked
 *  against the host-visible half the face sizes its launches by. */
template <nk_dtype_t vector_dtype_, nk_capability_t capability_>
struct dense_cuda_kit_of {
    using traits_t = dense_cuda_kit_traits<vector_dtype_, capability_>;
    using type = typename dense_quads_chain<typename traits_t::scalar_t, capability_>::kit_t;
    static_assert(std::is_same_v<typename traits_t::scalar_t, typename type::scalar_t> &&
                  std::is_same_v<typename traits_t::fragment_t, typename type::fragment_t> &&
                  std::is_same_v<typename traits_t::dot_t, typename type::dot_t> &&
                  std::is_same_v<typename traits_t::norms_t, typename type::norms_t>);
    static_assert(traits_t::picked_k == type::capability_k);
    static_assert(traits_t::gathered_bytes_k == sizeof(dense_vram_gathered<type>));
};

template <nk_dtype_t vector_dtype_, nk_capability_t capability_>
using dense_cuda_kit_t = typename dense_cuda_kit_of<vector_dtype_, capability_>::type;

/** The kernels' twin of a host-computed arena slice layout. */
constexpr dense_vram_arena_shape_t dense_cuda_arena_shape(dense_cuda_arena_shape_t shape) noexcept {
    return {shape.visited_slots, shape.visited_hash_shift, shape.tail_entries};
}

/** One bit per row over device-resident words, marked concurrently by a whole grid - a reachability
 *  audit's reached set and frontiers, a refinement pass's touched rows. Every access is relaxed:
 *  the launch boundary orders the marks before any later launch or host copy reads them. */
struct dense_cuda_bitset_t {
    u32_t* words = nullptr;

    __device__ __forceinline__ bool contains(u32_t row) const noexcept {
        return (cuda_atomic_ref<u32_t>(words[row >> 5]).load(cuda::memory_order_relaxed) >> (row & 31u)) & 1u;
    }

    /** Sets the bit of @p row, answering whether this call was the one that set it. */
    __device__ __forceinline__ bool insert(u32_t row) const noexcept {
        u32_t const bit = 1u << (row & 31u);
        return !(cuda_atomic_ref<u32_t>(words[row >> 5]).fetch_or(bit, cuda::memory_order_relaxed) & bit);
    }
};

/** Grouped-query heads as a device rescoring: the walk navigates by a token's first head
 *  and keeps its dots, the warp reads the batch once more for every further head, every
 *  head's dots land in the warp's scratch for the attend fold, and a node's distance is 1 −
 *  maxₕ qₕ · k, so every head's own region is explored. */
template <dense_vram_quads_kit kit_>
struct dense_cuda_heads_rescoring {
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;

    /** Every token's heads as staged rows, @b [tokens·heads,row_fragments], device-resident. */
    fragment_t const* heads;

    /** The walking warp's scratch, opening with the batch dots, @b [heads,32]; set per walk. */
    f32_t* dots;

    /** Heads per token. */
    u32_t heads_count;

    /** Fragments per staged head row. */
    u32_t row_fragments;

    template <typename frontier_>
    __device__ __forceinline__ dense_cuda_heads_rescoring
    for_query(u32_t query_index, dense_vram_warp_context<frontier_> const& context) const noexcept {
        dense_cuda_heads_rescoring narrowed = *this;
        narrowed.heads += static_cast<std::size_t>(query_index) * heads_count * row_fragments;
        narrowed.dots = reinterpret_cast<f32_t*>(context.scratch);
        return narrowed;
    }

    /** Scores the batch against every head past the first as @ref dense_vram_score_batch scores it
     *  against the query: a sub-warp of @c lanes_per_row lanes owns one row, and each key fragment
     *  it loads meets every head. */
    __device__ __forceinline__ dense_vram_scored_node_t rescore(dense_vram_planes_view<kit_> const& planes,
                                                                dense_vram_gathered<kit_>& gathered,
                                                                u32_t gathered_count, int lane,
                                                                dense_vram_scored_node_t scored) const noexcept {
        u32_t const lanes_log2 = 31 - __clz(planes.lanes_per_row);
        u32_t const lane_in_row = static_cast<u32_t>(lane) & (planes.lanes_per_row - 1);
        bool const live = static_cast<u32_t>(lane) < gathered_count;
        if (live) dots[lane] = static_cast<f32_t>(gathered.dots[lane]);
        for (u32_t row_base = 0; heads_count != 1 && row_base < gathered_count; row_base += 32u >> lanes_log2) {
            u32_t const row = row_base + (static_cast<u32_t>(lane) >> lanes_log2);
            for (u32_t step = 0; step != planes.steps_per_row; ++step) {
                u32_t const fragment_index = lane_in_row + (step << lanes_log2);
                bool const loads = row < gathered_count && fragment_index < planes.row_fragments;
                fragment_t const key = loads ? planes.vector_for(gathered.ids[row], fragment_index) : fragment_t {};
                for (u32_t head = 1; head != heads_count; ++head) {
                    fragment_t const query = loads ? heads[head * row_fragments + fragment_index] : fragment_t {};
                    dot_t const dot = dense_vram_reduce<cuda_lanes_t>(kit_::accumulate_dot(key, query, dot_t {}),
                                                                      planes.lanes_per_row);
                    f32_t& slot = dots[head * 32 + row];
                    if (lane_in_row == 0 && row < gathered_count) slot = (step ? slot : 0.f) + static_cast<f32_t>(dot);
                }
            }
        }
        __syncwarp();
        if (!live) return scored;
        f32_t best = dots[lane];
        for (u32_t head = 1; head != heads_count; ++head) best = fmaxf(best, dots[head * 32 + lane]);
        scored = dense_vram_scored_node_t::make(1 - best, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/** Default dynamic-shared opt-in threshold: launches asking for more must raise the kernel's
 *  attribute before the first launch. */
inline constexpr u32_t dense_cuda_static_scratch_limit_k = 48 * 1024;

/** Launches one traversal kernel with @p blocks of the standard warp complement on @p stream,
 *  raising the dynamic-shared ceiling first when the warp slices outgrow the static window. */
template <auto kernel_, typename view_type_, typename... extras_types_>
status_t dense_cuda_launch(view_type_ view, u32_t blocks, cudaStream_t stream, extras_types_... extras) noexcept {
    u32_t const scratch_bytes = dense_vram_warps_per_block_k * view.scratch_bytes_per_warp;
    if (scratch_bytes > dense_cuda_static_scratch_limit_k &&
        cudaFuncSetAttribute(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch_bytes) != cudaSuccess)
        return status_t::device_error_k;
    return cuda_launch(kernel_, blocks, dense_vram_warps_per_block_k * 32, scratch_bytes, stream, view, extras...);
}

/** Lowers a runtime pool width onto the closed 2/4/8/16 register ladder, invoking @p callback with
 *  the width as an integral constant. */
template <typename callback_type_>
void dense_cuda_with_pool_slots(u32_t pool_slots, callback_type_&& callback) noexcept {
    switch (pool_slots) {
    case 2: callback(std::integral_constant<u32_t, 2> {}); break;
    case 4: callback(std::integral_constant<u32_t, 4> {}); break;
    case 8: callback(std::integral_constant<u32_t, 8> {}); break;
    default: callback(std::integral_constant<u32_t, 16> {});
    }
}

/**
 *  @brief One warp inserts one node, and warps never synchronize beyond per-neighbor slot-0
 *      locks: no device watermark, no publication order, no compare-and-swap.
 *
 *  @section dense_vram_nvidia_index_concurrency Concurrency & Publication
 *
 *  IDs are batch-local: the host claims a dense range, and each warp assigns itself the next offset
 *  within it, so sibling batches never cross-claim. To an inserting walk a node is usable exactly
 *  when its slot 0 acquire-loads a non-sentinel; a self-loop there means published-empty, and every
 *  give-up path still publishes one, so an assigned ID never stalls anyone. Landmark sampling
 *  probes past unpublished hits, so out-of-order publication costs retries, never correctness. The
 *  admission ramp holds a node until @c published + max(32, published) covers its ID, bounding lock
 *  contention while the graph is small. Searches see a batch only once the host has drained it and
 *  turned the watermark past it, a whole batch at a time and in claim order, and follow no neighbor
 *  at or past the watermark they launched under.
 */
template <dense_vram_quads_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_vram_warps_per_block_k * 32, dense_vram_resident_blocks_per_multiprocessor_k)
    dense_cuda_insert_kernel(dense_vram_build_view<kit_> view) {
    // The dynamic slice hosts the query span while traversing, then the candidate cache while
    // selecting; the exact visited set lives in the warp's arena slice, not in shared memory.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_vram_warps_per_block_k][dense_vram_cache_rows_k];
    __shared__ dense_vram_gathered<kit_> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<cuda_lanes_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_vram_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_vram_selection_scratch_t const selection = dense_vram_selection_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_vram_cache_rows_k);
    while (true) {
        u32_t new_node_id;
        if (lane == 0)
            new_node_id = view.first_id +
                          cuda_atomic_ref<u32_t>(*view.assigned_rows).fetch_add(1u, cuda::memory_order_relaxed);
        new_node_id = __shfl_sync(0xFFFFFFFFu, new_node_id, 0);
        if (new_node_id >= view.end_id) return;

        while (true) { // Admission ramp - the only bootstrap throttle.
            u32_t const published = cuda_atomic_ref<u32_t>(*view.published_rows).load(cuda::memory_order_relaxed);
            u32_t const window = max_of(published, admission_ramp_floor_k);
            if (new_node_id < published + window) break;
            __nanosleep(2048);
        }

        dense_vram_warp_context<dense_vram_register_frontier<pool_slots_, cuda_lanes_t>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(new_node_id) * 33 + lane)};
        dense_vram_visited_set<cuda_lanes_t> visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                      view.epoch_base + (new_node_id - view.first_id) + 1};
        dense_vram_insert_one<kit_>( //
            view, new_node_id, visited, query_span, cache, selection, cache_norms[warp], gathered[warp], context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<cuda_lanes_t, dense_vram_traversal_t::inserting_k>(view.telemetry, lane);
    }
}

/** One warp refines one node, assigning itself node IDs from the pass counter; warps only read the
 *  live planes and write disjoint shadow neighbors. */
template <dense_vram_quads_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_vram_warps_per_block_k * 32, dense_vram_resident_blocks_per_multiprocessor_k)
    dense_cuda_refine_kernel(dense_vram_refine_view<kit_> view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_vram_warps_per_block_k][dense_vram_cache_rows_k];
    __shared__ dense_vram_gathered<kit_> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<cuda_lanes_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_vram_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_vram_selection_scratch_t const selection = dense_vram_selection_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_vram_cache_rows_k);
    while (true) {
        u32_t offset;
        if (lane == 0) offset = cuda_atomic_ref<u32_t>(*view.assigned_rows).fetch_add(1u, cuda::memory_order_relaxed);
        offset = __shfl_sync(0xFFFFFFFFu, offset, 0);
        // Under an ID array, end_id bounds it; the assigned offset indexes it.
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_ids ? view.node_ids[offset] : view.first_id + offset;

        dense_vram_warp_context<dense_vram_register_frontier<pool_slots_, cuda_lanes_t>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(node_id) * 33 + lane)};
        dense_vram_visited_set<cuda_lanes_t> visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                      view.epoch_base + offset + 1};
        dense_vram_refine_one<kit_>( //
            view, node_id, visited, query_span, cache, selection, cache_norms[warp], gathered[warp], context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<cuda_lanes_t, dense_vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** Hands @p callback every neighbor of @p row the calling lane owns - its lane-strided slots, past
 *  sentinels, self-loops and IDs at or past @p rows_count. */
template <typename callback_type_>
__device__ __forceinline__ void dense_cuda_for_each_link(u32_t const* neighbors, u32_t connectivity, u32_t rows_count,
                                                         u32_t row, int lane, callback_type_&& callback) noexcept {
    u32_t const* const slots = neighbors + static_cast<std::size_t>(row) * connectivity;
    for (u32_t slot = static_cast<u32_t>(lane); slot < connectivity; slot += 32)
        if (u32_t const target = slots[slot]; target != dense_node_id_unknown_k && target != row && target < rows_count)
            callback(target);
}

/** Scatters the walked rows' shadow neighbors in reverse: every forward pick @b u → @b v lands
 *  @b u in @b v's reverse bucket, and marks both rows in @p touched - the rows the merge
 *  re-selects. The walked rows are the range from @p first_id, or a heal round's islands listed in
 *  @p node_ids, which mark only themselves and land in no bucket; appends past @p bucket_width are
 *  counted, never written. */
static __global__ void dense_cuda_scatter_links_kernel(u32_t const* shadow_neighbors, u32_t rows_count, u32_t first_id,
                                                       u32_t walk_count, u32_t const* node_ids, u32_t connectivity,
                                                       u32_t bucket_width, u32_t* bucket_ids, u32_t* bucket_counts,
                                                       dense_cuda_bitset_t touched) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t offset = warp_index; offset < walk_count; offset += warps) {
        u32_t const row = node_ids ? node_ids[offset] : first_id + static_cast<u32_t>(offset);
        if (lane == 0) touched.insert(row);
        if (node_ids) continue;
        dense_cuda_for_each_link(shadow_neighbors, connectivity, rows_count, row, lane, [&](u32_t target) noexcept {
            touched.insert(target);
            u32_t const landing =
                cuda_atomic_ref<u32_t>(bucket_counts[target]).fetch_add(1u, cuda::memory_order_relaxed);
            if (landing < bucket_width) bucket_ids[static_cast<std::size_t>(target) * bucket_width + landing] = row;
        });
    }
}

/** Selects one node's final neighbors from its forward picks and its reverse bucket, the diversity
 *  heuristic over the union ascending by distance to the node. The neighbors land in the live plane
 *  with plain stores, since the pass owns the graph exclusively. */
template <dense_vram_quads_kit kit_>
__device__ __forceinline__ void dense_cuda_merge_links_one( //
    dense_vram_refine_view<kit_> const& view, u32_t const* bucket_ids, u32_t const* bucket_counts, u32_t bucket_width,
    dense_node_id_t node_id, dense_vram_vector_cache<typename kit_::fragment_t> cache,
    dense_vram_scored_node_t* sorted_span, u32_t* kept_positions,
    typename kit_::norms_t (&cache_norms)[dense_vram_cache_rows_k], dense_vram_gathered<kit_>& gathered,
    dense_vram_telemetry_t& telemetry, int lane) noexcept {

    dense_vram_planes_view<kit_> const& planes = view.planes;
    // The last cache row hosts the node's own vector, so only the rows before it can host
    // candidates.
    constexpr u32_t query_row_k = dense_vram_cache_rows_k - 1;
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
        cache.at(query_row_k, fragment) = planes.vector_for(node_id, fragment);
    typename kit_::norms_t const query_state = planes.norm_for(node_id);
    __syncwarp();

    // Stage the forward picks; the shadow neighbors are this pass's own output, already distinct.
    u32_t candidates_count = 0;
    u32_t const* const shadow_slots = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) {
        u32_t const slot = slot_base + lane;
        u32_t const forward_id = slot < planes.connectivity ? shadow_slots[slot] : dense_node_id_unknown_k;
        bool const usable = forward_id != dense_node_id_unknown_k && forward_id != node_id;
        unsigned const usable_mask = __ballot_sync(0xFFFFFFFFu, usable);
        u32_t const ordinal = __popc(usable_mask & ((1u << lane) - 1));
        if (usable) gathered.ids[candidates_count + ordinal] = forward_id;
        candidates_count += __popc(usable_mask);
        __syncwarp();
    }
    u32_t const forward_count = candidates_count;

    // Stage the reverse bucket, dropping IDs the forward prefix already holds - mutual
    // picks land in both - and clamping the union to the candidate cache.
    u32_t const landed = min_of(bucket_counts[node_id], bucket_width);
    for (u32_t slot_base = 0; slot_base < landed && candidates_count < query_row_k; slot_base += 32) {
        u32_t const slot = slot_base + lane;
        u32_t const reverse_id = slot < landed ? bucket_ids[static_cast<std::size_t>(node_id) * bucket_width + slot]
                                               : dense_node_id_unknown_k;
        bool usable = reverse_id != dense_node_id_unknown_k && reverse_id != node_id;
        for (u32_t forward_index = 0; usable && forward_index != forward_count; ++forward_index)
            usable = gathered.ids[forward_index] != reverse_id;
        unsigned const usable_mask = __ballot_sync(0xFFFFFFFFu, usable);
        u32_t const ordinal = __popc(usable_mask & ((1u << lane) - 1));
        if (usable && candidates_count + ordinal < query_row_k) gathered.ids[candidates_count + ordinal] = reverse_id;
        candidates_count = min_of(candidates_count + __popc(usable_mask), query_row_k);
        __syncwarp();
    }

    // Score every candidate against the node & sort ascending; IDs are cache-row indices.
    telemetry.scored += candidates_count;
    dense_vram_sorted_buffer<cuda_lanes_t> sorted_candidates {sorted_span, 0, query_row_k};
    for (u32_t base = 0; base < candidates_count; base += 32) {
        u32_t const batch_count = min_of(candidates_count - base, 32u);
        dense_vram_scored_node_t const scored = dense_vram_score_batch<kit_>( //
            planes, cache.row(query_row_k), query_state, gathered.ids + base, batch_count, gathered, lane);
        dense_vram_scored_node_t const sorted_batch = dense_sort_lanes<cuda_lanes_t>(
            static_cast<u32_t>(lane) < batch_count
                ? dense_vram_scored_node_t::make(scored.distance(), base + lane) // ID = candidate index.
                : dense_vram_scored_node_t::sentinel(),
            lane);
        sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
    }

    // Cache the candidate vectors & norms for the pairwise diversity checks.
    u32_t const candidate_words = candidates_count * planes.row_fragments;
    for (u32_t word = lane; word < candidate_words; word += 32) {
        u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
        __pipeline_memcpy_async(&cache.at(cache_row, fragment),
                                planes.vector_span_for(gathered.ids[cache_row]) + fragment,
                                sizeof(typename kit_::fragment_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    for (u32_t cache_row = lane; cache_row < candidates_count; cache_row += 32)
        cache_norms[cache_row] = planes.norm_for(gathered.ids[cache_row]);
    __syncwarp();

    // The greedy diversity walk, ascending by distance to the node.
    u32_t refined_count = 0;
    for (u32_t merge_index = 0; merge_index != sorted_candidates.size_ && refined_count != planes.connectivity;
         ++merge_index) {
        dense_vram_scored_node_t const entry = sorted_candidates.at(merge_index);
        f32_t const to_query = entry.distance();
        u32_t const candidate_position = entry.id();
        bool diverse = true;
        for (u32_t base = 0; base < refined_count && diverse; base += 32) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < refined_count) {
                u32_t const kept_position = kept_positions[base + lane];
                typename kit_::dot_t pair_accumulator {};
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    pair_accumulator = kit_::accumulate_dot(cache.at(candidate_position, fragment),
                                                            cache.at(kept_position, fragment), pair_accumulator);
                f32_t const between = kit_::distance(planes.kernel, pair_accumulator, cache_norms[candidate_position],
                                                     cache_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++telemetry.scored;
            }
            diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
        }
        if (diverse) {
            if (lane == 0) kept_positions[refined_count] = candidate_position;
            ++refined_count;
            __syncwarp();
        }
    }

    // Fill the remaining slots with the nearest pruned candidates. The diverse prefix is a
    // subsequence of the sorted candidates, so one cursor tells a kept candidate from a rejected
    // one.
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
        __syncwarp();
    }

    // Land the final neighbors in the live plane - the insertion's slot convention. The pass owns
    // the graph, so a slot-wise comparison is the settling signal.
    __syncwarp();
    u32_t* const live_slots = planes.neighbors_for(node_id);
    unsigned changed_mask = 0;
    for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) { // Warp-uniform bounds.
        u32_t const slot = slot_base + lane;
        u32_t const value = slot < refined_count ? gathered.ids[kept_positions[slot]]
                            : slot == 0          ? node_id
                                                 : dense_node_id_unknown_k;
        changed_mask |= __ballot_sync(0xFFFFFFFFu, slot < planes.connectivity && live_slots[slot] != value);
        if (slot < planes.connectivity) live_slots[slot] = value;
    }
    telemetry.changed_nodes += changed_mask != 0;
    __syncwarp();
}

/** One warp merges one node's forward & reverse candidates into its final neighbors, assigning
 *  itself node IDs from the pass counter. */
template <dense_vram_quads_kit kit_>
__global__ void __launch_bounds__(dense_vram_warps_per_block_k * 32, dense_vram_resident_blocks_per_multiprocessor_k)
    dense_cuda_merge_links_kernel(dense_vram_refine_view<kit_> view, u32_t const* bucket_ids,
                                  u32_t const* bucket_counts, u32_t bucket_width, dense_cuda_bitset_t touched) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_vram_warps_per_block_k][dense_vram_cache_rows_k];
    __shared__ dense_vram_gathered<kit_> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<cuda_lanes_t>(view, dynamic_scratch);
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
        u32_t node_id;
        if (lane == 0)
            node_id = view.first_id +
                      cuda_atomic_ref<u32_t>(*view.assigned_rows).fetch_add(1u, cuda::memory_order_relaxed);
        node_id = __shfl_sync(0xFFFFFFFFu, node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the walk neither re-selected nor picked keeps its neighbors: re-judging it could
        // only drop links.
        if (!touched.contains(node_id)) continue;

        dense_vram_telemetry_t telemetry;
        dense_cuda_merge_links_one<kit_>( //
            view, bucket_ids, bucket_counts, bucket_width, node_id, cache, sorted_span, kept_positions,
            cache_norms[warp], gathered[warp], telemetry, lane);
        telemetry.template flush<cuda_lanes_t, dense_vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/**
 *  @brief Adds one batch's @p count value rows into @p heads_count accumulator rows in the
 *      warp's shared scratch: row h of @p accumulators is scaled by rescalesₕ, then takes Σᵣ
 *      weightsₕᵣ · rowᵣ, @p weights being @b [heads,32].
 *
 *  Each lane brings its own batch row's start in @p row and owns the accumulator fragments
 *  congruent to it modulo 32: it reads every row's fragment there once and adds it into every head,
 *  so the rows accumulate in place without atomics, and a row's last, partial fragment is read
 *  whole bytes at a time.
 */
template <dense_vram_quads_kit kit_>
__device__ __forceinline__ void dense_cuda_gather_weighted_rows( //
    typename kit_::fragment_t const* row, u32_t count, f32_t const* weights, f32_t const* rescales, u32_t heads_count,
    u32_t columns, f32_t* accumulators, int lane) noexcept {
    using fragment_t = typename kit_::fragment_t;
    constexpr u32_t columns_per_fragment_k = sizeof(fragment_t) * 8 / kit_::scalar_t::bits_per_dimension();
    u32_t const fragments = divide_round_up(columns, columns_per_fragment_k);
    for (u32_t fragment_base = 0; fragment_base < fragments; fragment_base += 32) {
        u32_t const fragment = fragment_base + lane;
        u32_t const fragment_columns = fragment < fragments
                                           ? min_of(columns - fragment * columns_per_fragment_k, columns_per_fragment_k)
                                           : 0;
        f32_t* const slots = accumulators + fragment * columns_per_fragment_k;
        for (u32_t head = 0; head != heads_count; ++head)
            if (rescales[head] != 1)
                for (u32_t column = 0; column != fragment_columns; ++column)
                    slots[head * columns + column] *= rescales[head];
        for (u32_t source_lane = 0; source_lane != count; ++source_lane) {
            fragment_t const* const source = reinterpret_cast<fragment_t const*>(
                __shfl_sync(0xFFFFFFFFu, reinterpret_cast<std::uintptr_t>(row), source_lane));
            if (!fragment_columns) continue;
            fragment_t value {};
            if (fragment_columns == columns_per_fragment_k) value = source[fragment];
            else
                std::memcpy(&value, source + fragment,
                            divide_round_up(fragment_columns * kit_::scalar_t::bits_per_dimension(), 8u));
            for (u32_t head = 0; head != heads_count; ++head) {
                f32_t const weight = weights[head * 32 + source_lane];
                if (weight == 0) continue;
                f32_t* const slot = slots + head * columns;
                // The sums round-trip through the shared row, fully unrolled so they never land in
                // local memory.
                typename kit_::weighted_t sums;
#pragma unroll
                for (u32_t column = 0; column != columns_per_fragment_k; ++column)
                    sums[column] = column < fragment_columns ? slot[column] : 0;
                sums = kit_::accumulate_weighted(value, weight, sums);
#pragma unroll
                for (u32_t column = 0; column != columns_per_fragment_k; ++column)
                    if (column < fragment_columns) slot[column] = sums[column];
            }
        }
    }
}

/** Resolves @p node_id to its value row through @p value_lookup, called on the node ID itself. */
template <typename value_lookup_type_>
__device__ __forceinline__ auto dense_cuda_value_row(value_lookup_type_ const& value_lookup,
                                                     dense_node_id_t node_id) noexcept {
    return value_lookup(node_id);
}

/** Resolves @p node_id through a shard's device key column, then the caller's key-level lookup. */
template <typename key_type_, typename value_lookup_type_>
__device__ __forceinline__ auto dense_cuda_value_row(
    dense_cuda_keyed_lookup<key_type_, value_lookup_type_> const& lookup, dense_node_id_t node_id) noexcept {
    return lookup.value_lookup(lookup.keys[node_id]);
}

/**
 *  @brief One token's attention as a walk fold, FlashAttention-style over every node the walk
 *      scores.
 *
 *  The per-head dots @ref dense_cuda_heads_rescoring left in the warp's scratch become base-2
 *  weights against each head's running maximum, and every value row the lookup resolves is read
 *  once and added into every head's accumulator row. The running maxima, partitions and rescales
 *  are warp-uniform, as every lane derives them from the same reductions.
 */
template <dense_vram_quads_kit kit_, typename value_lookup_type_>
struct dense_cuda_attend_fold {
    using fragment_t = typename kit_::fragment_t;

    /** Resolves a node ID to its value row's start, 16-byte aligned, or to null when it has
     *  none. */
    value_lookup_type_ value_lookup;

    /** The warp's shared scratch, carved in order: the batch dots @b [heads,32], each head's
     *  running base-2 maximum, partition and latest rescale, then the accumulator rows
     *  @b [heads,columns]. */
    f32_t* dots;
    f32_t* maxima;
    f32_t* partitions;
    f32_t* rescales;
    f32_t* accumulators;

    /** The factor turning a dot into a base-2 logit, @c score_scale · log₂e. */
    f32_t logit_scale;

    /** Heads per token, and columns per value and output row. */
    u32_t heads_count, columns;

    /** Whether the walk met a node the lookup had no row for, identical on every lane. */
    bool missed = false;

    __device__ __forceinline__ walk_demand_t absorb(dense_vram_gathered<kit_> const& gathered, u32_t gathered_count,
                                                    int lane) noexcept {
        constexpr unsigned all_lanes_k = 0xFFFFFFFFu;
        bool const live = static_cast<u32_t>(lane) < gathered_count;
        fragment_t const* const row =
            live ? static_cast<fragment_t const*>(dense_cuda_value_row(value_lookup, gathered.ids[lane])) : nullptr;
        // A node without its value row would leave a wrong softmax, so the walk stops there and the
        // call refuses.
        missed = __any_sync(all_lanes_k, live && !row);
        if (missed) return walk_demand_t::sated_k;
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const logit = live ? logit_scale * dots[head * 32 + lane] : std::numeric_limits<f32_t>::lowest();
            f32_t batch_maximum = logit;
            for (int offset = 16; offset; offset >>= 1)
                batch_maximum = fmaxf(batch_maximum, __shfl_xor_sync(all_lanes_k, batch_maximum, offset));
            f32_t const previous_maximum = maxima[head];
            f32_t const maximum = fmaxf(previous_maximum, batch_maximum);
            f32_t const weight = live ? exp2f(logit - maximum) : 0;
            f32_t batch_partition = weight;
            for (int offset = 16; offset; offset >>= 1)
                batch_partition += __shfl_xor_sync(all_lanes_k, batch_partition, offset);
            f32_t const rescale = exp2f(previous_maximum - maximum);
            if (live) dots[head * 32 + lane] = weight;
            // Every lane reads the previous state before the leader replaces it.
            __syncwarp();
            if (lane == 0) {
                maxima[head] = maximum;
                partitions[head] = partitions[head] * rescale + batch_partition;
                rescales[head] = rescale;
            }
        }
        __syncwarp();
        dense_cuda_gather_weighted_rows<kit_>(row, gathered_count, dots, rescales, heads_count, columns, accumulators,
                                              lane);
        return walk_demand_t::continue_k;
    }
};

/** Writes each token's attention out of its walk: opens the fold over the warp's scratch, then
 *  writes every head's normalized row and its carry once the walk ends, and raises the miss word if
 *  the walk met a row-less node. */
template <dense_vram_quads_kit kit_, typename value_lookup_type_>
struct dense_cuda_attend_sink {
    using fold_t = dense_cuda_attend_fold<kit_, value_lookup_type_>;

    /** Resolves a node ID to its value row, device-callable. */
    value_lookup_type_ value_lookup;

    /** The attended rows, @b [tokens·heads,columns], device-resident. */
    f32_t* outputs;

    /** Each row's softmax carry, @b [tokens·heads], device-resident. */
    dense_softmax_state_t* states;

    /** Set by any walk that met a node the lookup had no row for, zeroed and device-resident. */
    u32_t* missed;

    /** The factor turning a dot into a base-2 logit, @c score_scale · log₂e. */
    f32_t logit_scale;

    /** Heads per token, and columns per value and output row. */
    u32_t heads_count, columns;

    template <typename frontier_>
    __device__ __forceinline__ fold_t open(u32_t, dense_vram_warp_context<frontier_> const& context,
                                           int lane) const noexcept {
        f32_t* const dots = reinterpret_cast<f32_t*>(context.scratch);
        f32_t* const maxima = dots + heads_count * 32;
        f32_t* const partitions = maxima + heads_count;
        f32_t* const rescales = partitions + heads_count;
        f32_t* const accumulators = rescales + heads_count;
        fold_t fold {value_lookup, dots, maxima, partitions, rescales, accumulators, logit_scale, heads_count, columns};
        for (u32_t head = lane; head < heads_count; head += 32) {
            maxima[head] = std::numeric_limits<f32_t>::lowest();
            partitions[head] = 0;
        }
        // The first batch rescales the accumulators by zero, which a NaN left in shared memory
        // would survive.
        for (u32_t slot = lane; slot < heads_count * columns; slot += 32) accumulators[slot] = 0;
        __syncwarp();
        return fold;
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, dense_vram_warp_context<frontier_>&, u32_t query_index,
                                          int lane) const noexcept {
        if (fold.missed && lane == 0) cuda_atomic_ref<u32_t>(*missed).store(1u, cuda::memory_order_relaxed);
        // The lanes accumulated by fragment and write out by column, so the rows cross lanes here.
        __syncwarp();
        std::size_t const first = static_cast<std::size_t>(query_index) * heads_count;
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const partition = fold.partitions[head];
            f32_t const normalizer = partition > 0 ? 1 / partition : 0;
            f32_t const* const accumulated = fold.accumulators + head * columns;
            f32_t* const output = outputs + (first + head) * columns;
            for (u32_t column = lane; column < columns; column += 32) output[column] = accumulated[column] * normalizer;
            // Carries keep the natural-log maximum, so they merge with NumKong's log-sum-exp.
            if (lane == 0) states[first + head] = {fold.maxima[head] / std::numbers::log2e_v<f32_t>, partition};
        }
    }
};

/** One warp per query over the register frontier - expansions within the ladder - each walk scored
 *  under @p rescorings and written out through @p sink. */
template <dense_vram_quads_kit kit_, u32_t pool_slots_, typename sink_, typename rescoring_>
__global__ void __launch_bounds__(dense_vram_warps_per_block_k * 32, pool_slots_ <= 4 ? 4 : 3)
    dense_cuda_walk_kernel(dense_vram_search_view<kit_> view, sink_ sink, rescoring_ rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ dense_vram_gathered<kit_> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<cuda_lanes_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_vram_warps_per_block_k) {
        dense_vram_visited_set<cuda_lanes_t> visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                      view.epoch_base + query_index + 1};
        dense_vram_walk_one<kit_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, sink, rescorings, query_index, query_span, visited,
            dense_vram_register_frontier<pool_slots_, cuda_lanes_t> {}, gathered[warp], lane);
    }
}

/** One warp per query past the register ladder's width: the same walk over the arena frontier,
 *  whose budget is the requested expansion itself; ranks cover the frontier's head of 64, so the
 *  index admits this path for searches only within it. */
template <dense_vram_quads_kit kit_, typename sink_, typename rescoring_>
__global__ void __launch_bounds__(dense_vram_warps_per_block_k * 32, 4)
    dense_cuda_walk_wide_kernel(dense_vram_search_view<kit_> view, sink_ sink, rescoring_ rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ dense_vram_gathered<kit_> gathered[dense_vram_warps_per_block_k];

    dense_vram_warp_slice_t const slice = dense_vram_carve_slice<cuda_lanes_t>(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_vram_arena_frontier<cuda_lanes_t> frontier;
    frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_vram_warps_per_block_k) {
        dense_vram_visited_set<cuda_lanes_t> visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                      view.epoch_base + query_index + 1};
        dense_vram_walk_one<kit_, 4u>(view, sink, rescorings, query_index, query_span, visited, frontier,
                                      gathered[warp], lane);
    }
}

/** Rebuilds the norms plane from the vector plane for rows @c [first_row,first_row+rows_count), one
 *  warp per row; norms are derived state, never serialized. */
template <dense_vram_quads_kit kit_>
__global__ void dense_cuda_derive_norms_kernel(dense_vram_planes_view<kit_> planes, u32_t first_row, u32_t rows_count) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = first_row + warp_index; row < first_row + rows_count; row += warps) {
        typename kit_::norms_accumulator_t accumulator {};
        for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
            accumulator = kit_::accumulate_norms(planes.vector_for(static_cast<dense_node_id_t>(row), fragment),
                                                 accumulator);
        accumulator = dense_vram_reduce_norms<cuda_lanes_t>(accumulator);
        if (lane == 0) planes.norm_ref_for(static_cast<dense_node_id_t>(row)) = kit_::partials_to_norms(accumulator);
    }
}

/** Post-drain structural audit of the neighbor plane: every row published, every ID in range,
 *  self-loop rows childless, no duplicate neighbors within a row. One warp scans one row, a
 *  warp-width of slots at a time; each bad row bumps @p violations once. */
static __global__ void dense_cuda_invariant_kernel(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                   u64_t* violations) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        u32_t const first = neighbors[row * connectivity];
        bool bad = first == dense_node_id_unknown_k;
        for (u32_t base = 0; base < connectivity; base += 32) { // Warp-uniform trip count.
            u32_t const slot = base + lane;
            u32_t const value = slot < connectivity ? neighbors[row * connectivity + slot] : dense_node_id_unknown_k;
            bad |= value != dense_node_id_unknown_k && value >= rows_count;
            bad |= first == static_cast<u32_t>(row) && slot > 0 && value != dense_node_id_unknown_k;
            // Duplicate-neighbor scan: rotate every not-earlier batch past this lane's value.
            for (u32_t other_base = base; other_base < connectivity; other_base += 32) {
                u32_t const other_own = other_base + lane;
                u32_t const other_value = other_base == base         ? value
                                          : other_own < connectivity ? neighbors[row * connectivity + other_own]
                                                                     : dense_node_id_unknown_k;
                for (int shift = 0; shift < 32; ++shift) {
                    int const source = (lane + shift) & 31;
                    u32_t const other = __shfl_sync(0xFFFFFFFFu, other_value, source);
                    u32_t const other_slot = other_base + static_cast<u32_t>(source);
                    bad |= value != dense_node_id_unknown_k && other == value && other_slot > slot &&
                           other_slot < connectivity;
                }
            }
        }
        if (__ballot_sync(0xFFFFFFFFu, bad) && lane == 0)
            cuda_atomic_ref<u64_t>(*violations).fetch_add(1u, cuda::memory_order_relaxed);
    }
}

/** Marks @p seed_count stride-spread rows reached and in the first frontier - the audit's landmark
 *  set, spread over the ID space like the traversal samplers. */
static __global__ void dense_cuda_mark_seeds_kernel(dense_cuda_bitset_t reached, dense_cuda_bitset_t frontier,
                                                    u32_t rows_count, u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    reached.insert(row);
    frontier.insert(row);
}

/** One level of forward reachability: every row of @p frontier marks its targets reached, and the
 *  ones it reaches first join @p next_frontier. Only the frontier expands, so a level is the same
 *  set of rows however the warps interleave, and sweeping until a level expands nothing takes the
 *  landmarks' eccentricity in levels. */
static __global__ void dense_cuda_reachability_sweep_kernel(u32_t const* neighbors, u32_t rows_count,
                                                            u32_t connectivity, dense_cuda_bitset_t reached,
                                                            dense_cuda_bitset_t frontier,
                                                            dense_cuda_bitset_t next_frontier, u32_t* expanded) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        if (!frontier.contains(static_cast<u32_t>(row))) continue;
        if (lane == 0) cuda_atomic_ref<u32_t>(*expanded).fetch_add(1u, cuda::memory_order_relaxed);
        dense_cuda_for_each_link(neighbors, connectivity, rows_count, static_cast<u32_t>(row), lane,
                                 [&](u32_t target) noexcept {
                                     if (reached.insert(target)) next_frontier.insert(target);
                                 });
    }
}

/** Counts the rows @p reached lacks - the island population - in @p islands_cursor, collecting
 *  their IDs into @p island_ids when it is given. */
static __global__ void dense_cuda_reachability_islands_kernel(dense_cuda_bitset_t reached, u32_t rows_count,
                                                              u32_t* island_ids, u32_t* islands_cursor) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count || reached.contains(row)) return;
    u32_t const slot = cuda_atomic_ref<u32_t>(*islands_cursor).fetch_add(1u, cuda::memory_order_relaxed);
    if (island_ids) island_ids[slot] = row;
}

/** Counts, for every row, the reached rows that point at it - the links a reconnection may take
 *  from a reached resident only while another one remains. */
static __global__ void dense_cuda_reached_in_links_kernel(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                          dense_cuda_bitset_t reached, u32_t* in_links) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        if (!reached.contains(static_cast<u32_t>(row))) continue;
        dense_cuda_for_each_link(neighbors, connectivity, rows_count, static_cast<u32_t>(row), lane,
                                 [&](u32_t target) noexcept {
                                     cuda_atomic_ref<u32_t>(in_links[target]).fetch_add(1u, cuda::memory_order_relaxed);
                                 });
    }
}

/** Reconnects each island into the first of its reached forward picks with room for it, or into the
 *  landmark covering its ID, bypassing the diversity judgment that dropped it: a free slot when one
 *  is left, otherwise a landing spread across the tail. Every landing is a compare-and-swap that
 *  never displaces an unreached sibling, so islands sharing one target all land, nor a reached
 *  resident whose @p in_links would drop to zero, so no reconnection strands another node;
 *  @p reconnected counts the links actually written. Exclusive pass: the launch boundary publishes
 *  every write, so each access is relaxed. */
static __global__ void dense_cuda_reconnect_island_kernel(u32_t* neighbors, u32_t connectivity, u32_t const* island_ids,
                                                          u32_t islands_count, dense_cuda_bitset_t reached,
                                                          u32_t rows_count, u32_t seed_count, u32_t* in_links,
                                                          u32_t* reconnected) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const spread = dense_heal_spread(connectivity);
    constexpr cuda::memory_order relaxed_k = cuda::memory_order_relaxed;

    // Whether the island now sits in the target's row, landing it there when a slot allows.
    auto const land = [&](u32_t target) noexcept {
        u32_t* const target_row = neighbors + static_cast<std::size_t>(target) * connectivity;
        auto const claim = [&](u32_t slot, u32_t resident) noexcept {
            if (!cuda_atomic_ref<u32_t>(target_row[slot]).compare_exchange_strong(resident, island, relaxed_k))
                return false;
            cuda_atomic_ref<u32_t>(*reconnected).fetch_add(1u, relaxed_k);
            return true;
        };
        for (u32_t slot = 0; slot != connectivity; ++slot) {
            u32_t const resident = cuda_atomic_ref<u32_t>(target_row[slot]).load(relaxed_k);
            if (resident == island) return true; // Already wired; the audit raced a heal.
            if (resident != dense_node_id_unknown_k && resident != target) continue;
            if (claim(slot, resident)) return true;
        }
        for (u32_t attempt = 0; attempt != spread; ++attempt) {
            u32_t const landing = connectivity - 1 - (island_index + attempt) % spread;
            u32_t const resident = cuda_atomic_ref<u32_t>(target_row[landing]).load(relaxed_k);
            if (resident == island) return true;
            if (resident != dense_node_id_unknown_k && !reached.contains(resident))
                continue; // A sibling island landed first.
            // A reached resident gives this link up only while another reached row still points at
            // it.
            bool const evicting = resident != dense_node_id_unknown_k && resident != target;
            if (evicting && cuda_atomic_ref<u32_t>(in_links[resident]).fetch_sub(1u, relaxed_k) <= 1u) {
                cuda_atomic_ref<u32_t>(in_links[resident]).fetch_add(1u, relaxed_k);
                continue;
            }
            if (claim(landing, resident)) return true;
            if (evicting) cuda_atomic_ref<u32_t>(in_links[resident]).fetch_add(1u, relaxed_k);
        }
        return false;
    };

    u32_t const* const island_row = neighbors + static_cast<std::size_t>(island) * connectivity;
    for (u32_t slot = 0; slot != connectivity; ++slot) {
        u32_t const pick = island_row[slot];
        if (pick != dense_node_id_unknown_k && pick != island && reached.contains(pick) && land(pick)) return;
    }
    // A walk starts at the landmark covering this ID by definition, which is what makes it reached.
    if (!seed_count) return;
    u32_t const seed = static_cast<u32_t>(static_cast<u64_t>(island) * seed_count / rows_count);
    u32_t const landmark = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    if (landmark != island) land(landmark);
}

/** Merges the normalized attended rows and carries of @p shards walks into the first's by the flash
 *  rule, a warp per row: each rescales to the joint maximum and weighs by its share of the joint
 *  partition. Every shard holds a block of @b [rows,columns] rows, then @b [rows] carries, the
 *  blocks lying back to back from @p blocks. */
static __global__ void dense_cuda_merge_attention_kernel(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards) {
    u32_t const row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    std::size_t const outputs_bytes = std::size_t {rows} * columns * sizeof(f32_t);
    std::size_t const block_bytes = outputs_bytes + rows * sizeof(dense_softmax_state_t);
    auto const carry_of = [&](u32_t shard) noexcept -> dense_softmax_state_t& {
        auto* const carries = reinterpret_cast<dense_softmax_state_t*>(blocks + shard * block_bytes + outputs_bytes);
        return carries[row];
    };
    auto const row_of = [&](u32_t shard) noexcept {
        return reinterpret_cast<f32_t*>(blocks + shard * block_bytes) + std::size_t {row} * columns;
    };
    f32_t maximum = carry_of(0).maximum_score, partition = 0;
    for (u32_t shard = 1; shard != shards; ++shard) maximum = fmaxf(maximum, carry_of(shard).maximum_score);
    // The carries keep natural-log maxima, so their gaps turn base-2 for the exponent.
    auto const weight_of = [&](u32_t shard) noexcept {
        dense_softmax_state_t const carry = carry_of(shard);
        return carry.partition * exp2f((carry.maximum_score - maximum) * std::numbers::log2e_v<f32_t>);
    };
    for (u32_t shard = 0; shard != shards; ++shard) partition += weight_of(shard);
    f32_t* const merged = row_of(0);
    for (u32_t shard = 0; shard != shards; ++shard) {
        f32_t const share = partition > 0 ? weight_of(shard) / partition : 0;
        f32_t const* const partial = row_of(shard);
        for (u32_t column = lane; column < columns; column += 32)
            merged[column] = (shard ? merged[column] : 0) + share * partial[column];
    }
    // Every lane reads the first carry above before the leader replaces it.
    __syncwarp();
    if (lane == 0) carry_of(0) = {maximum, partition};
}

#pragma region Launches

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
auto dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                             traversals_allocator_type_>::planes() const noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    static_assert(max_node_count_k == dense_vram_scored_node_t::id_capacity_k);
    u32_t const fragments = row_fragments();
    return dense_vram_planes_view<kit_t> {reinterpret_cast<fragment_t*>(vectors_plane_.base()),
                                          reinterpret_cast<norms_t*>(norms_plane_.base()),
                                          reinterpret_cast<u32_t*>(neighbors_plane_.base()),
                                          static_cast<u32_t>(connectivity()),
                                          fragments,
                                          lanes_per_row(),
                                          steps_per_row(),
                                          static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments),
                                          kernel_};
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_insert_( //
    fragment_t const* staging, dense_node_id_t first_id, u32_t end_id, byte_t* arena,
    dense_cuda_arena_shape_t arena_shape, u32_t epoch_base, u32_t pool_slots,
    dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    dense_vram_build_view<kit_t> const view {planes(),
                                             staging,
                                             end_id,
                                             insert_scratch_bytes_per_warp(),
                                             first_id,
                                             arena,
                                             dense_cuda_arena_shape(arena_shape),
                                             epoch_base,
                                             published_rows_.get(),
                                             context.telemetry_.get(),
                                             context.assigned_rows_.get()};
    status_t launched = status_t::success_k;
    dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
        launched = dense_cuda_launch<dense_cuda_insert_kernel<kit_t, slots.value>>( //
            view, device_.multiprocessor_count * insert_blocks_per_multiprocessor_k, context.stream());
    });
    return launched;
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_refine_( //
    refine_launch_t const& launch, dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    dense_vram_refine_view<kit_t> const view {planes(), //
                                              launch.shadow_neighbors,
                                              static_cast<u32_t>(launch.nodes.bound),
                                              insert_scratch_bytes_per_warp(),
                                              launch.nodes.first,
                                              launch.arena,
                                              dense_cuda_arena_shape(launch.arena_shape),
                                              launch.epoch_base,
                                              context.telemetry_.get(),
                                              context.assigned_rows_.get(),
                                              launch.node_ids,
                                              launch.nodes_count};
    status_t launched = status_t::success_k;
    dense_cuda_with_pool_slots(launch.pool_slots, [&](auto slots) {
        launched = dense_cuda_launch<dense_cuda_refine_kernel<kit_t, slots.value>>( //
            view, device_.multiprocessor_count * insert_blocks_per_multiprocessor_k, context.stream());
    });
    return launched;
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_merge_( //
    refine_launch_t const& launch, dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    cudaStream_t const stream = context.stream();
    u32_t const blocks = device_.multiprocessor_count * insert_blocks_per_multiprocessor_k;
    dense_cuda_bitset_t const touched {launch.touched};
    if (status_t const launched = cuda_launch(
            dense_cuda_scatter_links_kernel, blocks, block_threads_k, 0, stream,
            static_cast<u32_t const*>(launch.shadow_neighbors), launch.nodes_count,
            static_cast<u32_t>(launch.nodes.first), static_cast<u32_t>(launch.nodes.size()), launch.node_ids,
            static_cast<u32_t>(connectivity()), launch.bucket_width, launch.buckets, launch.bucket_counts, touched);
        failed(launched))
        return launched;
    // The merge sweeps the whole ID space and skips the rows the scatter left untouched.
    dense_vram_refine_view<kit_t> const view {planes(), //
                                              launch.shadow_neighbors,
                                              launch.nodes_count,
                                              merge_scratch_bytes_per_warp(),
                                              0,
                                              launch.arena,
                                              dense_cuda_arena_shape(launch.arena_shape),
                                              launch.epoch_base,
                                              context.telemetry_.get(),
                                              context.assigned_rows_.get(),
                                              nullptr,
                                              launch.nodes_count};
    return dense_cuda_launch<dense_cuda_merge_links_kernel<kit_t>>( //
        view, blocks, stream, static_cast<u32_t const*>(launch.buckets),
        static_cast<u32_t const*>(launch.bucket_counts), launch.bucket_width, touched);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
template <typename sink_type_, typename rescoring_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_walk_( //
    walk_launch_t const& walk, sink_type_ const& sink, rescoring_type_ const& rescoring,
    dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    dense_vram_search_view<kit_t> const view {planes(),
                                              walk.queries,
                                              walk.queries_count,
                                              walk.visible,
                                              walk.scratch_bytes_per_warp,
                                              walk.expansion,
                                              walk.arena,
                                              dense_cuda_arena_shape(walk.arena_shape),
                                              walk.epoch_base,
                                              context.telemetry_.get()};
    u32_t const blocks = device_.multiprocessor_count * search_blocks_per_multiprocessor_k;
    cudaStream_t const stream = context.stream();
    if (walk.frontier == frontier_kind_t::wide_arena_k)
        return dense_cuda_launch<dense_cuda_walk_wide_kernel<kit_t, sink_type_, rescoring_type_>>(view, blocks, stream,
                                                                                                  sink, rescoring);
    status_t launched = status_t::success_k;
    dense_cuda_with_pool_slots(walk.pool_slots, [&](auto slots) {
        launched = dense_cuda_launch<dense_cuda_walk_kernel<kit_t, slots.value, sink_type_, rescoring_type_>>(
            view, blocks, stream, sink, rescoring);
    });
    return launched;
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_search_( //
    walk_launch_t const& walk, dense_node_id_t* ids, dense_distance_t* distances, u32_t count,
    dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    dense_vram_search_sink<kit_t> const sink {ids, distances, count, kernel_};
    // Only the floating dtypes take indexer heads, so only they carry the rescored kernels.
    if constexpr (dense_floating_scalar<scalar_t>)
        if (walk.weights)
            return launch_walk_(
                walk, sink,
                dense_vram_indexer_rescoring<kit_t> {walk.heads, walk.weights, walk.heads_count, row_fragments()},
                context);
    return launch_walk_(walk, sink, dense_vram_no_rescoring {}, context);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_multiply_( //
    walk_launch_t const& walk, dense_node_id_t* ids, dot_t* products, u32_t* counts, u32_t wanted,
    dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    return launch_walk_(walk, dense_vram_multiply_sink<kit_t> {ids, products, counts, wanted},
                        dense_vram_no_rescoring {}, context);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
template <typename value_lookup_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_attend_( //
    walk_launch_t const& walk, value_lookup_type_ const& value_lookup, f32_t* outputs, dense_softmax_state_t* states,
    f32_t logit_scale, dense_traversal_context_t& context) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    dense_cuda_attend_sink<kit_t, value_lookup_type_> const sink {value_lookup,
                                                                  outputs,
                                                                  states,
                                                                  context.missed_.get(),
                                                                  logit_scale,
                                                                  walk.heads_count,
                                                                  static_cast<u32_t>(dimensions())};
    return launch_walk_(walk, sink,
                        dense_cuda_heads_rescoring<kit_t> {walk.heads, nullptr, walk.heads_count, row_fragments()},
                        context);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_derive_norms_( //
    std::size_t first_row, std::size_t rows, cudaStream_t stream) noexcept {
    using kit_t = dense_cuda_kit_t<vector_dtype_, capability_>;
    u32_t const blocks = static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k;
    return cuda_launch(dense_cuda_derive_norms_kernel<kit_t>, blocks, block_threads_k, 0, stream, planes(),
                       static_cast<u32_t>(first_row), static_cast<u32_t>(rows));
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_audit_seeds_( //
    u32_t* reached, u32_t* frontier, std::size_t rows, std::size_t seed_count, cudaStream_t stream) noexcept {
    return cuda_launch(dense_cuda_mark_seeds_kernel,
                       divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads_k), block_threads_k, 0,
                       stream, dense_cuda_bitset_t {reached}, dense_cuda_bitset_t {frontier}, static_cast<u32_t>(rows),
                       static_cast<u32_t>(seed_count));
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_audit_sweep_( //
    u32_t* reached, u32_t* frontier, u32_t* next_frontier, u32_t* expanded, std::size_t rows,
    cudaStream_t stream) noexcept {
    u32_t const grid = static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k;
    return cuda_launch(dense_cuda_reachability_sweep_kernel, grid, block_threads_k, 0, stream,
                       reinterpret_cast<u32_t const*>(neighbors_plane_.base()), static_cast<u32_t>(rows),
                       static_cast<u32_t>(connectivity()), dense_cuda_bitset_t {reached},
                       dense_cuda_bitset_t {frontier}, dense_cuda_bitset_t {next_frontier}, expanded);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_audit_islands_( //
    u32_t* reached, std::size_t rows, u32_t* island_ids, u32_t* islands_cursor, cudaStream_t stream) noexcept {
    return cuda_launch(dense_cuda_reachability_islands_kernel,
                       divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads_k), block_threads_k, 0, stream,
                       dense_cuda_bitset_t {reached}, static_cast<u32_t>(rows), island_ids, islands_cursor);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_reconnect_islands_( //
    audit_t const& audit, u32_t islands, std::size_t rows, cudaStream_t stream) noexcept {
    u32_t* const neighbors = reinterpret_cast<u32_t*>(neighbors_plane_.base());
    dense_cuda_bitset_t const reached {audit.reached};
    if (status_t const counted = cuda_launch(
            dense_cuda_reached_in_links_kernel,
            static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k, block_threads_k, 0,
            stream, static_cast<u32_t const*>(neighbors), static_cast<u32_t>(rows), static_cast<u32_t>(connectivity()),
            reached, audit.in_links);
        failed(counted))
        return counted;
    return cuda_launch(dense_cuda_reconnect_island_kernel, divide_round_up<u32_t>(islands, block_threads_k),
                       block_threads_k, 0, stream, neighbors, static_cast<u32_t>(connectivity()),
                       static_cast<u32_t const*>(audit.island_ids), islands, reached, static_cast<u32_t>(rows),
                       static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows)), audit.in_links,
                       audit.reconnected);
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename vectors_allocator_type_,
          typename neighbors_allocator_type_, typename traversals_allocator_type_>
status_t dense_vram_nvidia_index<vector_dtype_, capability_, vectors_allocator_type_, neighbors_allocator_type_,
                                 traversals_allocator_type_>::launch_merge_attention_( //
    byte_t* blocks, u32_t rows, u32_t columns, u32_t shards, cudaStream_t stream) noexcept {
    return cuda_launch(dense_cuda_merge_attention_kernel, divide_round_up<u32_t>(rows, dense_vram_warps_per_block_k),
                       dense_vram_warps_per_block_k * 32, 0, stream, blocks, rows, columns, shards);
}

#pragma endregion Launches

} // namespace usearch
} // namespace unum
