/**
 *  @file include/usearch/dense/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD index's kernels: one template over the kernel, the class, the dtype, the rung and
 *      the attend value lookup.
 *
 *  Included only by the class units, which HIP-Clang compiles device-only, once per class over
 *  every compiled dtype. Each @c amd_kernel instantiation turns the plain records of
 *  @c vram_amd_index.hpp into the shared views of @c vram_index.cuh and runs the shared bodies
 *  through @ref hip_warp_t, and the host finds it by the mangled name @c amd_kernel_name
 *  spells.
 */
#pragma once

#include <cstddef>     // `std::size_t`
#include <limits>      // `std::numeric_limits` - the running maximum's identity
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::conditional_t` - a builder's frontier by its rung
#include <utility>     // `std::index_sequence` - every rung of one dtype

#include <numkong/types.h>       // `NUMKONG_ARCH_ROCM_`, which the two below test before including it
#include <numkong/cast/simt.cuh> // `nk_cast_unit_simt_` - the import casts' device body
#include <numkong/each/rocm.cuh> // `nk_each_rmscast_e4m3_rocm_vectors_` & kin - the unit-RMS import's device bodies

#include "usearch/utilities/vram_queue.cuh"
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
            refine.part_bases,
            refine.parts,
            refine.round};
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

/** Copies @p bytes from @p source into @p target one at a time, since `std::memcpy` names the host
 *  function under HIP. */
__device__ __forceinline__ void amd_copy_bytes(void* target, void const* source, std::size_t bytes) noexcept {
    for (std::size_t offset = 0; offset != bytes; ++offset)
        static_cast<byte_t*>(target)[offset] = static_cast<byte_t const*>(source)[offset];
}

/** Grouped-query heads as a device rescoring: the walk navigates by a token's first head and keeps
 *  its dots, the wavefront reads the batch once more for every further head, every head's dots
 *  land in the wavefront's scratch for the attend fold, and a node's distance is 1 − maxₕ qₕ · k,
 *  so every head's own region is explored. */
template <vram_quads_kit kit_>
struct amd_heads_rescoring {
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;

    /** Every token's heads as staged rows, @b [tokens·heads,row_fragments], device-resident. */
    fragment_t const* heads;

    /** The walking wavefront's scratch, opening with the batch dots, @b [heads,lanes]; set per
     *  walk. */
    f32_t* dots;

    /** Heads per token. */
    u32_t heads_count;

    /** Fragments per staged head row. */
    u32_t row_fragments;

    template <typename frontier_>
    __device__ __forceinline__ amd_heads_rescoring
    for_query(u32_t query_index, vram_warp_context<frontier_> const& context) const noexcept {
        amd_heads_rescoring narrowed = *this;
        narrowed.heads = vram_row(heads, query_index, heads_count * row_fragments);
        narrowed.dots = reinterpret_cast<f32_t*>(context.scratch);
        return narrowed;
    }

    /** Scores the batch against every head past the first as @ref vram_score_batch scores it
     *  against the query: a sub-wave of @c lanes_per_row lanes owns one row, and each key fragment
     *  it loads meets every head. */
    __device__ __forceinline__ vram_scored_node_t rescore(vram_flat_view<kit_> const& storage,
                                                          vram_gathered<kit_>& gathered, u32_t gathered_count,
                                                          u32_t lane, vram_scored_node_t scored) const noexcept {
        constexpr u32_t lanes_k = hip_warp_t::width_k;
        u32_t const lanes_log2 = 31 - __clz(storage.lanes_per_row);
        u32_t const lane_in_row = lane & (storage.lanes_per_row - 1);
        bool const live = lane < gathered_count;
        if (live) dots[lane] = static_cast<f32_t>(gathered.dots[lane]);
        for (u32_t row_base = 0; heads_count != 1 && row_base < gathered_count; row_base += lanes_k >> lanes_log2) {
            u32_t const row = row_base + (lane >> lanes_log2);
            for (u32_t step = 0; step != storage.steps_per_row; ++step) {
                u32_t const fragment_index = lane_in_row + (step << lanes_log2);
                bool const loads = row < gathered_count && fragment_index < storage.row_fragments;
                fragment_t const key = loads ? storage.vector_for(gathered.ids[row], fragment_index) : fragment_t {};
                for (u32_t head = 1; head != heads_count; ++head) {
                    fragment_t const query = loads ? heads[head * row_fragments + fragment_index] : fragment_t {};
                    dot_t const dot = vram_reduce<hip_warp_t>(kit_::accumulate_dot(key, query, dot_t {}),
                                                              storage.lanes_per_row);
                    f32_t& slot = dots[head * lanes_k + row];
                    if (lane_in_row == 0 && row < gathered_count) slot = (step ? slot : 0.f) + static_cast<f32_t>(dot);
                }
            }
        }
        hip_warp_t::sync();
        if (!live) return scored;
        f32_t best = dots[lane];
        for (u32_t head = 1; head != heads_count; ++head) best = fmaxf(best, dots[head * lanes_k + lane]);
        scored = vram_scored_node_t::make(1 - best, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/** One wavefront inserts one node, as the NVIDIA insertion kernel does. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t pool_slots_>
__device__ __forceinline__ void amd_insert(vram_build_view<kit_> const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_vector_cache<typename kit_::fragment_t> const cache {
        query_span, view.storage.row_fragments, view.arena_shape.cache_bytes ? gathered[warp].ids : nullptr};
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
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t pool_slots_>
__device__ __forceinline__ void amd_refine(vram_refine_view<kit_> const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_vector_cache<typename kit_::fragment_t> const cache {
        query_span, view.storage.row_fragments, view.arena_shape.cache_bytes ? gathered[warp].ids : nullptr};
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

/**
 *  @brief Adds one batch's @p count value rows into @p heads_count accumulator rows in the
 *      wavefront's scratch: row h of @p accumulators is scaled by rescalesₕ, then takes Σᵣ
 *      weightsₕᵣ · rowᵣ, @p weights being @b [heads,lanes].
 *
 *  Each lane brings its own batch row's start in @p row and owns the accumulator fragments
 *  congruent to it modulo the wavefront's width: it reads every row's fragment there once and adds
 *  it into every head, so the rows accumulate in place without atomics, and a row's last, partial
 *  fragment is read whole bytes at a time.
 */
template <vram_quads_kit kit_>
__device__ __forceinline__ void amd_gather_weighted_rows( //
    typename kit_::fragment_t const* row, u32_t count, f32_t const* weights, f32_t const* rescales, u32_t heads_count,
    u32_t columns, f32_t* accumulators, u32_t lane) noexcept {
    using fragment_t = typename kit_::fragment_t;
    constexpr u32_t lanes_k = hip_warp_t::width_k;
    constexpr u32_t columns_per_fragment_k = sizeof(fragment_t) * 8 / kit_::scalar_t::bits_per_dimension();
    u32_t const fragments = divide_round_up(columns, columns_per_fragment_k);
    for (u32_t fragment_base = 0; fragment_base < fragments; fragment_base += lanes_k) {
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
                hip_warp_t::shuffle(reinterpret_cast<std::uintptr_t>(row), source_lane));
            if (!fragment_columns) continue;
            fragment_t value {};
            if (fragment_columns == columns_per_fragment_k) value = source[fragment];
            else
                amd_copy_bytes(&value, source + fragment,
                               divide_round_up(fragment_columns * kit_::scalar_t::bits_per_dimension(), 8u));
            for (u32_t head = 0; head != heads_count; ++head) {
                f32_t const weight = weights[head * lanes_k + source_lane];
                if (weight == 0) continue;
                f32_t* const slot = slots + head * columns;
                // The sums round-trip through the shared row, fully unrolled so they never spill to
                // private memory.
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
__device__ __forceinline__ auto amd_value_row(value_lookup_type_ const& value_lookup, node_id_t node_id) noexcept {
    return value_lookup(node_id);
}

/** Resolves @p node_id through a shard's device key column, then the caller's key-level lookup. */
template <typename key_type_, typename value_lookup_type_>
__device__ __forceinline__ auto amd_value_row(amd_keyed_lookup<key_type_, value_lookup_type_> const& lookup,
                                              node_id_t node_id) noexcept {
    return lookup.value_lookup(lookup.keys[node_id]);
}

/**
 *  @brief One token's attention as a walk fold, FlashAttention-style over every node the walk
 *      scores.
 *
 *  The per-head dots @ref amd_heads_rescoring left in the wavefront's scratch become base-2 weights
 *  against each head's running maximum, and every value row the lookup resolves is read once and
 *  added into every head's accumulator row. The running maxima, partitions and rescales are
 *  wave-uniform, as every lane derives them from the same reductions.
 */
template <vram_quads_kit kit_, typename value_lookup_type_>
struct amd_attend_fold {
    using fragment_t = typename kit_::fragment_t;

    /** Resolves a node ID to its value row's start, 16-byte aligned, or to null when it has
     *  none. */
    value_lookup_type_ value_lookup;

    /** The wavefront's scratch, carved in order: the batch dots @b [heads,lanes], each head's
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

    __device__ __forceinline__ walk_demand_t absorb(vram_gathered<kit_> const& gathered, u32_t gathered_count,
                                                    u32_t lane) noexcept {
        constexpr u32_t lanes_k = hip_warp_t::width_k;
        bool const live = lane < gathered_count;
        fragment_t const* const row =
            live ? static_cast<fragment_t const*>(amd_value_row(value_lookup, gathered.ids[lane])) : nullptr;
        // A node without its value row would leave a wrong softmax, so the walk stops there and the
        // call refuses.
        missed = hip_warp_t::any(live && !row);
        if (missed) return walk_demand_t::sated_k;
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const logit = live ? logit_scale * dots[head * lanes_k + lane] : std::numeric_limits<f32_t>::lowest();
            f32_t batch_maximum = logit;
            for (u32_t offset = lanes_k / 2; offset; offset >>= 1)
                batch_maximum = fmaxf(batch_maximum, hip_warp_t::shuffle_xor(batch_maximum, offset));
            f32_t const previous_maximum = maxima[head];
            f32_t const maximum = fmaxf(previous_maximum, batch_maximum);
            f32_t const weight = live ? exp2f(logit - maximum) : 0;
            f32_t batch_partition = weight;
            for (u32_t offset = lanes_k / 2; offset; offset >>= 1)
                batch_partition += hip_warp_t::shuffle_xor(batch_partition, offset);
            f32_t const rescale = exp2f(previous_maximum - maximum);
            if (live) dots[head * lanes_k + lane] = weight;
            // Every lane reads the previous state before the leader replaces it.
            hip_warp_t::sync();
            if (lane == 0) {
                maxima[head] = maximum;
                partitions[head] = partitions[head] * rescale + batch_partition;
                rescales[head] = rescale;
            }
        }
        hip_warp_t::sync();
        amd_gather_weighted_rows<kit_>(row, gathered_count, dots, rescales, heads_count, columns, accumulators, lane);
        return walk_demand_t::continue_k;
    }
};

/** Writes each token's attention out of its walk: opens the fold over the wavefront's scratch, then
 *  writes every head's normalized row and its carry once the walk ends, and raises the miss word if
 *  the walk met a row-less node. */
template <vram_quads_kit kit_, typename value_lookup_type_>
struct amd_attend_sink {
    using fold_t = amd_attend_fold<kit_, value_lookup_type_>;

    /** Resolves a node ID to its value row, device-callable. */
    value_lookup_type_ value_lookup;

    /** The attended rows, @b [tokens·heads,columns], device-resident. */
    f32_t* outputs;

    /** Each row's softmax carry, @b [tokens·heads], device-resident. */
    softmax_state_t* states;

    /** Set by any walk that met a node the lookup had no row for, zeroed and device-resident. */
    u32_t* missed;

    /** The factor turning a dot into a base-2 logit, @c score_scale · log₂e. */
    f32_t logit_scale;

    /** Heads per token, and columns per value and output row. */
    u32_t heads_count, columns;

    template <typename frontier_>
    __device__ __forceinline__ fold_t open(u32_t, vram_warp_context<frontier_> const& context,
                                           u32_t lane) const noexcept {
        constexpr u32_t lanes_k = hip_warp_t::width_k;
        f32_t* const dots = reinterpret_cast<f32_t*>(context.scratch);
        f32_t* const maxima = dots + heads_count * lanes_k;
        f32_t* const partitions = maxima + heads_count;
        f32_t* const rescales = partitions + heads_count;
        f32_t* const accumulators = rescales + heads_count;
        fold_t fold {value_lookup, dots, maxima, partitions, rescales, accumulators, logit_scale, heads_count, columns};
        for (u32_t head = lane; head < heads_count; head += lanes_k) {
            maxima[head] = std::numeric_limits<f32_t>::lowest();
            partitions[head] = 0;
        }
        // The first batch rescales the accumulators by zero, which a NaN left in the LDS would
        // survive.
        for (u32_t slot = lane; slot < heads_count * columns; slot += lanes_k) accumulators[slot] = 0;
        hip_warp_t::sync();
        return fold;
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, vram_warp_context<frontier_>&, u32_t query_index,
                                          u32_t lane) const noexcept {
        constexpr u32_t lanes_k = hip_warp_t::width_k;
        if (fold.missed && lane == 0) hip_warp_t::store_relaxed(*missed, 1u);
        // The lanes accumulated by fragment and write out by column, so the rows cross lanes here.
        hip_warp_t::sync();
        f32_t* const token_outputs = vram_row(outputs, query_index, heads_count * columns);
        softmax_state_t* const token_states = vram_row(states, query_index, heads_count);
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const partition = fold.partitions[head];
            f32_t const normalizer = partition > 0 ? 1 / partition : 0;
            f32_t const* const accumulated = fold.accumulators + head * columns;
            f32_t* const output = token_outputs + head * columns;
            for (u32_t column = lane; column < columns; column += lanes_k)
                output[column] = accumulated[column] * normalizer;
            // Carries keep the natural-log maximum, so they merge with NumKong's log-sum-exp.
            if (lane == 0) token_states[head] = {fold.maxima[head] / std::numbers::log2e_v<f32_t>, partition};
        }
    }
};

/** One wavefront per query over the register frontier, each walk scored through the schedule under
 *  @p rescorings and written out through @p sink; wavefronts take queries from the batch counter, as
 *  builders take rows. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t pool_slots_, typename sink_, typename rescoring_>
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
        vram_walk_one<kit_, schedule_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, sink, rescorings, query_index, query_span, visited,
            vram_register_frontier<pool_slots_, hip_warp_t> {}, gathered[slice.warp], slice.lane);
    }
}

/** One wavefront per query past the register ladder, over the arena frontier. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, typename sink_, typename rescoring_>
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
        vram_walk_one<kit_, schedule_, 4u>(view, sink, rescorings, query_index, query_span, visited, frontier,
                                           gathered[slice.warp], slice.lane);
    }
}

/** One wavefront merges one touched node's forward and reverse candidates into its final
 *  neighbors. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_>
__device__ __forceinline__ void amd_merge_back_links(vram_refine_view<kit_> const& view, u32_t const* bucket_ids,
                                                     u32_t const* bucket_counts, u32_t bucket_width,
                                                     hip_bitset_t touched) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<hip_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_vector_cache<typename kit_::fragment_t> const cache {
        reinterpret_cast<typename kit_::fragment_t*>(slice.scratch), view.storage.row_fragments,
        view.arena_shape.cache_bytes ? gathered[warp].ids : nullptr};
    schedule_ schedule {schedule_states[warp], lane};
    while (true) {
        u32_t node_id = lane == 0 ? hip_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        node_id = view.first_id + hip_warp_t::shuffle(node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the walk neither re-selected nor picked keeps its neighbors: re-judging it could
        // only drop links.
        if (!amd_bitset_contains(touched, node_id)) continue;

        vram_telemetry_t telemetry;
        vram_merge_back_links_one<kit_>( //
            view, bucket_ids, bucket_counts, bucket_width, node_id, cache, schedule, cache_norms[warp], gathered[warp],
            telemetry, lane);
        telemetry.template flush<hip_warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** Scatters the walked rows' shadow neighbors in reverse, marking every row the merge
 *  re-selects. */
__device__ __forceinline__ void amd_scatter_back_links(u32_t const* shadow_neighbors, u32_t rows_count, u32_t first_id,
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
 *  was queued past convergence and has nothing to expand. The first expansion of a level counts it
 *  in @p levels, so the host reads the depth from one word. */
__device__ __forceinline__ void amd_reachability_sweep(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                       hip_bitset_t reached, hip_bitset_t frontier,
                                                       hip_bitset_t next_frontier, u32_t* expanded,
                                                       u32_t const* expanded_before, u32_t* levels) {
    // The level before landed ahead of this launch, so every thread reads one settled word.
    if (expanded_before && !*expanded_before) return;
    vram_grid_warp_t const warp = vram_grid_warp<hip_warp_t>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!amd_bitset_contains(frontier, row)) continue;
        if (warp.lane == 0 && !hip_warp_t::fetch_add_relaxed(*expanded, 1u)) hip_warp_t::fetch_add_relaxed(*levels, 1u);
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
            u32_t const landing = connectivity - 1 - (island + attempt) % spread;
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

/** Merges the normalized attended rows and carries of @p shards walks into the first's by the flash
 *  rule, a wavefront per row: each rescales to the joint maximum and weighs by its share of the
 *  joint partition. Every shard holds a block of @b [rows,columns] rows, then @b [rows] carries,
 *  the blocks lying back to back from @p blocks. */
__device__ __forceinline__ void amd_merge_attention(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards) {
    constexpr u32_t lanes_k = hip_warp_t::width_k;
    u32_t const row = blockIdx.x * (blockDim.x / lanes_k) + threadIdx.x / lanes_k, lane = threadIdx.x % lanes_k;
    if (row >= rows) return;
    std::size_t const outputs_bytes = std::size_t {rows} * columns * sizeof(f32_t);
    std::size_t const block_bytes = outputs_bytes + rows * sizeof(softmax_state_t);
    auto const carry_of = [&](u32_t shard) noexcept -> softmax_state_t& {
        auto* const carries = reinterpret_cast<softmax_state_t*>(blocks + shard * block_bytes + outputs_bytes);
        return carries[row];
    };
    auto const row_of = [&](u32_t shard) noexcept {
        return vram_row(reinterpret_cast<f32_t*>(blocks + shard * block_bytes), row, columns);
    };
    f32_t maximum = carry_of(0).maximum_score, partition = 0;
    for (u32_t shard = 1; shard != shards; ++shard) maximum = fmaxf(maximum, carry_of(shard).maximum_score);
    // The carries keep natural-log maxima, so their gaps turn base-2 for the exponent.
    auto const weight_of = [&](u32_t shard) noexcept {
        softmax_state_t const carry = carry_of(shard);
        return carry.partition * exp2f((carry.maximum_score - maximum) * std::numbers::log2e_v<f32_t>);
    };
    for (u32_t shard = 0; shard != shards; ++shard) partition += weight_of(shard);
    f32_t* const merged = row_of(0);
    for (u32_t shard = 0; shard != shards; ++shard) {
        f32_t const share = partition > 0 ? weight_of(shard) / partition : 0;
        f32_t const* const partial = row_of(shard);
        for (u32_t column = lane; column < columns; column += lanes_k)
            merged[column] = (shard ? merged[column] : 0) + share * partial[column];
    }
    // Every lane reads the first carry above before the leader replaces it.
    hip_warp_t::sync();
    if (lane == 0) carry_of(0) = {maximum, partition};
}

#pragma endregion Bodies

#pragma region Kernels

/** Runs one walk family on its rung, the arena frontier at slot zero. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void amd_walk_rung(vram_search_view<kit_> const& view, sink_ const& sink,
                                              rescoring_ const& rescorings) {
    if constexpr (slots_ == 0) amd_walk_wide<kit_, schedule_>(view, sink, rescorings);
    else amd_walk<kit_, schedule_, slots_>(view, sink, rescorings);
}

/** Every AMD kernel: @p kernel_ picks the body, @p gpu_class_ and @p dtype_ the kit, @p gpu_class_
 *  alone the schedule, @p slots_ the rung, and @p lookup_ an attend's value lookup through
 *  @ref amd_value_lookup_of. */
template <u32_t kernel_, u32_t gpu_class_, u32_t dtype_, u32_t slots_, u32_t lookup_>
__global__ void __launch_bounds__(vram_warps_per_block_k* hip_warp_t::width_k) amd_kernel(amd_arguments_t arguments) {
    using kit_t = amd_kit<gpu_class_, dtype_>;
    using schedule_t = rocm_schedule_of<amd_capabilities_of(static_cast<gpu_class_t>(gpu_class_))>;
    constexpr amd_kernel_t kernel = static_cast<amd_kernel_t>(kernel_);
    amd_graph_t const& graph = arguments.graph;
    if constexpr (kernel == amd_kernel_t::insert_k)
        amd_insert<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.build));
    else if constexpr (kernel == amd_kernel_t::refine_k)
        amd_refine<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.refine));
    else if constexpr (kernel == amd_kernel_t::search_k)
        amd_walk_rung<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.search),
                                                 amd_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    else if constexpr (kernel == amd_kernel_t::search_indexer_k)
        amd_walk_rung<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.search),
                                                 amd_view<kit_t>(arguments.search_sink),
                                                 amd_view<kit_t>(arguments.rescoring));
    else if constexpr (kernel == amd_kernel_t::multiply_k)
        amd_walk_rung<kit_t, schedule_t, slots_>(amd_view<kit_t>(arguments.search),
                                                 amd_view<kit_t>(arguments.multiply_sink), vram_no_rescoring {});
    else if constexpr (kernel == amd_kernel_t::attend_k) {
        using value_lookup_t = typename amd_value_lookup_of<lookup_>::type;
        amd_attend_t const& attend = arguments.attend;
        amd_rescoring_t const& heads = arguments.rescoring;
        value_lookup_t value_lookup;
        amd_copy_bytes(&value_lookup, attend.value_lookup, sizeof(value_lookup_t));
        amd_walk_rung<kit_t, schedule_t, slots_>(
            amd_view<kit_t>(arguments.search),
            amd_attend_sink<kit_t, value_lookup_t> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                    attend.logit_scale, attend.heads_count, attend.columns},
            amd_heads_rescoring<kit_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads), nullptr,
                                        heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel == amd_kernel_t::merge_back_links_k)
        amd_merge_back_links<kit_t, schedule_t>(amd_view<kit_t>(arguments.refine), graph.bucket_ids,
                                                graph.bucket_counts, graph.bucket_width, graph.touched);
    else if constexpr (kernel == amd_kernel_t::scatter_back_links_k)
        amd_scatter_back_links(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count, graph.node_ids,
                               graph.connectivity, graph.bucket_width, graph.bucket_ids, graph.bucket_counts,
                               graph.touched);
    else if constexpr (kernel == amd_kernel_t::derive_norms_k)
        amd_derive_norms<kit_t>(amd_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel == amd_kernel_t::rebase_links_k)
        vram_rebase_links<hip_warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                      graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel == amd_kernel_t::mark_seeds_k)
        amd_mark_seeds(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel == amd_kernel_t::reachability_sweep_k)
        amd_reachability_sweep(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached, graph.frontier,
                               graph.next_frontier, graph.counter, graph.previous_counter, graph.levels);
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
    else if constexpr (kernel == amd_kernel_t::merge_attention_k)
        amd_merge_attention(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
                            arguments.attend.columns, arguments.attend.shards);
    else
        amd_reconnect_island(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count, graph.reached,
                             graph.rows_count, graph.seed_count, graph.in_links, graph.reconnected);
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
