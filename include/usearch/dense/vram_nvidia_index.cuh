/**
 *  @file include/usearch/dense/vram_nvidia_index.cuh
 *  @author Ash Vardanian
 *  @date July 18, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  The kernel side of the index: the device bodies and the one kernel template over five integers -
 *  the kernel, the capability ceiling, the dtype, the register rung and the attend value lookup -
 *  whose instantiations a fatbin carries. Only NVCC's device-only passes include it, once per
 *  ceiling over every compiled dtype; each @c cuda_kernel turns the plain records of
 *  @c vram_nvidia_index.hpp into the shared views of @c vram_index.cuh, and the host finds it by
 *  the mangled name @c cuda_kernel_name spells.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy`

#include <limits>      // `std::numeric_limits` - the running maximum's identity
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::is_same_v`, `std::conditional_t` - a walk's rescoring, a builder's frontier
#include <utility>     // `std::index_sequence` - every rung of one dtype

#include <numkong/types.h>       // `NUMKONG_ARCH_CUDA_`, which the two below test before including it
#include <numkong/cast/simt.cuh> // `nk_cast_unit_simt_` - the import and export casts' device body
#include <numkong/each/cuda.cuh> // `nk_each_rmscast_e4m3_cuda_vectors_` & kin - the unit-RMS import's device bodies

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/vram_index.cuh"
#include "usearch/dense/vram_nvidia_index.hpp"
#include "usearch/dense/vram_nvidia_metric.cuh"

namespace unum::usearch::dense {

static_assert(sizeof(vram_scored_node_t) == cuda_scored_node_bytes_k);
static_assert(sizeof(cuda_arena_shape_t) == sizeof(vram_arena_shape_t));
// Every dispensable ID fits a scored node's ID field, the sentinel the claim word closes on aside.
static_assert(vram_nvidia_index<>::max_node_count_k <= vram_scored_node_t::id_capacity_k);

/** The quads kit @c quads_chain picks for @p vector_dtype_ under @p capability_, checked
 *  against the host-visible half the face sizes its launches by. */
template <nk_dtype_t vector_dtype_, nk_capability_t capability_>
struct cuda_kit_of {
    using traits_t = cuda_kit_traits<vector_dtype_, capability_>;
    using type = typename quads_chain<typename traits_t::scalar_t, capability_>::kit_t;
    static_assert(std::is_same_v<typename traits_t::scalar_t, typename type::scalar_t> &&
                  sizeof(typename traits_t::fragment_t) == sizeof(typename type::fragment_t) &&
                  std::is_same_v<typename traits_t::dot_t, typename type::dot_t> &&
                  std::is_same_v<typename traits_t::norms_t, typename type::norms_t>);
    static_assert(traits_t::picked_k == type::capability_k);
    static_assert(traits_t::gathered_bytes_k == sizeof(vram_gathered<type>));
    static_assert(traits_t::schedule_bytes_k == sizeof(typename cuda_schedule_of<capability_>::shared_t));
};

template <nk_dtype_t vector_dtype_, nk_capability_t capability_>
using cuda_kit_t = typename cuda_kit_of<vector_dtype_, capability_>::type;

/** Grouped-query heads as a device rescoring: the walk navigates by a token's first head
 *  and keeps its dots, the warp reads the batch once more for every further head, every
 *  head's dots land in the warp's scratch for the attend fold, and a node's distance is 1 −
 *  maxₕ qₕ · k, so every head's own region is explored. */
template <vram_quads_kit kit_>
struct cuda_heads_rescoring {
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
    __device__ __forceinline__ cuda_heads_rescoring
    for_query(u32_t query_index, vram_warp_context<frontier_> const& context) const noexcept {
        cuda_heads_rescoring narrowed = *this;
        narrowed.heads = vram_row(heads, query_index, heads_count * row_fragments);
        narrowed.dots = reinterpret_cast<f32_t*>(context.scratch);
        return narrowed;
    }

    /** Scores the batch against every head past the first as @ref vram_score_batch scores it
     *  against the query: a sub-warp of @c lanes_per_row lanes owns one row, and each key fragment
     *  it loads meets every head. */
    __device__ __forceinline__ vram_scored_node_t rescore(vram_flat_view<kit_> const& storage,
                                                          vram_gathered<kit_>& gathered, u32_t gathered_count,
                                                          u32_t lane, vram_scored_node_t scored) const noexcept {
        u32_t const lanes_log2 = 31 - __clz(storage.lanes_per_row);
        u32_t const lane_in_row = lane & (storage.lanes_per_row - 1);
        bool const live = lane < gathered_count;
        if (live) dots[lane] = static_cast<f32_t>(gathered.dots[lane]);
        for (u32_t row_base = 0; heads_count != 1 && row_base < gathered_count; row_base += 32u >> lanes_log2) {
            u32_t const row = row_base + (lane >> lanes_log2);
            for (u32_t step = 0; step != storage.steps_per_row; ++step) {
                u32_t const fragment_index = lane_in_row + (step << lanes_log2);
                bool const loads = row < gathered_count && fragment_index < storage.row_fragments;
                fragment_t const key = loads ? storage.vector_for(gathered.ids[row], fragment_index) : fragment_t {};
                for (u32_t head = 1; head != heads_count; ++head) {
                    fragment_t const query = loads ? heads[head * row_fragments + fragment_index] : fragment_t {};
                    dot_t const dot = vram_reduce<cuda_warp_t>(kit_::accumulate_dot(key, query, dot_t {}),
                                                               storage.lanes_per_row);
                    f32_t& slot = dots[head * 32 + row];
                    if (lane_in_row == 0 && row < gathered_count) slot = (step ? slot : 0.f) + static_cast<f32_t>(dot);
                }
            }
        }
        cuda_warp_t::sync();
        if (!live) return scored;
        f32_t best = dots[lane];
        for (u32_t head = 1; head != heads_count; ++head) best = fmaxf(best, dots[head * 32 + lane]);
        scored = vram_scored_node_t::make(1 - best, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

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
 *  host launches a batch in admission waves, each no wider than the rows already published nor
 *  narrower than 32, so no walk waits on its graph and lock contention stays bounded while the
 *  graph is small. Searches see a batch only once the host has drained it and turned the watermark
 *  past it, a whole batch at a time and in claim order, and follow no neighbor at or past the
 *  watermark they launched under.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t pool_slots_>
__device__ __forceinline__ void cuda_insert(vram_build_view<kit_> const& view) {
    // The scratch slice hosts the query span while traversing, then the candidate cache while
    // selecting; the exact visited set lives in the warp's arena slice, not in shared memory.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_vector_cache<typename kit_::fragment_t> const cache {
        query_span, view.storage.row_fragments, view.arena_shape.cache_bytes ? gathered[warp].ids : nullptr};
    schedule_ schedule {schedule_states[warp], lane};
    vram_selection_scratch_t const selection = vram_selection_scratch_t::carve(cache, view.storage.connectivity);
    using frontier_t = std::conditional_t<pool_slots_ == 0, vram_arena_frontier<cuda_warp_t>,
                                          vram_register_frontier<pool_slots_, cuda_warp_t>>;
    while (true) {
        u32_t new_node_id = lane == 0 ? cuda_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        new_node_id = view.first_id + cuda_warp_t::shuffle(new_node_id, 0);
        if (new_node_id >= view.end_id) return;

        vram_warp_context<frontier_t> context;
        if constexpr (pool_slots_ == 0) {
            context.frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
            context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
            context.frontier.expansion_target_ = view.expansion;
        }
        context.random_generator = vram_lane_generator<cuda_warp_t>(new_node_id, lane);
        vram_visited_set<cuda_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                               view.epoch_base + (new_node_id - view.first_id) + 1};
        vram_insert_one<kit_>( //
            view, new_node_id, visited, query_span, cache, schedule, selection, cache_norms[warp], gathered[warp],
            context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<cuda_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
    }
}

/** One warp refines one node, assigning itself node IDs from the pass counter; warps only read the
 *  live storage and write disjoint shadow neighbors. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t pool_slots_>
__device__ __forceinline__ void cuda_refine(vram_refine_view<kit_> const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_vector_cache<typename kit_::fragment_t> const cache {
        query_span, view.storage.row_fragments, view.arena_shape.cache_bytes ? gathered[warp].ids : nullptr};
    schedule_ schedule {schedule_states[warp], lane};
    vram_selection_scratch_t const selection = vram_selection_scratch_t::carve(cache, view.storage.connectivity);
    using frontier_t = std::conditional_t<pool_slots_ == 0, vram_arena_frontier<cuda_warp_t>,
                                          vram_register_frontier<pool_slots_, cuda_warp_t>>;
    while (true) {
        u32_t offset = lane == 0 ? cuda_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        offset = cuda_warp_t::shuffle(offset, 0);
        // Under an ID array, end_id bounds it; the assigned offset indexes it.
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_ids ? view.node_ids[offset] : view.first_id + offset;

        vram_warp_context<frontier_t> context;
        if constexpr (pool_slots_ == 0) {
            context.frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
            context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
            context.frontier.expansion_target_ = view.expansion;
        }
        context.random_generator = vram_lane_generator<cuda_warp_t>(node_id, lane);
        vram_visited_set<cuda_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                               view.epoch_base + offset + 1};
        vram_refine_one<kit_>( //
            view, node_id, visited, query_span, cache, schedule, selection, cache_norms[warp], gathered[warp], context,
            lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<cuda_warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** One warp merges one node's forward & reverse candidates into its final neighbors, assigning
 *  itself node IDs from the pass counter. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_>
__device__ __forceinline__ void cuda_merge_back_links(vram_refine_view<kit_> const& view, u32_t const* bucket_ids,
                                                      u32_t const* bucket_counts, u32_t bucket_width,
                                                      cuda_bitset_t touched) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_vector_cache<typename kit_::fragment_t> const cache {
        reinterpret_cast<typename kit_::fragment_t*>(slice.scratch), view.storage.row_fragments,
        view.arena_shape.cache_bytes ? gathered[warp].ids : nullptr};
    schedule_ schedule {schedule_states[warp], lane};
    while (true) {
        u32_t node_id = lane == 0 ? cuda_warp_t::fetch_add_relaxed(*view.assigned_rows, 1u) : 0;
        node_id = view.first_id + cuda_warp_t::shuffle(node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the walk neither re-selected nor picked keeps its neighbors: re-judging it could
        // only drop links.
        if (!cuda_bitset_contains(touched, node_id)) continue;

        vram_telemetry_t telemetry;
        vram_merge_back_links_one<kit_>( //
            view, bucket_ids, bucket_counts, bucket_width, node_id, cache, schedule, cache_norms[warp], gathered[warp],
            telemetry, lane);
        telemetry.template flush<cuda_warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
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
template <vram_quads_kit kit_>
__device__ __forceinline__ void cuda_gather_weighted_rows( //
    typename kit_::fragment_t const* row, u32_t count, f32_t const* weights, f32_t const* rescales, u32_t heads_count,
    u32_t columns, f32_t* accumulators, u32_t lane) noexcept {
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
                cuda_warp_t::shuffle(reinterpret_cast<std::uintptr_t>(row), source_lane));
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
__device__ __forceinline__ auto cuda_value_row(value_lookup_type_ const& value_lookup, node_id_t node_id) noexcept {
    return value_lookup(node_id);
}

/** Resolves @p node_id through a shard's device key column, then the caller's key-level lookup. */
template <typename key_type_, typename value_lookup_type_>
__device__ __forceinline__ auto cuda_value_row(cuda_keyed_lookup<key_type_, value_lookup_type_> const& lookup,
                                               node_id_t node_id) noexcept {
    return lookup.value_lookup(lookup.keys[node_id]);
}

/**
 *  @brief One token's attention as a walk fold, FlashAttention-style over every node the walk
 *      scores.
 *
 *  The per-head dots @ref cuda_heads_rescoring left in the warp's scratch become base-2 weights
 *  against each head's running maximum, and every value row the lookup resolves is read once and
 *  added into every head's accumulator row. The running maxima, partitions and rescales are
 *  warp-uniform, as every lane derives them from the same reductions.
 */
template <vram_quads_kit kit_, typename value_lookup_type_>
struct cuda_attend_fold {
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

    __device__ __forceinline__ walk_demand_t absorb(vram_gathered<kit_> const& gathered, u32_t gathered_count,
                                                    u32_t lane) noexcept {
        bool const live = lane < gathered_count;
        fragment_t const* const row =
            live ? static_cast<fragment_t const*>(cuda_value_row(value_lookup, gathered.ids[lane])) : nullptr;
        // A node without its value row would leave a wrong softmax, so the walk stops there and the
        // call refuses.
        missed = cuda_warp_t::any(live && !row);
        if (missed) return walk_demand_t::sated_k;
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const logit = live ? logit_scale * dots[head * 32 + lane] : std::numeric_limits<f32_t>::lowest();
            f32_t batch_maximum = logit;
            for (u32_t offset = cuda_warp_t::width_k / 2; offset; offset >>= 1)
                batch_maximum = fmaxf(batch_maximum, cuda_warp_t::shuffle_xor(batch_maximum, offset));
            f32_t const previous_maximum = maxima[head];
            f32_t const maximum = fmaxf(previous_maximum, batch_maximum);
            f32_t const weight = live ? exp2f(logit - maximum) : 0;
            f32_t batch_partition = weight;
            for (u32_t offset = cuda_warp_t::width_k / 2; offset; offset >>= 1)
                batch_partition += cuda_warp_t::shuffle_xor(batch_partition, offset);
            f32_t const rescale = exp2f(previous_maximum - maximum);
            if (live) dots[head * 32 + lane] = weight;
            // Every lane reads the previous state before the leader replaces it.
            cuda_warp_t::sync();
            if (lane == 0) {
                maxima[head] = maximum;
                partitions[head] = partitions[head] * rescale + batch_partition;
                rescales[head] = rescale;
            }
        }
        cuda_warp_t::sync();
        cuda_gather_weighted_rows<kit_>(row, gathered_count, dots, rescales, heads_count, columns, accumulators, lane);
        return walk_demand_t::continue_k;
    }
};

/** Writes each token's attention out of its walk: opens the fold over the warp's scratch, then
 *  writes every head's normalized row and its carry once the walk ends, and raises the miss word if
 *  the walk met a row-less node. */
template <vram_quads_kit kit_, typename value_lookup_type_>
struct cuda_attend_sink {
    using fold_t = cuda_attend_fold<kit_, value_lookup_type_>;

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
        cuda_warp_t::sync();
        return fold;
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, vram_warp_context<frontier_>&, u32_t query_index,
                                          u32_t lane) const noexcept {
        if (fold.missed && lane == 0) cuda_warp_t::store_relaxed(*missed, 1u);
        // The lanes accumulated by fragment and write out by column, so the rows cross lanes here.
        cuda_warp_t::sync();
        f32_t* const token_outputs = vram_row(outputs, query_index, heads_count * columns);
        softmax_state_t* const token_states = vram_row(states, query_index, heads_count);
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const partition = fold.partitions[head];
            f32_t const normalizer = partition > 0 ? 1 / partition : 0;
            f32_t const* const accumulated = fold.accumulators + head * columns;
            f32_t* const output = token_outputs + head * columns;
            for (u32_t column = lane; column < columns; column += 32) output[column] = accumulated[column] * normalizer;
            // Carries keep the natural-log maximum, so they merge with NumKong's log-sum-exp.
            if (lane == 0) token_states[head] = {fold.maxima[head] / std::numbers::log2e_v<f32_t>, partition};
        }
    }
};

/** One warp per query over the register frontier - expansions within the ladder - each walk
 *  scored through the schedule under @p rescorings and written out through @p sink; warps take
 *  queries from the batch counter, as builders take rows. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t pool_slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void cuda_walk(vram_search_view<kit_> const& view, sink_ const& sink,
                                          rescoring_ const& rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    while (true) {
        u32_t query_index = slice.lane == 0 ? cuda_warp_t::fetch_add_relaxed(*view.assigned_queries, 1u) : 0;
        query_index = cuda_warp_t::shuffle(query_index, 0);
        if (query_index >= view.queries_count) return;
        vram_visited_set<cuda_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                               view.epoch_base + query_index + 1};
        vram_walk_one<kit_, schedule_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, sink, rescorings, query_index, query_span, visited,
            vram_register_frontier<pool_slots_, cuda_warp_t> {}, gathered[slice.warp], slice.lane);
    }
}

/** One warp per query past the register ladder's width: the same walk over the arena frontier,
 *  whose budget is the requested expansion itself, ranking everything it kept at the end. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, typename sink_, typename rescoring_>
__device__ __forceinline__ void cuda_walk_wide(vram_search_view<kit_> const& view, sink_ const& sink,
                                               rescoring_ const& rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_> gathered[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_arena_frontier<cuda_warp_t> frontier;
    frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    while (true) {
        u32_t query_index = slice.lane == 0 ? cuda_warp_t::fetch_add_relaxed(*view.assigned_queries, 1u) : 0;
        query_index = cuda_warp_t::shuffle(query_index, 0);
        if (query_index >= view.queries_count) return;
        vram_visited_set<cuda_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                               view.epoch_base + query_index + 1};
        vram_walk_one<kit_, schedule_, 4u>(view, sink, rescorings, query_index, query_span, visited, frontier,
                                           gathered[slice.warp], slice.lane);
    }
}

/** Rebuilds the norms column from the vectors matrix for rows @c [first_row,first_row+rows_count),
 *  one warp per row; norms are derived state, never serialized. */
template <vram_quads_kit kit_>
__device__ __forceinline__ void cuda_derive_norms(vram_flat_view<kit_> const& storage, u32_t first_row,
                                                  u32_t rows_count) {
    vram_grid_warp_t const warp = vram_grid_warp<cuda_warp_t>();
    for (u32_t row = first_row + warp.index; row < first_row + rows_count; row += warp.count) {
        typename kit_::norms_accumulator_t accumulator {};
        for (u32_t fragment = warp.lane; fragment < storage.row_fragments; fragment += cuda_warp_t::width_k)
            accumulator = kit_::accumulate_norms(storage.vector_for(row, fragment), accumulator);
        accumulator = vram_reduce_norms<cuda_warp_t>(accumulator);
        if (warp.lane == 0) storage.norm_ref_for(row) = kit_::partials_to_norms(accumulator);
    }
}

/** Post-drain structural audit of the neighbors matrix: every row published, every ID in range,
 *  self-loop rows childless, no duplicate neighbors within a row. One warp scans one row, a
 *  warp-width of slots at a time; each bad row bumps @p violations once. */
static __global__ void cuda_invariant_kernel(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                             u64_t* violations) {
    constexpr u32_t width_k = cuda_warp_t::width_k;
    vram_grid_warp_t const warp = vram_grid_warp<cuda_warp_t>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        u32_t const* const slots = vram_row(neighbors, row, connectivity);
        u32_t const first = slots[0];
        bool bad = first == node_id_unknown_k;
        for (u32_t base = 0; base < connectivity; base += width_k) { // Warp-uniform trip count.
            u32_t const slot = base + warp.lane;
            u32_t const value = slot < connectivity ? slots[slot] : node_id_unknown_k;
            bad |= value != node_id_unknown_k && value >= rows_count;
            bad |= first == row && slot > 0 && value != node_id_unknown_k;
            // Duplicate-neighbor scan: rotate every not-earlier batch past this lane's value.
            for (u32_t other_base = base; other_base < connectivity; other_base += width_k) {
                u32_t const other_own = other_base + warp.lane;
                u32_t const other_value = other_base == base         ? value
                                          : other_own < connectivity ? slots[other_own]
                                                                     : node_id_unknown_k;
                for (u32_t shift = 0; shift < width_k; ++shift) {
                    u32_t const source = (warp.lane + shift) % width_k;
                    u32_t const other = cuda_warp_t::shuffle(other_value, source);
                    u32_t const other_slot = other_base + source;
                    bad |= value != node_id_unknown_k && other == value && other_slot > slot &&
                           other_slot < connectivity;
                }
            }
        }
        if (cuda_warp_t::ballot(bad) && warp.lane == 0) cuda_warp_t::fetch_add_relaxed(*violations, u64_t {1});
    }
}

/** Marks @p seed_count stride-spread rows reached and in the first frontier - the audit's landmark
 *  set, spread over the ID space like the traversal samplers. */
__device__ __forceinline__ void cuda_mark_seeds(cuda_bitset_t reached, cuda_bitset_t frontier, u32_t rows_count,
                                                u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    cuda_bitset_insert(reached, row);
    cuda_bitset_insert(frontier, row);
}

/** One level of forward reachability: every row of @p frontier marks its targets reached, and the
 *  ones it reaches first join @p next_frontier. Only the frontier expands, so a level is the same
 *  set of rows however the warps interleave, and sweeping until a level expands nothing takes the
 *  landmarks' eccentricity in levels; a level whose @p expanded_before reads zero was queued past
 *  convergence and has nothing to expand. The first expansion of a level counts it in @p levels, so
 *  the host reads the depth from one word. */
__device__ __forceinline__ void cuda_reachability_sweep(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                        cuda_bitset_t reached, cuda_bitset_t frontier,
                                                        cuda_bitset_t next_frontier, u32_t* expanded,
                                                        u32_t const* expanded_before, u32_t* levels) {
    // The level before landed ahead of this launch, so every thread reads one settled word.
    if (expanded_before && !*expanded_before) return;
    vram_grid_warp_t const warp = vram_grid_warp<cuda_warp_t>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!cuda_bitset_contains(frontier, row)) continue;
        if (warp.lane == 0 && !cuda_warp_t::fetch_add_relaxed(*expanded, 1u))
            cuda_warp_t::fetch_add_relaxed(*levels, 1u);
        vram_for_each_link<cuda_warp_t>(neighbors, connectivity, rows_count, row, warp.lane,
                                        [&](u32_t target) noexcept {
                                            if (cuda_bitset_insert(reached, target))
                                                cuda_bitset_insert(next_frontier, target);
                                        });
    }
}

/** Counts the rows @p reached lacks - the island population - in @p islands_cursor, collecting
 *  their IDs into @p island_ids when it is given. */
__device__ __forceinline__ void cuda_reachability_islands(cuda_bitset_t reached, u32_t rows_count, u32_t* island_ids,
                                                          u32_t* islands_cursor) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count || cuda_bitset_contains(reached, row)) return;
    u32_t const slot = cuda_warp_t::fetch_add_relaxed(*islands_cursor, 1u);
    if (island_ids) island_ids[slot] = row;
}

/** Counts, for every row, the reached rows that point at it - the links a reconnection may take
 *  from a reached resident only while another one remains. */
__device__ __forceinline__ void cuda_reached_in_links(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                      cuda_bitset_t reached, u32_t* in_links) {
    vram_grid_warp_t const warp = vram_grid_warp<cuda_warp_t>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!cuda_bitset_contains(reached, row)) continue;
        vram_for_each_link<cuda_warp_t>(
            neighbors, connectivity, rows_count, row, warp.lane,
            [&](u32_t target) noexcept { cuda_warp_t::fetch_add_relaxed(in_links[target], 1u); });
    }
}

/** Reconnects each island into the first of its reached forward picks with room for it, or into the
 *  landmark covering its ID, bypassing the diversity judgment that dropped it: a free slot when one
 *  is left, otherwise a landing spread across the tail. Every landing is a compare-and-swap that
 *  never displaces an unreached sibling, so islands sharing one target all land, nor a reached
 *  resident whose @p in_links would drop to zero, so no reconnection strands another node;
 *  @p reconnected counts the links actually written. Exclusive pass: the launch boundary publishes
 *  every write, so each access is relaxed. */
__device__ __forceinline__ void cuda_reconnect_island(u32_t* neighbors, u32_t connectivity, u32_t const* island_ids,
                                                      u32_t islands_count, cuda_bitset_t reached, u32_t rows_count,
                                                      u32_t seed_count, u32_t* in_links, u32_t* reconnected) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const spread = heal_spread(connectivity);
    constexpr cuda::memory_order relaxed_k = cuda::memory_order_relaxed;

    // Whether the island now sits in the target's row, landing it there when a slot allows.
    auto const land = [&](u32_t target) noexcept {
        u32_t* const target_row = vram_row(neighbors, target, connectivity);
        auto const claim = [&](u32_t slot, u32_t resident) noexcept {
            if (!cuda_atomic_ref<u32_t>(target_row[slot]).compare_exchange_strong(resident, island, relaxed_k))
                return false;
            cuda_atomic_ref<u32_t>(*reconnected).fetch_add(1u, relaxed_k);
            return true;
        };
        for (u32_t slot = 0; slot != connectivity; ++slot) {
            u32_t const resident = cuda_atomic_ref<u32_t>(target_row[slot]).load(relaxed_k);
            if (resident == island) return true; // Already wired; the audit raced a heal.
            if (resident != node_id_unknown_k && resident != target) continue;
            if (claim(slot, resident)) return true;
        }
        for (u32_t attempt = 0; attempt != spread; ++attempt) {
            u32_t const landing = connectivity - 1 - (island + attempt) % spread;
            u32_t const resident = cuda_atomic_ref<u32_t>(target_row[landing]).load(relaxed_k);
            if (resident == island) return true;
            if (resident != node_id_unknown_k && !cuda_bitset_contains(reached, resident))
                continue; // A sibling island landed first.
            // A reached resident gives this link up only while another reached row still points at
            // it.
            bool const evicting = resident != node_id_unknown_k && resident != target;
            if (evicting && cuda_atomic_ref<u32_t>(in_links[resident]).fetch_sub(1u, relaxed_k) <= 1u) {
                cuda_atomic_ref<u32_t>(in_links[resident]).fetch_add(1u, relaxed_k);
                continue;
            }
            if (claim(landing, resident)) return true;
            if (evicting) cuda_atomic_ref<u32_t>(in_links[resident]).fetch_add(1u, relaxed_k);
        }
        return false;
    };

    u32_t const* const island_row = vram_row(neighbors, island, connectivity);
    for (u32_t slot = 0; slot != connectivity; ++slot) {
        u32_t const pick = island_row[slot];
        if (pick != node_id_unknown_k && pick != island && cuda_bitset_contains(reached, pick) && land(pick)) return;
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
__device__ __forceinline__ void cuda_merge_attention(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards) {
    u32_t const row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
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
        for (u32_t column = lane; column < columns; column += 32)
            merged[column] = (shard ? merged[column] : 0) + share * partial[column];
    }
    // Every lane reads the first carry above before the leader replaces it.
    cuda_warp_t::sync();
    if (lane == 0) carry_of(0) = {maximum, partition};
}

/**
 *  @brief Scans every query against every visible row on the kit's tile step, a persistent grid of
 *      blocks claiming work items of one block of queries against one range of rows.
 *
 *  An item keeps each query's nearest @c count matches of its range in the scratch, and the last
 *  range of a block of queries to finish merges every range's matches into the sink. Each step
 *  multiplies a square tile of rows by the block's queries, four warps of 32 by 32, every warp two
 *  query tiles by four row tiles. Per 64 bytes of depth a lane loads one fragment of each of its
 *  queries and rows, its first two words feeding one tile step and its last two the next, so
 *  queries and rows take one permutation of the dimensions; a kit without a tensor-core step dots
 *  whole fragments instead, gathered from the four lanes holding them. The kit finalizes every dot
 *  as the walk does, and a candidate beating its query's worst match stages in shared memory; once
 *  a step could overflow some query's staging, and at the end, every query keeps its staged
 *  candidates among its matches, a warp per query.
 *
 *  Items are claimed through @c assigned_queries range by range, so the blocks in flight read the
 *  same rows. The scratch holds every block of queries' count of finished ranges, then each query's
 *  matches in every range, a short range's ending in the sentinel. Candidates order as
 *  @c vram_scored_node_t, so a tie in distance keeps the lower row. @p kernel_ fixes the ranking
 *  metric, so the epilogue branches on none, while the matches convert out by the sink's own
 *  metric, as a walk's do.
 */
template <vram_quads_kit kit_, nk_kernel_kind_t kernel_>
__device__ __forceinline__ void cuda_scan(vram_search_view<kit_> const& view, vram_search_sink<kit_> const& sink) {
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;
    using norms_t = typename kit_::norms_t;
    using tile_t = std::array<dot_t, 4>;
    static_assert(vram_warps_per_block_k == 4, "Four warps split a scanning step two by two");
    constexpr u32_t tile_k = cuda_scan_tile_k, staging_k = 2 * tile_k, rounds_k = staging_k / cuda_warp_t::width_k;
    constexpr u32_t width_k = cuda_warp_t::width_k;
    extern __shared__ vram_scored_node_t staged[];
    __shared__ vram_scored_node_t worst[tile_k];
    __shared__ u32_t staged_counts[tile_k], kept_counts[tile_k];
    __shared__ norms_t query_norms[tile_k];
    __shared__ u32_t staging_full[3];
    __shared__ u32_t claimed;
    __shared__ bool merging;

    vram_flat_view<kit_> const& storage = view.storage;
    u32_t const warp = threadIdx.x / width_k, lane = threadIdx.x % width_k;
    u32_t const group = lane / 4, quarter = lane % 4;
    u32_t const warp_queries = (warp >> 1) * 32, warp_rows = (warp & 1) * 32;
    u32_t const blocks = divide_round_up(view.queries_count, tile_k);
    u32_t const visible = view.sampling_bound, count = sink.count;
    u32_t const ranges = cuda_scan_ranges(gridDim.x, blocks, visible);
    u32_t const range_rows = round_up_to_multiple(divide_round_up(visible, ranges), tile_k);
    u32_t const spans = divide_round_up(storage.row_fragments, 4u);
    u32_t* const finished_ranges = reinterpret_cast<u32_t*>(view.arena);
    vram_scored_node_t* const matches = reinterpret_cast<vram_scored_node_t*>(finished_ranges +
                                                                              round_up_to_multiple(blocks, 4u));

    for (;;) {
        // The barrier also parts the last item's merge from this item's staging.
        if (threadIdx.x == 0) claimed = cuda_warp_t::fetch_add_relaxed(*view.assigned_queries, 1u);
        __syncthreads();
        u32_t const item = claimed;
        if (item >= blocks * ranges) return;
        u32_t const range = item / blocks, block = item % blocks;
        u32_t const first_query = block * tile_k, block_queries = min_of(tile_k, view.queries_count - first_query);
        u32_t const first_row = static_cast<u32_t>(min_of<u64_t>(u64_t {range} * range_rows, visible));
        u32_t const end_row = static_cast<u32_t>(min_of<u64_t>(u64_t {first_row} + range_rows, visible));

        // Queries past the block or keeping none rank nothing: no entry orders below the zero word.
        for (u32_t local = threadIdx.x; local < tile_k; local += blockDim.x) {
            worst[local] = local < block_queries && count ? vram_scored_node_t::sentinel() : vram_scored_node_t {0};
            staged_counts[local] = 0, kept_counts[local] = 0;
        }
        if (threadIdx.x < 3) staging_full[threadIdx.x] = 0;
        for (u32_t local = warp; local < tile_k; local += vram_warps_per_block_k) {
            fragment_t const* const query_row = vram_row(view.queries, first_query + min_of(local, block_queries - 1),
                                                         storage.row_fragments);
            typename kit_::norms_accumulator_t accumulator {};
            for (u32_t fragment = lane; fragment < storage.row_fragments; fragment += width_k)
                accumulator = kit_::accumulate_norms(query_row[fragment], accumulator);
            accumulator = vram_reduce_norms<cuda_warp_t>(accumulator);
            if (lane == 0) query_norms[local] = kit_::partials_to_norms(accumulator);
        }
        __syncthreads();

        // A lane holds queries `group` and `group + 8` against rows `2 * quarter` and one past it.
        fragment_t const* query_rows[2][2];
        norms_t lane_norms[2][2];
        f32_t radii[2][2];
#pragma unroll
        for (u32_t tile = 0; tile != 2; ++tile)
#pragma unroll
            for (u32_t half = 0; half != 2; ++half) {
                u32_t const local = warp_queries + tile * 16 + half * 8 + group;
                query_rows[tile][half] = vram_row(view.queries, first_query + min_of(local, block_queries - 1),
                                                  storage.row_fragments);
                lane_norms[tile][half] = query_norms[local];
            }

        for (u32_t base = first_row, step = 0;; base += tile_k, ++step) {
            bool const finished = base >= end_row;
            if (step == 0 || finished || staging_full[(step + 2) % 3]) {
                // Kept matches only move up, so moving them top down overwrites none unread.
                for (u32_t local = warp; local < block_queries; local += vram_warps_per_block_k) {
                    u32_t const pending = staged_counts[local], held = kept_counts[local];
                    if (!pending) continue;
                    vram_scored_node_t* const kept = matches +
                                                     (std::size_t {first_query + local} * ranges + range) * count;
                    vram_scored_node_t const* const candidates = staged + local * staging_k;
                    u32_t places[rounds_k];
#pragma unroll
                    for (u32_t round = 0; round != rounds_k; ++round) {
                        u32_t const index = round * width_k + lane;
                        places[round] = 0;
                        if (index >= pending) continue;
                        vram_scored_node_t const candidate = candidates[index];
                        for (u32_t other = 0; other != pending; ++other) places[round] += candidates[other] < candidate;
                        places[round] += vram_rank_below(held, candidate.bits_, vram_ties_t::excluded_k,
                                                         [&](u32_t rank) noexcept { return kept[rank].bits_; });
                    }
                    for (u32_t top = held; top;) {
                        u32_t const first = top > width_k ? top - width_k : 0;
                        u32_t const index = first + lane;
                        vram_scored_node_t const value = index < top ? kept[index] : vram_scored_node_t::sentinel();
                        cuda_warp_t::sync();
                        u32_t place = index;
                        for (u32_t other = 0; other != pending; ++other) place += candidates[other] < value;
                        if (index < top && place < count) kept[place] = value;
                        cuda_warp_t::sync();
                        top = first;
                    }
#pragma unroll
                    for (u32_t round = 0; round != rounds_k; ++round) {
                        u32_t const index = round * width_k + lane;
                        if (index < pending && places[round] < count) kept[places[round]] = candidates[index];
                    }
                    cuda_warp_t::sync();
                    u32_t const now_held = min_of(held + pending, count);
                    if (lane == 0) {
                        kept_counts[local] = now_held, staged_counts[local] = 0;
                        worst[local] = now_held == count ? kept[count - 1] : vram_scored_node_t::sentinel();
                    }
                }
                __syncthreads();
                // The widest distance the worst match covers, or NaN until the matches fill.
#pragma unroll
                for (u32_t tile = 0; tile != 2; ++tile)
#pragma unroll
                    for (u32_t half = 0; half != 2; ++half) {
                        u32_t const widest = worst[warp_queries + tile * 16 + half * 8 + group].distance_bits() << 1 |
                                             1u;
                        radii[tile][half] = __uint_as_float(widest ^
                                                            (widest & 0x80000000u ? 0x80000000u : 0xFFFFFFFFu));
                    }
                if (finished) break;
            }
            // The next step's flag was read two barriers ago, and is raised past this one.
            if (threadIdx.x == 0) staging_full[(step + 1) % 3] = 0;

            fragment_t const* rows[4];
            norms_t row_norms[4][2];
#pragma unroll
            for (u32_t tile = 0; tile != 4; ++tile) {
                rows[tile] = storage.vector_span_for(min_of(base + warp_rows + tile * 8 + group, end_row - 1));
                // Requested ahead of the products, which hide their latency from the epilogue.
#pragma unroll
                for (u32_t column = 0; column != 2; ++column)
                    row_norms[tile][column] = storage.norm_for(
                        min_of(base + warp_rows + tile * 8 + quarter * 2 + column, end_row - 1));
            }
            tile_t tiles[2][4] = {};
            for (u32_t span = 0; span != spans; ++span) {
                u32_t const fragment = span * 4 + quarter;
                bool const in_row = fragment < storage.row_fragments;
                fragment_t query_pieces[2][2], pieces[4];
#pragma unroll
                for (u32_t tile = 0; tile != 2; ++tile)
#pragma unroll
                    for (u32_t half = 0; half != 2; ++half)
                        query_pieces[tile][half] = in_row ? query_rows[tile][half][fragment] : fragment_t {};
#pragma unroll
                for (u32_t tile = 0; tile != 4; ++tile) pieces[tile] = in_row ? rows[tile][fragment] : fragment_t {};
                if constexpr (requires(tile_t tile) { kit_::accumulate_tile(fragment_t {}, uint2 {}, tile); }) {
#pragma unroll
                    for (u32_t query_tile = 0; query_tile != 2; ++query_tile) {
                        fragment_t const top = query_pieces[query_tile][0], bottom = query_pieces[query_tile][1];
                        fragment_t const first {top.x, bottom.x, top.y, bottom.y},
                            second {top.z, bottom.z, top.w, bottom.w};
#pragma unroll
                        for (u32_t row_tile = 0; row_tile != 4; ++row_tile) {
                            fragment_t const row = pieces[row_tile];
                            tile_t& tile = tiles[query_tile][row_tile];
                            tile = kit_::accumulate_tile(first, uint2 {row.x, row.y}, tile);
                            tile = kit_::accumulate_tile(second, uint2 {row.z, row.w}, tile);
                        }
                    }
                }
                else {
#pragma unroll
                    for (u32_t holder = 0; holder != 4; ++holder) {
                        fragment_t queries[2][2];
#pragma unroll
                        for (u32_t query_tile = 0; query_tile != 2; ++query_tile)
#pragma unroll
                            for (u32_t half = 0; half != 2; ++half) {
                                fragment_t const piece = query_pieces[query_tile][half];
                                u32_t const source = group * 4 + holder;
                                queries[query_tile][half] = {
                                    cuda_warp_t::shuffle(piece.x, source), cuda_warp_t::shuffle(piece.y, source),
                                    cuda_warp_t::shuffle(piece.z, source), cuda_warp_t::shuffle(piece.w, source)};
                            }
#pragma unroll
                        for (u32_t row_tile = 0; row_tile != 4; ++row_tile)
#pragma unroll
                            for (u32_t column = 0; column != 2; ++column) {
                                fragment_t const piece = pieces[row_tile];
                                u32_t const source = (quarter * 2 + column) * 4 + holder;
                                fragment_t const row {
                                    cuda_warp_t::shuffle(piece.x, source), cuda_warp_t::shuffle(piece.y, source),
                                    cuda_warp_t::shuffle(piece.z, source), cuda_warp_t::shuffle(piece.w, source)};
#pragma unroll
                                for (u32_t query_tile = 0; query_tile != 2; ++query_tile)
#pragma unroll
                                    for (u32_t half = 0; half != 2; ++half) {
                                        dot_t& dot = tiles[query_tile][row_tile][half * 2 + column];
                                        dot = kit_::accumulate_dot(queries[query_tile][half], row, dot);
                                    }
                            }
                    }
                }
            }
#pragma unroll
            for (u32_t query_tile = 0; query_tile != 2; ++query_tile)
#pragma unroll
                for (u32_t half = 0; half != 2; ++half)
#pragma unroll
                    for (u32_t row_tile = 0; row_tile != 4; ++row_tile)
#pragma unroll
                        for (u32_t column = 0; column != 2; ++column) {
                            u32_t const row = base + warp_rows + row_tile * 8 + quarter * 2 + column;
                            dot_t const dot = tiles[query_tile][row_tile][half * 2 + column];
                            distance_t const distance = kit_::distance(kernel_, dot, lane_norms[query_tile][half],
                                                                       row_norms[row_tile][column]);
                            if (row >= end_row || distance > radii[query_tile][half]) continue;
                            vram_scored_node_t const candidate = vram_scored_node_t::make(distance, row);
                            u32_t const local = warp_queries + query_tile * 16 + half * 8 + group;
                            if (!(candidate < worst[local])) continue;
                            u32_t const slot = cuda_warp_t::fetch_add_relaxed(staged_counts[local], 1u);
                            staged[local * staging_k + slot] = candidate;
                            if (slot >= staging_k - tile_k) staging_full[step % 3] = 1;
                        }
            __syncthreads();
        }

        for (u32_t local = threadIdx.x; local < block_queries; local += blockDim.x) {
            vram_scored_node_t* const kept = matches + (std::size_t {first_query + local} * ranges + range) * count;
            if (kept_counts[local] < count) kept[kept_counts[local]] = vram_scored_node_t::sentinel();
        }
        if (threadIdx.x == 0)
            cuda_warp_t::fetch_add_relaxed(view.telemetry->scored, u64_t {end_row - first_row} * block_queries);
        // Every range's matches are written before the finish that lets the last range read them.
        cuda_warp_t::fence_release();
        __syncthreads();
        if (threadIdx.x == 0)
            merging = cuda_atomic_ref<u32_t>(finished_ranges[block]).fetch_add(1u, cuda::memory_order_acq_rel) ==
                      ranges - 1;
        __syncthreads();
        if (!merging) continue;
        cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);

        // Each lane keeps its ranges' nearest unmerged matches; the warp's nearest is written next.
        vram_scored_node_t* const heads = staged + warp * ranges;
        u32_t* const taken = reinterpret_cast<u32_t*>(staged + vram_warps_per_block_k * ranges) + warp * ranges;
        for (u32_t local = warp; local < block_queries; local += vram_warps_per_block_k) {
            u32_t const query = first_query + local;
            vram_scored_node_t* const query_matches = matches + std::size_t {query} * ranges * count;
            node_id_t* const ids_row = vram_row(sink.result_ids, query, count);
            distance_t* const distances_row = sink.result_distances ? vram_row(sink.result_distances, query, count)
                                                                    : nullptr;
            for (u32_t of_range = lane; of_range < ranges; of_range += width_k) {
                heads[of_range] = count ? vram_scored_node_t {cuda_warp_t::load_relaxed(
                                              query_matches[std::size_t {of_range} * count].bits_)}
                                        : vram_scored_node_t::sentinel();
                taken[of_range] = 0;
            }
            u32_t chosen = ranges;
            bool advanced = true;
            for (u32_t rank = 0; rank != count; ++rank) {
                if (advanced) {
                    chosen = lane;
                    for (u32_t of_range = lane + width_k; of_range < ranges; of_range += width_k)
                        if (heads[of_range] < heads[chosen]) chosen = of_range;
                }
                vram_scored_node_t const held = chosen < ranges ? heads[chosen] : vram_scored_node_t::sentinel();
                vram_scored_node_t least = held;
#pragma unroll
                for (u32_t mask = width_k / 2; mask; mask >>= 1) {
                    vram_scored_node_t const other = shuffle_xor<cuda_warp_t>(least, mask);
                    if (other < least) least = other;
                }
                if (lane == 0) {
                    ids_row[rank] = least.id();
                    if (distances_row) distances_row[rank] = kit_::convert_to_distance(sink.kernel, least.distance());
                }
                // Every row sits in one range, so one lane at most holds the nearest.
                advanced = held == least && !(least == vram_scored_node_t::sentinel());
                if (advanced) {
                    u32_t const position = ++taken[chosen];
                    heads[chosen] = position < count
                                        ? vram_scored_node_t {cuda_warp_t::load_relaxed(
                                              query_matches[std::size_t {chosen} * count + position].bits_)}
                                        : vram_scored_node_t::sentinel();
                }
            }
        }
    }
}

#pragma region Views

/** The kit of one dtype under one capability ceiling, both as the kernels' template integers. */
template <u32_t capabilities_, u32_t dtype_>
using cuda_kit =
    cuda_kit_t<static_cast<nk_dtype_t>(dtype_), static_cast<nk_capability_t>(capabilities_) * nk_cap_cuda_k>;

/** The shared views and sinks, field by field from the host's records. */
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_flat_view<kit_> cuda_view(cuda_flat_storage_t const& storage) noexcept {
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
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_build_view<kit_> cuda_view(cuda_build_t const& build) noexcept {
    return {cuda_view<kit_>(build.storage),
            reinterpret_cast<typename kit_::fragment_t const*>(build.staging),
            build.end_id,
            build.scratch_bytes_per_warp,
            build.first_id,
            build.arena,
            cuda_arena_shape(build.arena_shape),
            build.expansion,
            build.epoch_base,
            build.published_rows,
            build.telemetry,
            build.assigned_rows};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_search_view<kit_> cuda_view(cuda_search_t const& search) noexcept {
    return {cuda_view<kit_>(search.storage),
            reinterpret_cast<typename kit_::fragment_t const*>(search.queries),
            search.queries_count,
            search.sampling_bound,
            search.scratch_bytes_per_warp,
            search.expansion,
            search.arena,
            cuda_arena_shape(search.arena_shape),
            search.epoch_base,
            search.telemetry,
            search.assigned_queries};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_refine_view<kit_> cuda_view(cuda_refine_t const& refine) noexcept {
    return {cuda_view<kit_>(refine.storage),
            refine.shadow_neighbors,
            refine.end_id,
            refine.scratch_bytes_per_warp,
            refine.first_id,
            refine.arena,
            cuda_arena_shape(refine.arena_shape),
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
__device__ __forceinline__ vram_search_sink<kit_> cuda_view(cuda_search_sink_t const& sink) noexcept {
    return {sink.result_ids, sink.result_distances, sink.count, sink.kernel};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_multiply_sink<kit_> cuda_view(cuda_multiply_sink_t const& sink) noexcept {
    return {sink.ids, reinterpret_cast<typename kit_::dot_t*>(sink.products), sink.counts, sink.wanted};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_indexer_rescoring<kit_> cuda_view(cuda_rescoring_t const& rescoring) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t const*>(rescoring.heads), rescoring.weights,
            rescoring.heads_count, rescoring.row_fragments};
}
__device__ __forceinline__ vram_ranking_view cuda_view(cuda_ranking_t const& ranking) noexcept {
    return {ranking.keys,          ranking.scores, ranking.counts,    ranking.keys_stride,
            ranking.scores_stride, ranking.width,  ranking.key_bytes, ranking.score_bytes};
}

#pragma endregion Views

#pragma region Kernels

/** Runs one walk family on its rung, the arena frontier at slot zero. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t slots_, typename sink_, typename rescoring_>
__device__ __forceinline__ void cuda_walk_rung(vram_search_view<kit_> const& view, sink_ const& sink,
                                               rescoring_ const& rescorings) {
    if constexpr (slots_ == 0) cuda_walk_wide<kit_, schedule_>(view, sink, rescorings);
    else cuda_walk<kit_, schedule_, slots_>(view, sink, rescorings);
}

/** Blocks of each kernel resident per multiprocessor, the register budget its launch bounds set:
 *  the builders' saturating count, a walk's by its rung, and one for the light graph sweeps. */
constexpr u32_t cuda_resident_blocks(cuda_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case cuda_kernel_t::insert_k:
    case cuda_kernel_t::refine_k: return cuda_insert_blocks_per_multiprocessor_k;
    case cuda_kernel_t::merge_back_links_k: return vram_resident_blocks_per_multiprocessor_k;
    case cuda_kernel_t::search_k:
    case cuda_kernel_t::search_indexer_k:
    case cuda_kernel_t::multiply_k: return slots <= 8 ? 4 : 3;
    case cuda_kernel_t::scan_k: return cuda_scan_blocks_per_multiprocessor_k;
    // The attend fold carries per-head state, which spills at 8 slots under four blocks' budget.
    case cuda_kernel_t::attend_k: return slots <= 4 ? 4 : 3;
    default: return 1;
    }
}

/** Every NVIDIA kernel: @p kernel_ picks the body, @p capabilities_ and @p dtype_ the kit,
 *  @p capabilities_ alone the schedule, @p slots_ the rung, and @p lookup_ an attend's
 *  value lookup through @ref cuda_value_lookup_of. */
template <u32_t kernel_, u32_t capabilities_, u32_t dtype_, u32_t slots_, u32_t lookup_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32,
                                  cuda_resident_blocks(static_cast<cuda_kernel_t>(kernel_), slots_))
    cuda_kernel(cuda_arguments_t arguments) {
    using kit_t = cuda_kit<capabilities_, dtype_>;
    using schedule_t = cuda_schedule_of<static_cast<nk_capability_t>(capabilities_) * nk_cap_cuda_k>;
    constexpr cuda_kernel_t kernel = static_cast<cuda_kernel_t>(kernel_);
    cuda_graph_t const& graph = arguments.graph;
    // Launched overlapping its predecessor's drain, so nothing is read before that predecessor
    // lands; the successor's blocks may then seat as this grid's blocks retire.
    if constexpr (cuda_overlaps_launches(static_cast<nk_capability_t>(capabilities_) * nk_cap_cuda_k))
        asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory");
    if constexpr (kernel == cuda_kernel_t::insert_k)
        cuda_insert<kit_t, schedule_t, slots_>(cuda_view<kit_t>(arguments.build));
    else if constexpr (kernel == cuda_kernel_t::refine_k)
        cuda_refine<kit_t, schedule_t, slots_>(cuda_view<kit_t>(arguments.refine));
    else if constexpr (kernel == cuda_kernel_t::search_k)
        cuda_walk_rung<kit_t, schedule_t, slots_>(cuda_view<kit_t>(arguments.search),
                                                  cuda_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    else if constexpr (kernel == cuda_kernel_t::search_indexer_k) {
        // Only the floating dtypes take indexer heads; others export a plain walk never launched.
        if constexpr (floating_scalar<typename kit_t::scalar_t>)
            cuda_walk_rung<kit_t, schedule_t, slots_>(cuda_view<kit_t>(arguments.search),
                                                      cuda_view<kit_t>(arguments.search_sink),
                                                      cuda_view<kit_t>(arguments.rescoring));
        else
            cuda_walk_rung<kit_t, schedule_t, slots_>(cuda_view<kit_t>(arguments.search),
                                                      cuda_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    }
    else if constexpr (kernel == cuda_kernel_t::scan_k) {
        vram_search_view<kit_t> const view = cuda_view<kit_t>(arguments.search);
        vram_search_sink<kit_t> const sink = cuda_view<kit_t>(arguments.search_sink);
        // Each kit's last metric is the one its `distance` falls back to.
        if constexpr (std::is_same_v<typename kit_t::scalar_t, nk::u1x8_t>) switch (sink.kernel) {
            case nk_kernel_dot_k: cuda_scan<kit_t, nk_kernel_dot_k>(view, sink); break;
            case nk_kernel_hamming_k: cuda_scan<kit_t, nk_kernel_hamming_k>(view, sink); break;
            default: cuda_scan<kit_t, nk_kernel_jaccard_k>(view, sink); break;
            }
        else switch (sink.kernel) {
            case nk_kernel_dot_k: cuda_scan<kit_t, nk_kernel_dot_k>(view, sink); break;
            case nk_kernel_angular_k: cuda_scan<kit_t, nk_kernel_angular_k>(view, sink); break;
            default: cuda_scan<kit_t, nk_kernel_euclidean_k>(view, sink); break;
            }
    }
    else if constexpr (kernel == cuda_kernel_t::multiply_k)
        cuda_walk_rung<kit_t, schedule_t, slots_>(cuda_view<kit_t>(arguments.search),
                                                  cuda_view<kit_t>(arguments.multiply_sink), vram_no_rescoring {});
    else if constexpr (kernel == cuda_kernel_t::attend_k) {
        using value_lookup_t = typename cuda_value_lookup_of<lookup_>::type;
        cuda_attend_t const& attend = arguments.attend;
        cuda_rescoring_t const& heads = arguments.rescoring;
        value_lookup_t value_lookup;
        std::memcpy(&value_lookup, attend.value_lookup, sizeof(value_lookup_t));
        cuda_walk_rung<kit_t, schedule_t, slots_>(
            cuda_view<kit_t>(arguments.search),
            cuda_attend_sink<kit_t, value_lookup_t> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                     attend.logit_scale, attend.heads_count, attend.columns},
            cuda_heads_rescoring<kit_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads), nullptr,
                                         heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel == cuda_kernel_t::merge_back_links_k)
        cuda_merge_back_links<kit_t, schedule_t>(cuda_view<kit_t>(arguments.refine), graph.bucket_ids,
                                                 graph.bucket_counts, graph.bucket_width, graph.touched);
    else if constexpr (kernel == cuda_kernel_t::scatter_back_links_k)
        cuda_scatter_back_links(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count, graph.node_ids,
                                graph.connectivity, graph.bucket_width, graph.bucket_ids, graph.bucket_counts,
                                graph.touched);
    else if constexpr (kernel == cuda_kernel_t::derive_norms_k)
        cuda_derive_norms<kit_t>(cuda_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel == cuda_kernel_t::rebase_links_k)
        vram_rebase_links<cuda_warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                       graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel == cuda_kernel_t::mark_seeds_k)
        cuda_mark_seeds(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel == cuda_kernel_t::reachability_sweep_k)
        cuda_reachability_sweep(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached, graph.frontier,
                                graph.next_frontier, graph.counter, graph.previous_counter, graph.levels);
    else if constexpr (kernel == cuda_kernel_t::reachability_islands_k)
        cuda_reachability_islands(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel == cuda_kernel_t::reached_in_links_k)
        cuda_reached_in_links(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached, graph.in_links);
    else if constexpr (kernel == cuda_kernel_t::reconnect_island_k)
        cuda_reconnect_island(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count, graph.reached,
                              graph.rows_count, graph.seed_count, graph.in_links, graph.reconnected);
    else if constexpr (kernel == cuda_kernel_t::cast_k) {
        nk_size_t const threads = static_cast<nk_size_t>(gridDim.x) * blockDim.x;
        for (nk_size_t unit = static_cast<nk_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             unit < arguments.cast.plan.units; unit += threads)
            nk_cast_unit_simt_(&arguments.cast, unit);
    }
    else if constexpr (kernel == cuda_kernel_t::rmscast_k) {
        // Only narrow floats take a unit-RMS import; every other dtype exports one never launched.
        if constexpr (dtype_ == nk_f16_k) nk_each_rmscast_f16_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e4m3_k) nk_each_rmscast_e4m3_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e5m2_k) nk_each_rmscast_e5m2_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e2m3_k) nk_each_rmscast_e2m3_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e3m2_k) nk_each_rmscast_e3m2_cuda_vectors_(arguments.rmscast);
    }
    else if constexpr (kernel == cuda_kernel_t::resolve_matches_k)
        vram_resolve_matches<cuda_warp_t>(arguments.matches.ids, arguments.matches.distances, arguments.matches.walked,
                                          arguments.matches.keys, arguments.matches.queries_count,
                                          cuda_view(arguments.matches.ranking));
    else if constexpr (kernel == cuda_kernel_t::merge_matches_k)
        vram_merge_matches<cuda_warp_t>(cuda_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                        arguments.matches.shards, arguments.matches.queries_count,
                                        cuda_view(arguments.matches.ranking));
    else
        cuda_merge_attention(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
                             arguments.attend.columns, arguments.attend.shards);
}

template <u32_t capabilities_, u32_t dtype_, std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<static_cast<u32_t>(cuda_rungs[rungs_].kernel), capabilities_, dtype_,
                                    cuda_rungs[rungs_].slots, 0>),
     ...);
}

template <u32_t capabilities_, u32_t dtype_, u32_t lookup_, std::size_t... rungs_>
__device__ void cuda_instantiate_attends(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<static_cast<u32_t>(cuda_kernel_t::attend_k), capabilities_, dtype_,
                                    cuda_walk_rungs[rungs_], lookup_>),
     ...);
}

/** Instantiates every kernel one dtype exports under one ceiling, and its attends over each of
 *  @p lookups_; a unit compiling a fatbin names it once per compiled dtype. Taking the kernels'
 *  addresses in device code makes NVCC emit them, while a host-only pass emits no launch stub. */
template <nk_capability_t capability_, nk_dtype_t dtype_, u32_t... lookups_>
__device__ void cuda_instantiate() noexcept {
    constexpr u32_t capabilities = cuda_capability_bits(capability_), dtype = static_cast<u32_t>(dtype_);
    cuda_instantiate_rungs<capabilities, dtype>(
        std::make_index_sequence<sizeof(cuda_rungs) / sizeof(cuda_rungs[0])> {});
    (cuda_instantiate_attends<capabilities, dtype, lookups_>(
         std::make_index_sequence<sizeof(cuda_walk_rungs) / sizeof(cuda_walk_rungs[0])> {}),
     ...);
}

#pragma endregion Kernels

} // namespace unum::usearch::dense
