/**
 *  @file include/usearch/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 18, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy`

#include <atomic>  // `std::atomic` row claiming across concurrent inserting contexts
#include <mutex>   // `std::unique_lock` while one context grows the planes and claims
#include <bit>     // `std::bit_width` for hash-prefix widths, `std::has_single_bit` for connectivity
#include <span>    // `std::span` - the run a context maps its buffers onto
#include <utility> // `std::exchange` - a moved-from plan forgets its index
#include <limits>  // `std::numeric_limits` - the running maximum's identity

#include <cuda.h>         // Driver API: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`
#include <cuda_runtime.h> // Runtime API: streams, events, `cudaMemcpyAsync`
#if defined(__CUDACC__)
#include <cuda/atomic>     // `cuda::atomic_ref` - the device twin of `std::atomic_ref`
#include <cuda_pipeline.h> // `__pipeline_memcpy_async` - register-free global-to-shared copies
#endif

#include <forkunion/types.hpp> // `fu::round_up_to_multiple` & friends

#include <numkong/capabilities.h> // `nk_kernel_kind_t` & the kernel-kind constants
#include <numkong/matrix.hpp>
#include <numkong/types.h>
#include <numkong/vector.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"
#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_metric.cuh"

namespace unum {
namespace usearch {

/** Warps per thread block across every kernel of this design - four schedulers per multiprocessor
 *  on every modern NVIDIA architecture, one warp each. */
inline constexpr u32_t dense_cuda_warps_per_block_k = 4;

/** Resident blocks per multiprocessor - shared by the launch grids and the kernels' launch
 *  bounds, so the register allocator sizes to it; small-pool search kernels fit four. */
inline constexpr u32_t dense_cuda_resident_blocks_per_multiprocessor_k = 3;

/** Ceiling on concurrently in-flight candidate-vector gathers per lane - the memory-parallelism window
 *  over the @b candidate axis, never the dimension axis. */
inline constexpr u32_t dense_cuda_max_gathers_in_flight_k = 8;

/** Rows of the per-warp shared candidate cache; also bounds the diversity-selection pool prefix
 *  and the staging arrays. */
inline constexpr u32_t dense_cuda_cache_rows_k = 64;

/** Floor of the @c __nanosleep backoff while retrying a held slot-0 lock. */
inline constexpr u32_t lock_backoff_floor_ns_k = 512;

/** Jitter mask added to that floor, so convoying warps spread their retries. */
inline constexpr u32_t lock_backoff_jitter_mask_k = 1023;

/** Acquire attempts before a traversal drops a transiently locked candidate. */
inline constexpr u32_t traversal_peek_retries_k = 16;

/** Acquire attempts before a back-linker abandons one neighbor; abandonments are telemetered. */
inline constexpr u32_t back_link_lock_attempts_k = 256;

/** Admission-ramp floor: the in-flight window starts here and doubles with the published set,
 *  bounding bootstrap lock contention. */
inline constexpr u32_t admission_ramp_floor_k = 32;

/** Device-resident telemetry counters, one block per driving context, zeroed at each submission
 *  and read after its drain. */
struct dense_cuda_telemetry_block_t {
    u64_t hops, scored, lock_attempts, lock_skips, reselections, restarts, locked_drops;

    /** Scored candidates split by plane residency; zero unless the view names a resident range. */
    u64_t local_scored, remote_scored;

    /** Frontier tail entries shed past their slice, and visited-set claims out of probes. */
    u64_t tail_drops, visited_saturations;

    /** Nodes whose neighbors a refinement merge landed in a different sequence than they held. */
    u64_t changed_nodes;
};

#if defined(__CUDACC__)

/** Typed views over the three device planes, @b [rows,*] each: packed vectors, per-row norms, and
 *  fixed-capacity neighbors. Under the flat virtual-memory arena the node ID @b is the location -
 *  one multiply, no chunk directory. */
template <dense_cuda_kit kit_>
struct dense_cuda_planes_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** Vector plane, @b [rows,row_fragments] fragments, one aligned row per node. */
    fragment_t* vectors;

    /** Norms plane, @b [rows] records, written once pre-publication. */
    norms_t* norms;

    /** Neighbor plane, @b [rows,connectivity] IDs; slot 0 doubles as the lock word. */
    u32_t* neighbors;

    /** Neighbor slots per node: the stride of @c neighbors and the connectivity. */
    u32_t connectivity;

    /** Fragments per vector row - the stride of @c vectors; the row's byte length is the dimension
     *  count rounded up to whole fragments, tail bytes zeroed. */
    u32_t row_fragments;

    /** Lanes cooperating on one row while scoring: the smallest power of two covering
     *  @c row_fragments, capped at the warp width. */
    u32_t lanes_per_row;

    /** Scoring steps each of those lanes takes to cover one full row. */
    u32_t steps_per_row;

    /** Fixed-point reciprocal of @c row_fragments, @c ceil(2^31/row_fragments), so flattened word
     *  indices split without division; exact while word x row_fragments stays under 2^31. */
    u32_t row_fragments_magic;

    /** Rows physically resident on the executing device, feeding the locality telemetry; both zero
     *  when the view spans one device. */
    u32_t resident_begin = 0, resident_end = 0;

    __device__ __forceinline__ u32_t row_of_word(u32_t word) const noexcept {
        return static_cast<u32_t>((static_cast<u64_t>(word) * row_fragments_magic) >> 31);
    }
    __device__ __forceinline__ u32_t fragment_of_word(u32_t word, u32_t row) const noexcept {
        return word - row * row_fragments;
    }

    __device__ __forceinline__ fragment_t vector_for(dense_node_id_t node_id, u32_t fragment_index) const noexcept {
        return vectors[static_cast<std::size_t>(node_id) * row_fragments + fragment_index];
    }
    __device__ __forceinline__ fragment_t* vector_span_for(dense_node_id_t node_id) const noexcept {
        return vectors + static_cast<std::size_t>(node_id) * row_fragments;
    }
    __device__ __forceinline__ norms_t norm_for(dense_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ norms_t& norm_ref_for(dense_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ u32_t* neighbors_for(dense_node_id_t node_id) const noexcept {
        return neighbors + static_cast<std::size_t>(node_id) * connectivity;
    }
};

/**
 *  @brief Typed view over one node's neighbors - the only legal way to touch slots that
 *      concurrent warps may be reading, over @c cuda::atomic_ref at device scope.
 *
 *  Slot 0 is the combined lock and first-neighbor word: the all-ones sentinel while unpublished
 *  or locked, the node's own ID when published empty, otherwise the first neighbor. Acquisition
 *  is bounded - spinning against a descheduled owner warp would livelock - and an unlinked
 *  neighbor is the telemetered cost.
 */
struct dense_cuda_neighbors_ref_t {
    u32_t* row_; // Non-const even for readers: `cuda::atomic_ref` cannot bind const words pre-P3323.
    u32_t capacity_;

    using word_ref_t = cuda_atomic_ref<u32_t>;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }

    /** Warp-ballot mask of the lanes whose index is a valid neighbor slot - a collective read
     *  reports the sentinel on the lanes outside it. */
    __device__ __forceinline__ unsigned lane_mask() const noexcept {
        return capacity_ >= 32 ? 0xFFFFFFFFu : (1u << capacity_) - 1u;
    }

    /** Acquire read of slot @p index, pairing with the lock holder's release fence so an ID read
     *  here comes with its node's payload; slot 0 belongs to @c peek_first. */
    __device__ __forceinline__ dense_node_id_t load_slot(u32_t index) const noexcept {
        return word_ref_t(row_[index]).load(cuda::memory_order_acquire);
    }

    /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current lock
     *  holder. */
    __device__ __forceinline__ void store_slot(u32_t index, dense_node_id_t value) const noexcept {
        word_ref_t(row_[index]).store(value, cuda::memory_order_relaxed);
    }

    /** Acquire-load of the lock word: the sentinel means unpublished or locked, any other value is
     *  the first neighbor ID with the last unlocker's writes visible. */
    __device__ __forceinline__ dense_node_id_t peek_first() const noexcept {
        return word_ref_t(row_[0]).load(cuda::memory_order_acquire);
    }

    /** Bounded lock acquisition: exchanges the sentinel in with acquire order, up to
     *  @p max_attempts times with jittered backoff; an exchange that receives the sentinel
     *  back did @b not acquire. @return The displaced first-neighbor value, or the sentinel. */
    __device__ __forceinline__ dense_node_id_t lock_if_free(u32_t max_attempts,
                                                            splitmix64_t& random_generator) const noexcept {
        for (u32_t attempt = 0; attempt != max_attempts; ++attempt) {
            dense_node_id_t const first_value = word_ref_t(row_[0]).exchange( //
                dense_node_id_unknown_k, cuda::memory_order_acquire);
            if (first_value != dense_node_id_unknown_k) return first_value;
            __nanosleep(static_cast<unsigned>(lock_backoff_floor_ns_k + //
                                              (random_generator() & lock_backoff_jitter_mask_k)));
        }
        return dense_node_id_unknown_k;
    }

    /** Release-stores @p first_value into slot 0 - unlocking and simultaneously (re)publishing the
     *  edited neighbors to every future acquiring reader. */
    __device__ __forceinline__ void unlock(dense_node_id_t first_value) const noexcept {
        word_ref_t(row_[0]).store(first_value, cuda::memory_order_release);
    }

    /** Warp-collective read of the neighbors when unlocked: lane 0 acquires slot 0, the others
     *  their own; a sentinel on lane 0 means transiently locked or unpublished. A lock holder reads
     *  through @c locked_row instead. */
    __device__ __forceinline__ dense_node_id_t load_row(int lane) const noexcept {
        if (static_cast<u32_t>(lane) >= capacity_) return dense_node_id_unknown_k;
        return lane ? load_slot(lane) : peek_first();
    }

    /** Warp-collective read of the neighbors for the @b lock holder: lane 0 reports the displaced
     *  @p first_value, the others acquire their slots. */
    __device__ __forceinline__ dense_node_id_t locked_row(int lane, dense_node_id_t first_value) const noexcept {
        if (static_cast<u32_t>(lane) >= capacity_) return dense_node_id_unknown_k;
        return lane ? load_slot(lane) : first_value;
    }
};

/** Per-warp register telemetry, folded into the device counters once per node; 32-bit in registers,
 *  since every counter is bounded by hops times connectivity, widened at the flush. */
struct dense_cuda_telemetry_t {
    u32_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, reselections = 0, restarts = 0, locked_drops = 0;
    u32_t local_scored = 0, remote_scored = 0;
    u32_t tail_drops = 0, visited_saturations = 0;
    u32_t changed_nodes = 0;

    /** Folds the live counters into the device block; the search instantiation skips the lock and
     *  reselection atomics it never touches. */
    template <bool inserting_>
    __device__ __forceinline__ void flush(dense_cuda_telemetry_block_t* counters, int lane) noexcept {
        if (lane != 0) return;
        // Tallies alone: the drain after the launch publishes them, so every add is relaxed.
        auto const add = [](u64_t& counter, u32_t value) noexcept {
            cuda_atomic_ref<u64_t>(counter).fetch_add(value, cuda::memory_order_relaxed);
        };
        add(counters->hops, hops);
        add(counters->scored, scored);
        add(counters->restarts, restarts);
        add(counters->locked_drops, locked_drops);
        if (changed_nodes) add(counters->changed_nodes, changed_nodes);
        if (local_scored | remote_scored) {
            add(counters->local_scored, local_scored);
            add(counters->remote_scored, remote_scored);
        }
        if (tail_drops | visited_saturations) {
            add(counters->tail_drops, tail_drops);
            add(counters->visited_saturations, visited_saturations);
        }
        if constexpr (!inserting_) return;
        add(counters->lock_attempts, lock_attempts);
        add(counters->lock_skips, lock_skips);
        add(counters->reselections, reselections);
    }
};

/** One walking warp's state: the candidate pool, the random stream, and the running telemetry. */
template <typename frontier_>
struct dense_cuda_warp_context {
    frontier_ frontier;
    splitmix64_t random_generator;
    dense_cuda_telemetry_t telemetry;
};

/** One warp's staged batch: candidate IDs, then the scored nodes and native dots of up to 32 of them. */
template <typename dot_type_>
struct dense_cuda_gathered {

    /** Candidate node IDs, up to a full vector cache of them. */
    u32_t ids[dense_cuda_cache_rows_k];

    /** The scored nodes of one batch, by batch row. */
    dense_cuda_scored_node_t scored[32];

    /** The native dots of one batch, parallel to @c scored. */
    dot_type_ dots[32];
};

/** Graph products of one query row as a device fold: each scored batch appends its node IDs and
 *  native dots to the row until it fills, while @c scored keeps counting past it. */
template <typename dot_type_>
struct dense_cuda_multiply_fold {

    /** The row's node ID outputs. */
    dense_node_id_t* ids;

    /** The row's product outputs, positionally paired with @c ids. */
    dot_type_* products;

    /** The row's width. */
    u32_t wanted;

    /** Pairs written so far, identical on every lane. */
    u32_t written = 0;

    /** Pairs scored, written or not, identical on every lane. */
    u32_t scored = 0;

    __device__ __forceinline__ walk_demand_t absorb(dense_cuda_gathered<dot_type_> const& gathered,
                                                    u32_t gathered_count, int lane) noexcept {
        u32_t const writable = min_of<u32_t>(gathered_count, wanted - written);
        if (static_cast<u32_t>(lane) < writable) {
            ids[written + lane] = gathered.ids[lane];
            products[written + lane] = gathered.dots[lane];
        }
        written += writable;
        scored += gathered_count;
        return walk_demand_t::continue_k;
    }
};

/**
 *  @brief Scores the batch @p batch_ids points at, landing each row's scored node and native dot in
 *      @p gathered.
 *
 *  A sub-warp of @c lanes_per_row lanes owns one row, re-reading the query from shared memory each
 *  step, so no register budget caps the dimensions. Plain loads are safe: every staged ID came
 *  through an acquiring slot-0 read.
 *
 *  @return This lane's scored entry, or the sentinel on lanes past @p gathered_count.
 */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_multiprocessor_k>
__device__ __forceinline__ dense_cuda_scored_node_t dense_cuda_score_batch( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, u32_t const* batch_ids, u32_t gathered_count,
    dense_cuda_gathered<typename kit_::dot_t>& gathered, int lane) noexcept {

    using norms_t = typename kit_::norms_t;
    // 128 concurrent random gathers saturate the HBM path, so the window is 128 over the resident warps.
    constexpr u32_t gathers_in_flight_k = min_of(dense_cuda_max_gathers_in_flight_k,
                                                 128u / (dense_cuda_warps_per_block_k * resident_blocks_));
    u32_t const lanes_log2 = 31 - __clz(planes.lanes_per_row);
    u32_t const rows_per_gather = 32u >> lanes_log2;
    u32_t const lane_in_row = static_cast<u32_t>(lane) & (planes.lanes_per_row - 1);
    u32_t const subwarp = static_cast<u32_t>(lane) >> lanes_log2;
    bool const leader = lane_in_row == 0;

    // Row groups are independent; tiling them keeps at most `gathers_in_flight_k` loads live per lane.
#pragma unroll 1
    for (u32_t group_base = 0; group_base < planes.lanes_per_row; group_base += gathers_in_flight_k) {
        norms_t batch_norms[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (leader && group_base + slot < planes.lanes_per_row && row < gathered_count)
                batch_norms[slot] = planes.norm_for(batch_ids[row]);
        }
        typename kit_::dot_t accumulators[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) accumulators[slot] = {};
#pragma unroll 1
        for (u32_t step = 0; step != planes.steps_per_row; ++step) {
            u32_t const fragment_index = lane_in_row + (step << lanes_log2);
            bool const in_row = fragment_index < planes.row_fragments;
            typename kit_::fragment_t const query_value = in_row ? query_span[fragment_index]
                                                                 : typename kit_::fragment_t {};
            typename kit_::fragment_t loaded[gathers_in_flight_k];
#pragma unroll
            for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
                u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
                if (in_row && group_base + slot < planes.lanes_per_row && row < gathered_count)
                    loaded[slot] = planes.vector_for(batch_ids[row], fragment_index);
            }
#pragma unroll
            for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
                u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
                if (in_row && group_base + slot < planes.lanes_per_row && row < gathered_count)
                    accumulators[slot] = kit_::accumulate_dot(loaded[slot], query_value, accumulators[slot]);
            }
        }
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            typename kit_::dot_t const dot = dense_cuda_reduce(accumulators[slot], planes.lanes_per_row);
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (leader && group_base + slot < planes.lanes_per_row && row < gathered_count) {
                dense_distance_t const distance = kit_::distance(dot, query_state, batch_norms[slot]);
                gathered.scored[row] = dense_cuda_scored_node_t::make(distance, batch_ids[row]);
                gathered.dots[row] = dot;
            }
        }
    }
    __syncwarp();
    return lane < static_cast<int>(gathered_count) ? gathered.scored[lane] : dense_cuda_scored_node_t::sentinel();
}

/** The plain walk's rescoring: none, so every batch lands as the kit scored it. */
struct dense_cuda_no_rescoring {

    __device__ __forceinline__ dense_cuda_no_rescoring for_query(u32_t) const noexcept { return {}; }
};

/** Indexer heads as a device rescoring: each scored dot becomes `Σₕ wₕ·ReLU(qₕ·k)` and its distance
 *  `1 − score`, the warp scoring the batch once per head exactly as it scored the navigating query. */
template <dense_cuda_kit kit_>
struct dense_cuda_indexer_rescoring {
    using fragment_t = typename kit_::fragment_t;

    /** Every query's heads as staged rows, @b [queries·heads,row_fragments], device-resident. */
    fragment_t const* heads;

    /** The signed head weights, @b [queries,heads], device-resident. */
    f32_t const* weights;

    /** Heads per query. */
    u32_t heads_count;

    /** Fragments per staged head row. */
    u32_t row_fragments;

    __device__ __forceinline__ dense_cuda_indexer_rescoring for_query(u32_t query_index) const noexcept {
        dense_cuda_indexer_rescoring narrowed = *this;
        narrowed.heads += static_cast<std::size_t>(query_index) * heads_count * row_fragments;
        narrowed.weights += static_cast<std::size_t>(query_index) * heads_count;
        return narrowed;
    }

    __device__ __forceinline__ dense_cuda_scored_node_t rescore(dense_cuda_planes_view<kit_> const& planes,
                                                                dense_cuda_gathered<typename kit_::dot_t>& gathered,
                                                                u32_t gathered_count, int lane,
                                                                dense_cuda_scored_node_t scored) const noexcept {
        bool const live = static_cast<u32_t>(lane) < gathered_count;
        f32_t indexed = 0;
        for (u32_t head = 0; head != heads_count; ++head) {
            dense_cuda_score_batch<kit_>(planes, heads + head * row_fragments, typename kit_::norms_t {}, gathered.ids,
                                         gathered_count, gathered, lane);
            if (live) indexed += weights[head] * fmaxf(static_cast<f32_t>(gathered.dots[lane]), 0.f);
            // The next head's scoring overwrites the dots, so every lane reads its own first.
            __syncwarp();
        }
        if (!live) return scored;
        gathered.dots[lane] = indexed;
        scored = dense_cuda_scored_node_t::make(1 - indexed, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/** Scores one gathered batch against the query, lets @p rescoring rewrite the scores, folds it into the pool, and
 *  hands it to @p fold - whose answer says whether the walk goes on. */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_multiprocessor_k,
          typename frontier_, typename fold_, typename rescoring_ = dense_cuda_no_rescoring>
__device__ __forceinline__ walk_demand_t dense_cuda_score_into_frontier( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, dense_cuda_gathered<typename kit_::dot_t>& gathered, u32_t gathered_count,
    dense_cuda_warp_context<frontier_>& context, int lane, fold_&& fold,
    rescoring_ const& rescoring = rescoring_ {}) noexcept {
    context.telemetry.scored += gathered_count;
    if (planes.resident_end) {
        bool const counted = static_cast<u32_t>(lane) < gathered_count;
        u32_t const candidate = counted ? gathered.ids[lane] : 0;
        unsigned const local_mask = __ballot_sync(
            0xFFFFFFFFu, counted && candidate >= planes.resident_begin && candidate < planes.resident_end);
        context.telemetry.local_scored += __popc(local_mask);
        context.telemetry.remote_scored += gathered_count - __popc(local_mask);
    }
    dense_cuda_scored_node_t staged = dense_cuda_score_batch<kit_, resident_blocks_>( //
        planes, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
    // Even a call that changes nothing reorders the register-capped wide walk, so the plain walk makes none.
    if constexpr (!std::is_same_v<rescoring_, dense_cuda_no_rescoring>)
        staged = rescoring.rescore(planes, gathered, gathered_count, lane, staged);
    // No dedup: the exact visited set upstream guarantees every ID stages once.
    context.frontier.push_batch(staged, lane);
    return fold.absorb(gathered, gathered_count, lane);
}

/**
 *  @brief Samples one 32-wide batch of distinct, published landmarks below @p sampling_bound
 *      into @p candidate_ids - uniform draws gated on the slot-0 publication word, probing
 *      forward on unpublished hits so out-of-order publication only costs retries.
 *  @return The number of staged landmarks; zero when the visited set saturated the bound.
 */
template <dense_cuda_kit kit_>
__device__ __forceinline__ u32_t dense_cuda_sample_landmarks( //
    dense_cuda_planes_view<kit_> const& planes, dense_node_id_t sampling_bound, dense_cuda_visited_set_t& visited,
    splitmix64_t& random_generator, u32_t (&candidate_ids)[dense_cuda_cache_rows_k], int lane) noexcept {

    dense_node_id_t draw = uniform_below<dense_node_id_t> {sampling_bound}(random_generator);
    bool published = false;
    for (int attempt = 0; attempt != 4 && !published; ++attempt) {
        dense_cuda_neighbors_ref_t const row {planes.neighbors_for(draw), planes.connectivity};
        published = row.peek_first() != dense_node_id_unknown_k;
        if (!published) draw = draw + 1 >= sampling_bound ? 0 : draw + 1; // >=: a zero bound stays at row 0.
    }
    bool const fresh = visited.test_and_set(draw, lane, published);
    unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
    u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
    if (fresh) candidate_ids[ordinal] = draw;
    __syncwarp();
    return __popc(fresh_mask);
}

/** Continues a greedy best-first walk toward the query over whatever the frontier holds - one warp,
 *  one traversal; pre-staged entries compete with the landmark seeds drawn here and must already be
 *  visited. Read-only on the graph: no locks taken, transiently locked candidates dropped after a
 *  bounded retry. */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_multiprocessor_k,
          typename frontier_, typename fold_ = dummy_fold_t, typename rescoring_ = dense_cuda_no_rescoring>
__device__ __forceinline__ void dense_cuda_search_ahead( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, dense_node_id_t sampling_bound, dense_cuda_visited_set_t& visited,
    dense_cuda_gathered<typename kit_::dot_t>& gathered, dense_cuda_warp_context<frontier_>& context, int lane,
    fold_&& fold = fold_ {}, rescoring_ const& rescoring = rescoring_ {}) noexcept {

    constexpr u32_t restarts_cap_k = 8;
    frontier_& frontier = context.frontier;
    dense_cuda_telemetry_t& telemetry = context.telemetry;
    u32_t const hops_cap = frontier.hops_budget();

    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = dense_cuda_sample_landmarks<kit_>( //
                planes, sampling_bound, visited, context.random_generator, gathered.ids, lane);
            if (gathered_count && dense_cuda_score_into_frontier<kit_, resident_blocks_>( //
                                      planes, query_span, query_state, gathered, gathered_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                return;
            seeding = false;
        }
        else {
            dense_cuda_scored_node_t const candidate = frontier.pop_nearest(lane);
            if (candidate == dense_cuda_scored_node_t::sentinel()) {
                // A drained frontier before an exhausted walk means nothing reachable from its seeds remains.
                if (frontier.exhausted() || telemetry.restarts >= restarts_cap_k || telemetry.hops >= hops_cap) break;
                seeding = true;
                ++telemetry.restarts;
                continue;
            }
            if (frontier.beyond_radius(candidate)) break;
            if (telemetry.hops >= hops_cap) break;
            ++telemetry.hops;

            // Only the leader polls the racing lock word: per-lane polls could disagree on one
            // unlock, and a value-dependent `continue` would fork the warp across iterations.
            dense_cuda_neighbors_ref_t const row {planes.neighbors_for(candidate.id()), planes.connectivity};
            dense_node_id_t first_value = dense_node_id_unknown_k;
            if (lane == 0) {
                first_value = row.peek_first();
                for (u32_t attempt = 0; attempt != traversal_peek_retries_k && first_value == dense_node_id_unknown_k;
                     ++attempt) {
                    __nanosleep(512);
                    first_value = row.peek_first();
                }
            }
            first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
            // The shuffle moves the word, not the leader's acquire; `bar.warp.sync` orders the
            // other lanes' slot loads after it.
            __syncwarp();
            if (first_value == dense_node_id_unknown_k) {
                ++telemetry.locked_drops;
                continue;
            }
            // The first warp-width of slots is loop-free, which keeps the hot path compact.
            dense_node_id_t const neighbor_id = lane == 0                                   ? first_value
                                                : static_cast<u32_t>(lane) < row.capacity() ? row.load_slot(lane)
                                                                                            : dense_node_id_unknown_k;
            bool const usable = neighbor_id != dense_node_id_unknown_k && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(neighbor_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) gathered.ids[ordinal] = neighbor_id;
            __syncwarp();
            u32_t const gathered_count = __popc(fresh_mask);
            if (gathered_count && dense_cuda_score_into_frontier<kit_, resident_blocks_>( //
                                      planes, query_span, query_state, gathered, gathered_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                return;
            // Wider connectivities continue a warp-width at a time; the bound is warp-uniform for the collectives.
            for (u32_t slot_base = 32; slot_base < planes.connectivity; slot_base += 32) {
                u32_t const slot = slot_base + lane;
                dense_node_id_t const wide_id = slot < planes.connectivity ? row.load_slot(slot)
                                                                           : dense_node_id_unknown_k;
                bool const wide_usable = wide_id != dense_node_id_unknown_k && wide_id != candidate.id();
                bool const wide_fresh = visited.test_and_set(wide_id, lane, wide_usable);
                unsigned const wide_mask = __ballot_sync(0xFFFFFFFFu, wide_fresh);
                u32_t const wide_ordinal = __popc(wide_mask & ((1u << lane) - 1));
                if (wide_fresh) gathered.ids[wide_ordinal] = wide_id;
                __syncwarp();
                u32_t const wide_count = __popc(wide_mask);
                if (wide_count && dense_cuda_score_into_frontier<kit_, resident_blocks_>( //
                                      planes, query_span, query_state, gathered, wide_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                    return;
            }
        }
    }
}

/** Greedy best-first walk from random published landmarks toward the query: a fresh frontier
 *  handed to @ref dense_cuda_search_ahead. */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_multiprocessor_k,
          typename frontier_, typename fold_ = dummy_fold_t, typename rescoring_ = dense_cuda_no_rescoring>
__device__ __forceinline__ void dense_cuda_search_around( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, dense_node_id_t sampling_bound, dense_cuda_visited_set_t& visited,
    dense_cuda_gathered<typename kit_::dot_t>& gathered, dense_cuda_warp_context<frontier_>& context, int lane,
    fold_&& fold = fold_ {}, rescoring_ const& rescoring = rescoring_ {}) noexcept {
    context.frontier.reset(lane);
    dense_cuda_search_ahead<kit_, resident_blocks_>( //
        planes, query_span, query_state, sampling_bound, visited, gathered, context, lane, fold, rescoring);
}

/** Runtime-strided view over the warp's shared vector cache - rows of @c row_fragments_ fragments
 *  each, aliasing the scratch slice. */
template <typename fragment_type_>
struct dense_cuda_vector_cache {
    fragment_type_* data_;
    u32_t row_fragments_;

    __device__ __forceinline__ fragment_type_* row(u32_t row_index) const noexcept {
        return data_ + static_cast<std::size_t>(row_index) * row_fragments_;
    }
    __device__ __forceinline__ fragment_type_& at(u32_t row_index, u32_t fragment_index) const noexcept {
        return row(row_index)[fragment_index];
    }
};

/** Warp-cooperative ascending buffer of scored nodes over a shared-memory span: chunks arrive
 *  pre-sorted from the 32-wide bitonic network and merge in place, entries past @c capacity_ drop. */
struct dense_cuda_sorted_buffer_t {
    dense_cuda_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_;

    __device__ __forceinline__ dense_cuda_scored_node_t at(u32_t rank) const noexcept { return entries_[rank]; }

    __device__ __forceinline__ void insert_batch(dense_cuda_scored_node_t value, u32_t count, int lane) noexcept {
        constexpr u32_t span_batches_k = divide_round_up(dense_cuda_cache_rows_k, 32u);
        bool const incoming = static_cast<u32_t>(lane) < count;
        u32_t incoming_rank = incoming ? static_cast<u32_t>(lane) : 0;
        for (u32_t i = 0; i != size_; ++i) // Warp-uniform bound: every lane walks the span.
            incoming_rank += incoming && entries_[i].distance_bits() <= value.distance_bits();
        // Existing entries shift by the number of incoming values strictly ahead of them;
        // ties resolve the other way than above, so no two final positions collide.
        dense_cuda_scored_node_t moved[span_batches_k];
        u32_t moved_to[span_batches_k];
        u32_t batch = 0;
        for (u32_t base = 0; base < size_; base += 32, ++batch) {
            u32_t const index = base + lane;
            bool const active = index < size_;
            moved[batch] = active ? entries_[index] : dense_cuda_scored_node_t::sentinel();
            u32_t shift = 0;
            for (u32_t source = 0; source != count; ++source) // Warp-uniform: shuffles inside.
                shift += active &&
                         value.shuffle_from(static_cast<int>(source)).distance_bits() < moved[batch].distance_bits();
            moved_to[batch] = index + shift;
        }
        __syncwarp(); // Every displaced entry is staged in registers before any write lands.
        batch = 0;
        for (u32_t base = 0; base < size_; base += 32, ++batch)
            if (base + lane < size_ && moved_to[batch] < capacity_) entries_[moved_to[batch]] = moved[batch];
        if (incoming && incoming_rank < capacity_) entries_[incoming_rank] = value;
        __syncwarp();
        size_ = min_of(size_ + count, capacity_);
    }
};

/** Shared-memory spans backing one warp's neighbor re-selection, carved after the vector cache:
 *  the selected target IDs, the ascending per-candidate scores, and the kept cache positions. */
struct dense_cuda_selection_scratch_t {
    u32_t* selected_ids;
    dense_cuda_scored_node_t* scored_entries;
    u32_t* kept_positions;

    /** Carves the three spans past the vector cache's @p cache_bytes: the ascending candidate
     *  scores first, since they carry the widest alignment, then the two ID spans. */
    static __device__ __forceinline__ dense_cuda_selection_scratch_t carve(byte_t* scratch, u32_t connectivity,
                                                                           std::size_t cache_bytes) noexcept {
        u32_t const slots = min_of(connectivity, static_cast<u32_t>(dense_cuda_cache_rows_k));
        byte_t* const base = scratch + cache_bytes;
        return {reinterpret_cast<u32_t*>(base + sizeof(dense_cuda_scored_node_t) * slots),
                reinterpret_cast<dense_cuda_scored_node_t*>(base),
                reinterpret_cast<u32_t*>(base + (sizeof(dense_cuda_scored_node_t) + sizeof(u32_t)) * slots)};
    }
};

/**
 *  @brief Greedy diversity selection over the ascending pool prefix: a candidate is kept only
 *      when it is closer to the query than to every kept neighbor, so neighbors spread across
 *      directions; the kept neighbors are ascending by construction.
 *  @return The kept count; @p selected_ids holds that many node IDs on return.
 */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ u32_t dense_cuda_select_diverse( //
    dense_cuda_planes_view<kit_> const& planes, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], u32_t (&candidate_ids)[dense_cuda_cache_rows_k],
    u32_t* selected_ids, dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    frontier_& frontier = context.frontier;

    // Stage the ascending prefix IDs & cache their vectors and norms.
    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < dense_cuda_cache_rows_k; base += 32) {
        dense_cuda_scored_node_t const entry = base + lane < 64 ? frontier.rank_entry(base + lane)
                                                                : dense_cuda_scored_node_t::sentinel();
        unsigned const live = __ballot_sync(0xFFFFFFFFu, !(entry == dense_cuda_scored_node_t::sentinel()));
        if (!(entry == dense_cuda_scored_node_t::sentinel())) candidate_ids[base + lane] = entry.id();
        candidates_count += __popc(live);
        if (live != 0xFFFFFFFFu) break;
    }
    __syncwarp();
    for (u32_t word = lane; word < candidates_count * planes.row_fragments; word += 32) {
        u32_t const row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, row);
        __pipeline_memcpy_async(&cache.at(row, fragment), planes.vector_span_for(candidate_ids[row]) + fragment,
                                sizeof(typename kit_::fragment_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    for (u32_t row = lane; row < candidates_count; row += 32) cache_norms[row] = planes.norm_for(candidate_ids[row]);
    __syncwarp();

    u32_t selected_count = 0;
    for (u32_t candidate_index = 0; candidate_index < candidates_count && selected_count < planes.connectivity;
         ++candidate_index) {
        f32_t const to_query = frontier.rank_entry(candidate_index).distance();
        bool diverse = true;
        for (u32_t base = 0; base < selected_count && diverse; base += 32) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < selected_count) {
                u32_t const kept_position = selected_ids[base + lane];
                typename kit_::dot_t accumulator {};
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    accumulator = kit_::accumulate_dot(cache.at(candidate_index, fragment),
                                                       cache.at(kept_position, fragment), accumulator);
                f32_t const between = kit_::distance(accumulator, cache_norms[candidate_index],
                                                     cache_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++context.telemetry.scored;
            }
            diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
        }
        if (diverse) {
            if (lane == 0) selected_ids[selected_count] = candidate_index;
            ++selected_count;
            __syncwarp();
        }
    }
    // The slots diversity leaves empty take the nearest rejected candidates, as the CPU's `select_diverse` does:
    // on a dot graph, whose hubs reject one another, a row of diverse picks alone is a handful of links.
    u32_t const diverse_count = selected_count;
    for (u32_t candidate_index = 0, matched = 0;
         candidate_index < candidates_count && selected_count < planes.connectivity; ++candidate_index) {
        if (matched != diverse_count && selected_ids[matched] == candidate_index) {
            ++matched;
            continue;
        }
        if (lane == 0) selected_ids[selected_count] = candidate_index;
        ++selected_count;
        __syncwarp();
    }
    // Rewrite the kept positions as node IDs in place - the caller links to nodes, not cache rows.
    __syncwarp();
    for (u32_t slot = lane; slot < selected_count; slot += 32) selected_ids[slot] = candidate_ids[selected_ids[slot]];
    __syncwarp();
    return selected_count;
}

/** Back-links the node: locks each selected neighbor's slots through the bounded slot-0 exchange,
 *  appends when there is room, and otherwise re-runs the diversity heuristic over every neighbor
 *  plus the newcomer under the lock. At most one lock is held at a time, only after this node's own
 *  publication, and a lock busy past the budget skips that neighbor. */
template <dense_cuda_kit kit_>
__device__ __forceinline__ void dense_cuda_back_link_neighbors( //
    dense_cuda_planes_view<kit_> const& planes, dense_node_id_t new_node_id, typename kit_::norms_t new_norms,
    dense_cuda_selection_scratch_t selection, u32_t selected_count,
    dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], dense_cuda_gathered<typename kit_::dot_t>& gathered,
    splitmix64_t& random_generator, dense_cuda_telemetry_t& telemetry, int lane) noexcept {

    // The last two cache rows host the newcomer and the target, whose vector the scored batch reads as its query.
    constexpr u32_t newcomer_row_k = dense_cuda_cache_rows_k - 1;
    constexpr u32_t target_row_k = dense_cuda_cache_rows_k - 2;
    // Those two rows also bound how many neighbors one warp can re-select; wider connectivities keep sentinel tails.
    u32_t const connectivity = min_of(planes.connectivity, dense_cuda_cache_rows_k - 2u);

    for (u32_t selected_index = 0; selected_index < selected_count; ++selected_index) {
        dense_node_id_t const target = selection.selected_ids[selected_index];
        dense_cuda_neighbors_ref_t const row {planes.neighbors_for(target), connectivity};
        dense_node_id_t first_value = dense_node_id_unknown_k;
        if (lane == 0) first_value = row.lock_if_free(back_link_lock_attempts_k, random_generator);
        first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
        __syncwarp(); // The sweep below must follow the leader's acquiring exchange on every lane.
        ++telemetry.lock_attempts;
        if (first_value == dense_node_id_unknown_k) { // Bounded-wait policy: skip the neighbor.
            ++telemetry.lock_skips;
            continue;
        }
        if (first_value == target) { // Published-empty self-loop: the newcomer is the first neighbor.
            if (lane == 0) row.unlock(new_node_id);
            continue;
        }

        // Read every locked neighbor a warp-width at a time, remembering the first empty slot and
        // staging every live neighbor ID for the potential re-selection.
        u32_t empty_slot = connectivity;
        for (u32_t base = 0; base < connectivity; base += 32) {
            u32_t const slot = base + lane;
            dense_node_id_t const slot_value = slot == 0             ? first_value
                                               : slot < connectivity ? row.load_slot(slot)
                                                                     : dense_node_id_unknown_k;
            if (slot < connectivity) gathered.ids[slot] = slot_value;
            unsigned const empty_mask = __ballot_sync(0xFFFFFFFFu,
                                                      slot < connectivity && slot_value == dense_node_id_unknown_k);
            if (empty_mask && empty_slot == connectivity) empty_slot = base + __ffs(empty_mask) - 1;
        }
        __syncwarp();
        // Every slot store below is relaxed; this fence heads each lane's stores, so an acquiring
        // reader of any slot sees the newcomer's payload.
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
        if (empty_slot != connectivity) { // Fast path: append into that slot, unlock unchanged.
            if (lane == 0) row.store_slot(empty_slot, new_node_id);
            __syncwarp();
            if (lane == 0) row.unlock(first_value);
            continue;
        }

        // Every slot taken: re-select diversely over the neighbors plus the newcomer, ascending by
        // distance to the target itself.
        ++telemetry.reselections;
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            cache.at(target_row_k, word) = planes.vector_for(target, word);
        __syncwarp();
        typename kit_::norms_t const target_state = planes.norm_for(target);

        // The newcomer's distance to the target, from its freshly-owned vector row.
        typename kit_::dot_t accumulator {};
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            accumulator = kit_::accumulate_dot(planes.vector_for(new_node_id, word), cache.at(target_row_k, word),
                                               accumulator);
        typename kit_::dot_t const self_dot = dense_cuda_reduce(accumulator, 32u);
        f32_t const newcomer_distance = kit_::distance(__shfl_sync(0xFFFFFFFFu, self_dot, 0), target_state, new_norms);
        telemetry.scored += connectivity + 1;

        // Score the neighbors a warp-width at a time, pre-sort each batch bitonically, and merge
        // them into the ascending sorted buffer.
        dense_cuda_sorted_buffer_t sorted_candidates {selection.scored_entries, 0, connectivity};
        for (u32_t base = 0; base < connectivity; base += 32) {
            u32_t const batch_count = min_of(connectivity - base, 32u);
            dense_cuda_scored_node_t const scored = dense_cuda_score_batch<kit_>( //
                planes, cache.row(target_row_k), target_state, gathered.ids + base, batch_count, gathered, lane);
            dense_cuda_scored_node_t const sorted_batch = dense_cuda_sort32(
                static_cast<u32_t>(lane) < batch_count
                    ? dense_cuda_scored_node_t::make(scored.distance(), base + lane) // ID = slot index.
                    : dense_cuda_scored_node_t::sentinel(),
                lane);
            sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
        }

        // Cache the neighbors' vectors plus the newcomer at the tail row through `cp.async`, which
        // holds no registers, then merge-walk them ascending by distance to the target.
        u32_t const link_words = connectivity * planes.row_fragments;
        for (u32_t word = lane; word < link_words; word += 32) {
            u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
            __pipeline_memcpy_async(&cache.at(cache_row, fragment),
                                    planes.vector_span_for(gathered.ids[cache_row]) + fragment,
                                    sizeof(typename kit_::fragment_t));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            cache.at(newcomer_row_k, word) = planes.vector_for(new_node_id, word);
        for (u32_t cache_row = lane; cache_row < connectivity; cache_row += 32)
            cache_norms[cache_row] = planes.norm_for(gathered.ids[cache_row]);
        if (lane == 0) cache_norms[newcomer_row_k] = new_norms;
        __syncwarp();

        // The merge walk: the sorted neighbors and the newcomer, ascending by distance to the target, as cache rows.
        constexpr u32_t walked_out_k = ~0u;
        u32_t existing_index = 0;
        bool newcomer_pending = true;
        f32_t candidate_distance = 0;
        auto next_candidate = [&]() noexcept {
            dense_cuda_scored_node_t const next_link = existing_index < sorted_candidates.size_
                                                           ? sorted_candidates.at(existing_index)
                                                           : dense_cuda_scored_node_t::sentinel();
            bool const take_newcomer = newcomer_pending && newcomer_distance < next_link.distance();
            if (!take_newcomer && existing_index >= sorted_candidates.size_) return walked_out_k;
            candidate_distance = take_newcomer ? newcomer_distance : next_link.distance();
            if (take_newcomer) newcomer_pending = false;
            else ++existing_index;
            return take_newcomer ? newcomer_row_k : next_link.id();
        };

        u32_t refined_count = 0;
        while (refined_count != connectivity) {
            u32_t const candidate_position = next_candidate();
            if (candidate_position == walked_out_k) break;

            bool diverse = true;
            for (u32_t base = 0; base < refined_count && diverse; base += 32) { // Warp-uniform bounds.
                bool lane_diverse = true;
                if (base + lane < refined_count) {
                    u32_t const kept_position = selection.kept_positions[base + lane];
                    typename kit_::dot_t pair_accumulator {};
                    for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                        pair_accumulator = kit_::accumulate_dot(cache.at(candidate_position, fragment),
                                                                cache.at(kept_position, fragment), pair_accumulator);
                    f32_t const between = kit_::distance(pair_accumulator, cache_norms[candidate_position],
                                                         cache_norms[kept_position]);
                    lane_diverse = between >= candidate_distance;
                    ++telemetry.scored;
                }
                diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
            }
            if (diverse) {
                if (lane == 0) selection.kept_positions[refined_count] = candidate_position;
                ++refined_count;
                __syncwarp();
            }
        }

        // The slots diversity leaves empty take the nearest rejected candidates, as the CPU's and the refine merge's
        // do: a row that only shrinks drops links its neighbors may be reached through, and on a dot graph, whose
        // hubs reject one another, it strands half the nodes. The kept prefix is a subsequence of the same walk.
        u32_t const diverse_count = refined_count;
        existing_index = 0, newcomer_pending = true;
        for (u32_t matched = 0; refined_count != connectivity;) {
            u32_t const candidate_position = next_candidate();
            if (candidate_position == walked_out_k) break;
            if (matched != diverse_count && selection.kept_positions[matched] == candidate_position) {
                ++matched;
                continue;
            }
            if (lane == 0) selection.kept_positions[refined_count] = candidate_position;
            ++refined_count;
            __syncwarp();
        }

        // Rewrite the neighbors: tail slots relaxed, shrink sentinel-terminated, head unlocks.
        __syncwarp();
        for (u32_t slot = lane; slot < connectivity; slot += 32) {
            if (slot == 0) continue;
            dense_node_id_t linked = dense_node_id_unknown_k;
            if (slot < refined_count) {
                u32_t const kept_position = selection.kept_positions[slot];
                linked = kept_position == newcomer_row_k ? new_node_id : gathered.ids[kept_position];
            }
            row.store_slot(slot, linked);
        }
        __syncwarp();
        if (lane == 0) {
            u32_t const head_position = selection.kept_positions[0];
            row.unlock(head_position == newcomer_row_k ? new_node_id : gathered.ids[head_position]);
        }
    }
}

/** Layout of one warp's slice of the traversal arena, computed on the host and carried by the
 *  views: the exact visited set, then the wide frontier's spill tail when it has one. */
struct dense_cuda_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    constexpr std::size_t slice_bytes() const noexcept {
        return static_cast<std::size_t>(visited_slots) * sizeof(u64_t) +
               static_cast<std::size_t>(tail_entries) * sizeof(dense_cuda_scored_node_t);
    }
    __device__ __forceinline__ u64_t* visited_span(byte_t* base, std::size_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(base + warp_index * slice_bytes());
    }
    __device__ __forceinline__ dense_cuda_scored_node_t* tail_span(byte_t* base,
                                                                   std::size_t warp_index) const noexcept {
        return reinterpret_cast<dense_cuda_scored_node_t*>( //
            base + warp_index * slice_bytes() + static_cast<std::size_t>(visited_slots) * sizeof(u64_t));
    }
};

/** One warp's slice of a launch: its coordinates, its dynamic-shared scratch, and its arena
 *  visited span - the shared prologue of every traversal kernel. */
struct dense_cuda_warp_slice_t {
    int lane, warp;
    std::size_t warp_index;
    byte_t* scratch;
    u64_t* visited_slots;
};

/** Carves the calling warp's slice out of @p view and the block's @p dynamic_scratch. */
template <typename view_>
__device__ __forceinline__ dense_cuda_warp_slice_t dense_cuda_carve_slice(view_ const& view,
                                                                          byte_t* dynamic_scratch) noexcept {
    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    std::size_t const warp_index = blockIdx.x * static_cast<std::size_t>(dense_cuda_warps_per_block_k) + warp;
    byte_t* const scratch = dynamic_scratch + static_cast<std::size_t>(warp) * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch, view.arena_shape.visited_span(view.arena, warp_index)};
}

/** Geometry & plane pointers one insertion launch operates on. Every plane row an assignable
 *  new_node_id can touch is mapped and sentinel-filled ahead of the launch. */
template <dense_cuda_kit kit_>
struct dense_cuda_build_view {
    dense_cuda_planes_view<kit_> planes;

    /** Incoming vectors, @b [rows,row_fragments] in assigned-ID order, device-resident. */
    typename kit_::fragment_t const* staging;

    /** IDs strictly below this bound may be claimed & inserted by this launch. */
    u32_t end_id;

    /** Bytes of the per-warp dynamic-shared slice: the query span, grown to fit the candidate
     *  cache and the re-selection spans. */
    u32_t scratch_bytes_per_warp;

    /** The batch's base ID: staging row 0 lands there, and the warps assign themselves offsets
     *  within @b [first_id,end_id). */
    dense_node_id_t first_id;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice - the exact visited set. */
    dense_cuda_arena_shape_t arena_shape;

    /** The context's epoch floor at launch: batch-local epochs ride above it, so stale arena slots
     *  compare older and a warm arena is never re-zeroed. */
    u32_t epoch_base;

    /** The index's published-row count, advanced at each slot-0 release and polled by the admission
     *  ramp; it implies nothing about @b which rows. */
    u32_t* published_rows;
    dense_cuda_telemetry_block_t* telemetry;

    /** Rows of the batch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;
};

/** Inputs & outputs of one search launch: a batch of queries in, the nearest IDs and - when
 *  requested - their distances out, both ascending per query. */
template <dense_cuda_kit kit_>
struct dense_cuda_search_view {
    dense_cuda_planes_view<kit_> planes;

    /** The query rows, @b [queries,row_fragments], device-resident. */
    typename kit_::fragment_t const* queries;

    u32_t queries_count;

    /** Published-node ceiling for landmark sampling, read off @c size() at launch; the walk itself
     *  gates on slot-0 publication words. */
    u32_t sampling_bound;

    /** Bytes of the per-warp dynamic-shared slice - the query span. */
    u32_t scratch_bytes_per_warp;

    /** Requested expansion; past the register ladder it is the wide walk's budget and sizes the
     *  arena's spill tail. */
    u32_t expansion;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice: the exact visited set, then the wide frontier's spill tail. */
    dense_cuda_arena_shape_t arena_shape;

    /** The context's epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    dense_cuda_telemetry_block_t* telemetry;
};

/** Default dynamic-shared opt-in threshold: launches asking for more must raise the kernel's
 *  attribute before the first launch. */
inline constexpr u32_t dense_cuda_static_scratch_limit_k = 48 * 1024;

/** Launches one traversal kernel with @p blocks of the standard warp complement on @p stream,
 *  raising the dynamic-shared ceiling first when the warp slices outgrow the static window. */
template <auto kernel_, typename view_type_, typename... extras_types_>
status_t dense_cuda_launch(view_type_ view, u32_t blocks, cudaStream_t stream, extras_types_... extras) noexcept {
    u32_t const scratch_bytes = dense_cuda_warps_per_block_k * view.scratch_bytes_per_warp;
    if (scratch_bytes > dense_cuda_static_scratch_limit_k &&
        cudaFuncSetAttribute(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch_bytes) != cudaSuccess)
        return status_t::device_error_k;
    return cuda_launch(kernel_, blocks, dense_cuda_warps_per_block_k * 32, scratch_bytes, stream, view, extras...);
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

/** Inserts the node identified by @p new_node_id: lands the payload and norms, walks the published
 *  graph, selects diverse neighbors, publishes with one release store, then back-links them - one
 *  warp per node. */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_cuda_insert_one( //
    dense_cuda_build_view<kit_> const& view, dense_node_id_t new_node_id, dense_cuda_visited_set_t& visited,
    typename kit_::fragment_t* query_span, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_selection_scratch_t selection, typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k],
    dense_cuda_gathered<typename kit_::dot_t>& gathered, dense_cuda_warp_context<frontier_>& context,
    int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;

    // One pass lands the payload, derives the norms, and parks the query in its shared span.
    typename kit_::fragment_t const* staging_row =
        view.staging + static_cast<std::size_t>(new_node_id - view.first_id) * planes.row_fragments;
    typename kit_::fragment_t* owned_row = planes.vector_span_for(new_node_id);
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32) {
        typename kit_::fragment_t const value = staging_row[fragment];
        owned_row[fragment] = value;
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = dense_cuda_reduce_norms(norms_accumulator);
    norms_accumulator = dense_cuda_broadcast_norms(norms_accumulator);
    typename kit_::norms_t const new_norms = kit_::partials_to_norms(norms_accumulator);
    if (lane == 0) planes.norm_ref_for(new_node_id) = new_norms;
    typename kit_::norms_t const query_state = new_norms;
    __syncwarp();

    // Walk the watermark - a virgin index publishes an empty node instead.
    if (new_node_id != 0)
        dense_cuda_search_around<kit_>( //
            planes, query_span, query_state, new_node_id, visited, gathered, context, lane);
    else context.frontier.reset(lane);

    // The own row needs no lock: no reader arrives before the slot-0 release below flips the sentinel.
    u32_t const selected_count = dense_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered.ids, selection.selected_ids, context, lane);
    dense_cuda_neighbors_ref_t const own_neighbors {planes.neighbors_for(new_node_id), planes.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += 32)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    __syncwarp();
    dense_node_id_t const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) {
        own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
        // A tally for the admission ramp; the slot-0 release above is what publishes the row.
        cuda_atomic_ref<u32_t>(*view.published_rows).fetch_add(1u, cuda::memory_order_relaxed);
    }

    // Only now - published, no locks held - back-link the neighbors.
    dense_cuda_back_link_neighbors<kit_>( //
        planes, new_node_id, new_norms, selection, selected_count, cache, cache_norms, gathered,
        context.random_generator, context.telemetry, lane);
}

/**
 *  @brief One warp inserts one node, and warps never synchronize beyond per-neighbor slot-0
 *      locks: no watermark, no publication order, no compare-and-swap.
 *
 *  @section dense_vram_index_concurrency Concurrency & Publication
 *
 *  IDs are batch-local: the host claims a dense range, and each warp assigns itself the next
 *  offset within it, so sibling batches never cross-claim. A node is usable exactly when its
 *  slot 0 acquire-loads a non-sentinel; a self-loop there means published-empty, and every
 *  give-up path still publishes one, so an assigned ID never stalls anyone. Landmark sampling
 *  probes past unpublished hits, so out-of-order publication costs retries, never
 *  correctness. The admission ramp holds a node until @c published + max(32, published)
 *  covers its ID, bounding lock contention while the graph is small.
 */
template <dense_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_multiprocessor_k)
    dense_cuda_insert_kernel(dense_cuda_build_view<kit_> view) {
    // The dynamic slice hosts the query span while traversing, then the candidate cache while
    // selecting; the exact visited set lives in the warp's arena slice, not in shared memory.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_gathered<typename kit_::dot_t> gathered[dense_cuda_warps_per_block_k];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_cuda_selection_scratch_t const selection = dense_cuda_selection_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_cuda_cache_rows_k);
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

        dense_cuda_warp_context<dense_cuda_register_frontier<pool_slots_>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(new_node_id) * 33 + lane)};
        dense_cuda_visited_set_t visited {visited_slots, view.arena_shape.visited_hash_shift,
                                          view.epoch_base + (new_node_id - view.first_id) + 1};
        dense_cuda_insert_one<kit_>( //
            view, new_node_id, visited, query_span, cache, selection, cache_norms[warp], gathered[warp], context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.flush<true>(view.telemetry, lane);
    }
}

/** Geometry & plane pointers one refinement launch operates on: the live planes it reads, and the
 *  shadow neighbor plane it writes - same @b [rows,connectivity] shape, observed by no reader. */
template <dense_cuda_kit kit_>
struct dense_cuda_refine_view {
    dense_cuda_planes_view<kit_> planes;

    /** Refined neighbors land here, @b [rows,connectivity] IDs - plain stores, published wholesale
     *  by the host after the launch drains. */
    u32_t* shadow_neighbors;

    /** IDs strictly below this bound are refined by this launch. */
    u32_t end_id;

    /** Bytes of the per-warp dynamic-shared slice - the insertion path's shape. */
    u32_t scratch_bytes_per_warp;

    /** The pass's base ID: the warps assign themselves offsets in @b [first_id,end_id). */
    dense_node_id_t first_id;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice - the exact visited set. */
    dense_cuda_arena_shape_t arena_shape;

    /** The context's monotone epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    dense_cuda_telemetry_block_t* telemetry;

    /** Rows of the pass warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;

    /** Optional dense ID array to refine instead of the @b [first_id,end_id) range; null refines
     *  the range, and under an array @c end_id bounds its length and offsets index it. */
    u32_t const* node_ids = nullptr;

    /** Published-node ceiling for the walk's landmark sampling: always the graph size, never
     *  @c end_id, which under an ID array is its length. */
    u32_t sampling_bound = 0;
};

/** Re-selects the neighbors of @p node_id: the node's own vector is the query, its current
 *  neighbors seed the walk so every one already wired competes in the re-selection, and the diverse
 *  selection lands in the shadow slots. Read-only on the live planes. */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_cuda_refine_one( //
    dense_cuda_refine_view<kit_> const& view, dense_node_id_t node_id, dense_cuda_visited_set_t& visited,
    typename kit_::fragment_t* query_span, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_selection_scratch_t selection, typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k],
    dense_cuda_gathered<typename kit_::dot_t>& gathered, dense_cuda_warp_context<frontier_>& context,
    int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;

    // The query is the node's own stored vector; its norms are already derived state.
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
        query_span[fragment] = planes.vector_for(node_id, fragment);
    typename kit_::norms_t const query_state = planes.norm_for(node_id);
    __syncwarp();

    // The node itself never joins its own candidate pool - visited before anything stages.
    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane, true);

    // Seed the frontier with the current neighbors, staged like a traversal expansion minus the hop bookkeeping.
    dense_cuda_neighbors_ref_t const own_neighbors {planes.neighbors_for(node_id), planes.connectivity};
    dense_node_id_t first_value = dense_node_id_unknown_k;
    if (lane == 0) first_value = own_neighbors.peek_first();
    first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
    if (first_value != dense_node_id_unknown_k && first_value != node_id)
        for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) {
            u32_t const slot = slot_base + lane;
            dense_node_id_t const seed_id = slot == 0                    ? first_value
                                            : slot < planes.connectivity ? own_neighbors.load_slot(slot)
                                                                         : dense_node_id_unknown_k;
            bool const usable = seed_id != dense_node_id_unknown_k && seed_id != node_id;
            bool const fresh = visited.test_and_set(seed_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) gathered.ids[ordinal] = seed_id;
            __syncwarp();
            u32_t const gathered_count = __popc(fresh_mask);
            if (gathered_count)
                dense_cuda_score_into_frontier<kit_>( //
                    planes, query_span, query_state, gathered, gathered_count, context, lane, dummy_fold_t {});
        }

    // Walk the graph past the seeds, then re-select the neighbors from the whole pool.
    dense_cuda_search_ahead<kit_>( //
        planes, query_span, query_state, view.sampling_bound, visited, gathered, context, lane);
    u32_t const selected_count = dense_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered.ids, selection.selected_ids, context, lane);

    // The insertion's slot convention: neighbors ascending, sentinel tail, a slot-0 self-loop meaning published-empty.
    u32_t* const shadow_slots = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot = lane; slot < planes.connectivity; slot += 32)
        shadow_slots[slot] = slot < selected_count ? selection.selected_ids[slot]
                             : slot == 0           ? node_id
                                                   : dense_node_id_unknown_k;
    __syncwarp();
}

/** One warp refines one node, assigning itself node IDs from the pass counter; warps only read the
 *  live planes and write disjoint shadow neighbors. */
template <dense_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_multiprocessor_k)
    dense_cuda_refine_kernel(dense_cuda_refine_view<kit_> view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_gathered<typename kit_::dot_t> gathered[dense_cuda_warps_per_block_k];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_cuda_selection_scratch_t const selection = dense_cuda_selection_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_cuda_cache_rows_k);
    while (true) {
        u32_t offset;
        if (lane == 0) offset = cuda_atomic_ref<u32_t>(*view.assigned_rows).fetch_add(1u, cuda::memory_order_relaxed);
        offset = __shfl_sync(0xFFFFFFFFu, offset, 0);
        // Under an ID array, end_id bounds it; the assigned offset indexes it.
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_ids ? view.node_ids[offset] : view.first_id + offset;

        dense_cuda_warp_context<dense_cuda_register_frontier<pool_slots_>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(node_id) * 33 + lane)};
        dense_cuda_visited_set_t visited {visited_slots, view.arena_shape.visited_hash_shift,
                                          view.epoch_base + offset + 1};
        dense_cuda_refine_one<kit_>( //
            view, node_id, visited, query_span, cache, selection, cache_norms[warp], gathered[warp], context, lane);
        context.telemetry.tail_drops = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.flush<false>(view.telemetry, lane);
    }
}

/** Hands @p callback every neighbor of @p row the calling lane owns - its lane-strided slots, past sentinels,
 *  self-loops and IDs at or past @p rows_count. */
template <typename callback_type_>
__device__ __forceinline__ void dense_cuda_for_each_link(u32_t const* neighbors, u32_t connectivity, u32_t rows_count,
                                                         u32_t row, int lane, callback_type_&& callback) noexcept {
    u32_t const* const slots = neighbors + static_cast<std::size_t>(row) * connectivity;
    for (u32_t slot = static_cast<u32_t>(lane); slot < connectivity; slot += 32)
        if (u32_t const target = slots[slot]; target != dense_node_id_unknown_k && target != row && target < rows_count)
            callback(target);
}

/** Scatters the walked rows' shadow neighbors in reverse: every forward pick @b u → @b v lands @b u in @b v's
 *  reverse bucket, and marks both rows in @p touched - the rows the merge re-selects. The walked rows are the range
 *  from @p first_id, or a heal round's islands listed in @p node_ids, which mark only themselves and land in no bucket;
 *  appends past @p bucket_width are counted, never written. */
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
template <dense_cuda_kit kit_>
__device__ __forceinline__ void dense_cuda_merge_links_one( //
    dense_cuda_refine_view<kit_> const& view, u32_t const* bucket_ids, u32_t const* bucket_counts, u32_t bucket_width,
    dense_node_id_t node_id, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_scored_node_t* sorted_span, u32_t* kept_positions,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], dense_cuda_gathered<typename kit_::dot_t>& gathered,
    dense_cuda_telemetry_t& telemetry, int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;
    // The last cache row hosts the node's own vector, so only the rows before it can host candidates.
    constexpr u32_t query_row_k = dense_cuda_cache_rows_k - 1;
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
    dense_cuda_sorted_buffer_t sorted_candidates {sorted_span, 0, query_row_k};
    for (u32_t base = 0; base < candidates_count; base += 32) {
        u32_t const batch_count = min_of(candidates_count - base, 32u);
        dense_cuda_scored_node_t const scored = dense_cuda_score_batch<kit_>( //
            planes, cache.row(query_row_k), query_state, gathered.ids + base, batch_count, gathered, lane);
        dense_cuda_scored_node_t const sorted_batch = dense_cuda_sort32(
            static_cast<u32_t>(lane) < batch_count
                ? dense_cuda_scored_node_t::make(scored.distance(), base + lane) // ID = candidate index.
                : dense_cuda_scored_node_t::sentinel(),
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
        dense_cuda_scored_node_t const entry = sorted_candidates.at(merge_index);
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
                f32_t const between = kit_::distance(pair_accumulator, cache_norms[candidate_position],
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
    // subsequence of the sorted candidates, so one cursor tells a kept candidate from a rejected one.
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
template <dense_cuda_kit kit_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_multiprocessor_k)
    dense_cuda_merge_links_kernel(dense_cuda_refine_view<kit_> view, u32_t const* bucket_ids,
                                  u32_t const* bucket_counts, u32_t bucket_width, dense_cuda_bitset_t touched) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_gathered<typename kit_::dot_t> gathered[dense_cuda_warps_per_block_k];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {
        reinterpret_cast<typename kit_::fragment_t*>(slice.scratch), view.planes.row_fragments};
    // Past the candidate cache: the ascending scored span, then the kept-row indices.
    std::size_t const cache_bytes = static_cast<std::size_t>(view.planes.row_fragments) *
                                    sizeof(typename kit_::fragment_t) * dense_cuda_cache_rows_k;
    dense_cuda_scored_node_t* const sorted_span = reinterpret_cast<dense_cuda_scored_node_t*>(slice.scratch +
                                                                                              cache_bytes);
    u32_t* const kept_positions = reinterpret_cast<u32_t*>(slice.scratch + cache_bytes +
                                                           sizeof(dense_cuda_scored_node_t) * dense_cuda_cache_rows_k);
    while (true) {
        u32_t node_id;
        if (lane == 0)
            node_id = view.first_id +
                      cuda_atomic_ref<u32_t>(*view.assigned_rows).fetch_add(1u, cuda::memory_order_relaxed);
        node_id = __shfl_sync(0xFFFFFFFFu, node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the walk neither re-selected nor picked keeps its neighbors: re-judging it could only drop links.
        if (!touched.contains(node_id)) continue;

        dense_cuda_telemetry_t telemetry;
        dense_cuda_merge_links_one<kit_>( //
            view, bucket_ids, bucket_counts, bucket_width, node_id, cache, sorted_span, kept_positions,
            cache_norms[warp], gathered[warp], telemetry, lane);
        telemetry.flush<false>(view.telemetry, lane);
    }
}

/** Writes each query's nearest @c count IDs and distances out of a search walk, ascending, with
 *  sentinel IDs past the live prefix. */
template <dense_cuda_kit kit_>
struct dense_cuda_search_sink {
    using fold_t = dummy_fold_t;

    /** The landing for the matched IDs, @b [queries,count]. */
    dense_node_id_t* result_ids;

    /** The landing for the matched distances, @b [queries,count]; null skips the writes. */
    dense_distance_t* result_distances;

    /** Matches kept per query. */
    u32_t count;

    __device__ __forceinline__ fold_t open(u32_t, int) const noexcept { return {}; }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const&, dense_cuda_warp_context<frontier_>& context, u32_t query_index,
                                          int lane) const noexcept {
        // The rank extraction shuffles - every lane participates, only in-range ones store.
        context.frontier.finalize_ranks(lane);
        for (u32_t rank_base = 0; rank_base < count; rank_base += 32) {
            u32_t const rank = rank_base + lane;
            dense_cuda_scored_node_t const ranked_entry = context.frontier.rank_entry(rank);
            if (rank < count) {
                std::size_t const slot = static_cast<std::size_t>(query_index) * count + rank;
                result_ids[slot] = ranked_entry.id();
                if (result_distances) result_distances[slot] = kit_::convert_to_distance(ranked_entry.distance());
            }
        }
    }
};

/** Writes each query's kept node IDs and native dots out of a multiply walk, and the pairs it scored. */
template <dense_cuda_kit kit_>
struct dense_cuda_multiply_sink {
    using fold_t = dense_cuda_multiply_fold<typename kit_::dot_t>;

    /** Node IDs of the kept pairs, @b [queries,wanted], sentinel past them. */
    dense_node_id_t* ids;

    /** Native dots of the kept pairs, parallel to @c ids. */
    typename kit_::dot_t* products;

    /** Pairs scored per query, above @c wanted when a row was truncated. */
    u32_t* counts;

    /** Pairs kept per query. */
    u32_t wanted;

    __device__ __forceinline__ fold_t open(u32_t query_index, int) const noexcept {
        std::size_t const first = static_cast<std::size_t>(query_index) * wanted;
        return {ids + first, products + first, wanted};
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, dense_cuda_warp_context<frontier_>&, u32_t query_index,
                                          int lane) const noexcept {
        for (u32_t slot = fold.written + lane; slot < wanted; slot += 32) fold.ids[slot] = dense_node_id_unknown_k;
        if (lane == 0) counts[query_index] = fold.scored;
    }
};

/**
 *  @brief Writes each query's attention over its walk's nearest @c count nodes as an epilogue: the picks,
 *      the softmax carry, and the attended row, normalized.
 *
 *  Every lane derives the same maximum and partition from warp-wide reductions over the ranks. Each lane
 *  owns the value fragments congruent to it modulo 32 and gathers them from every pick once, so the row
 *  accumulates in registers, and a row's last, partial fragment is copied out whole bytes at a time.
 */
template <dense_cuda_kit kit_>
struct dense_cuda_attend_sink {
    using fold_t = dummy_fold_t;
    using fragment_t = typename kit_::fragment_t;
    using weighted_t = typename kit_::weighted_t;

    /** Output columns one fragment widens into. */
    inline static constexpr u32_t columns_per_fragment_k = sizeof(fragment_t) * 8 /
                                                           kit_::scalar_t::bits_per_dimension();

    /** The value rows every picked node ID addresses, stored like the keys, device-resident. */
    nk::matrix_view<typename kit_::scalar_t> values;

    /** The landing for the picked IDs, @b [queries,count], nearest first. */
    dense_node_id_t* attended;

    /** The attended rows, @b [queries,values.extent(1)] at @c output_stride, device-resident. */
    f32_t* outputs;

    /** Floats between consecutive output rows. */
    std::size_t output_stride;

    /** Each query's softmax carry, device-resident. */
    dense_softmax_state_t* states;

    /** The factor turning a dot into a logit. */
    f32_t score_scale;

    /** Picks kept per query. */
    u32_t count;

    __device__ __forceinline__ fold_t open(u32_t, int) const noexcept { return {}; }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const&, dense_cuda_warp_context<frontier_>& context, u32_t query_index,
                                          int lane) const noexcept {
        constexpr unsigned all_lanes_k = 0xFFFFFFFFu;
        frontier_& frontier = context.frontier;
        frontier.finalize_ranks(lane);
        // Nodes published after the call began have no value row, so like the sentinel they are never picked.
        auto const pick_at = [&](u32_t rank) noexcept {
            dense_cuda_scored_node_t const entry = frontier.rank_entry(rank);
            return rank < count && entry.id() < values.extent(0) ? entry : dense_cuda_scored_node_t::sentinel();
        };
        // The dot kernel's distance is `1 − dot`, so each pooled distance gives its logit back without rescoring.
        auto const logit_of = [&](dense_cuda_scored_node_t pick) noexcept {
            return score_scale * (1 - pick.distance());
        };

        dense_node_id_t* const picks = attended + static_cast<std::size_t>(query_index) * count;
        f32_t maximum_score = std::numeric_limits<f32_t>::lowest();
        for (u32_t rank_base = 0; rank_base < count; rank_base += 32) {
            dense_cuda_scored_node_t const pick = pick_at(rank_base + lane);
            if (rank_base + lane < count) picks[rank_base + lane] = pick.id();
            if (!(pick == dense_cuda_scored_node_t::sentinel())) maximum_score = fmaxf(maximum_score, logit_of(pick));
        }
        for (int offset = 16; offset; offset >>= 1)
            maximum_score = fmaxf(maximum_score, __shfl_xor_sync(all_lanes_k, maximum_score, offset));
        auto const weight_of = [&](dense_cuda_scored_node_t pick) noexcept {
            return pick == dense_cuda_scored_node_t::sentinel() ? 0.f : expf(logit_of(pick) - maximum_score);
        };
        f32_t partition = 0;
        for (u32_t rank_base = 0; rank_base < count; rank_base += 32) partition += weight_of(pick_at(rank_base + lane));
        for (int offset = 16; offset; offset >>= 1) partition += __shfl_xor_sync(all_lanes_k, partition, offset);

        u32_t const columns = static_cast<u32_t>(values.extent(1));
        u32_t const fragments = divide_round_up(columns, columns_per_fragment_k);
        f32_t* const output = outputs + static_cast<std::size_t>(query_index) * output_stride;
        f32_t const normalizer = partition > 0 ? 1 / partition : 0;
        for (u32_t fragment_base = 0; fragment_base < fragments; fragment_base += 32) {
            u32_t const fragment = fragment_base + lane;
            u32_t const fragment_columns =
                fragment < fragments ? min_of(columns - fragment * columns_per_fragment_k, columns_per_fragment_k) : 0;
            weighted_t sums {};
            for (u32_t rank_base = 0; rank_base < count; rank_base += 32) {
                dense_cuda_scored_node_t const pick = pick_at(rank_base + lane);
                f32_t const weight = weight_of(pick);
                for (u32_t source_lane = 0; source_lane != min_of(count - rank_base, 32u); ++source_lane) {
                    f32_t const pick_weight = __shfl_sync(all_lanes_k, weight, source_lane);
                    if (pick_weight == 0) continue;
                    dense_node_id_t const pick_id = __shfl_sync(all_lanes_k, pick.id(), source_lane);
                    fragment_t const* const row = reinterpret_cast<fragment_t const*>(
                        values.byte_data() + static_cast<std::ptrdiff_t>(pick_id) * values.stride_bytes(0));
                    if (fragment_columns == columns_per_fragment_k)
                        sums = kit_::accumulate_weighted(row[fragment], pick_weight, sums);
                    else if (fragment_columns) {
                        fragment_t tail {};
                        std::memcpy(&tail, row + fragment,
                                    divide_round_up(fragment_columns * kit_::scalar_t::bits_per_dimension(), 8u));
                        sums = kit_::accumulate_weighted(tail, pick_weight, sums);
                    }
                }
            }
            // Fully unrolled, so the sums stay in registers rather than land in local memory.
#pragma unroll
            for (u32_t column = 0; column != columns_per_fragment_k; ++column)
                if (column < fragment_columns)
                    output[fragment * columns_per_fragment_k + column] = sums[column] * normalizer;
        }
        if (lane == 0) states[query_index] = {maximum_score, partition};
    }
};

/** One query's full walk: stages the query into its shared span, walks the graph under the fold the
 *  sink opens and the query's slice of @p rescorings, and lets the sink close the query. Safe against
 *  concurrent insertions: the walk gates on every slot-0 word and never writes the graph. */
template <dense_cuda_kit kit_, u32_t resident_blocks_, typename sink_, typename rescoring_, typename frontier_>
__device__ __forceinline__ void dense_cuda_walk_one( //
    dense_cuda_search_view<kit_> const& view, sink_ const& sink, rescoring_ const& rescorings, u32_t query_index,
    typename kit_::fragment_t* query_span, dense_cuda_visited_set_t& visited, frontier_ frontier,
    dense_cuda_gathered<typename kit_::dot_t>& gathered, int lane) noexcept {

    typename kit_::fragment_t const* query_row = view.queries +
                                                 static_cast<std::size_t>(query_index) * view.planes.row_fragments;
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < view.planes.row_fragments; fragment += 32) {
        typename kit_::fragment_t const value = query_row[fragment];
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = dense_cuda_reduce_norms(norms_accumulator);
    norms_accumulator = dense_cuda_broadcast_norms(norms_accumulator);
    typename kit_::norms_t const query_state = kit_::partials_to_norms(norms_accumulator);
    __syncwarp();

    typename sink_::fold_t fold = sink.open(query_index, lane);
    rescoring_ const rescoring = rescorings.for_query(query_index);
    dense_cuda_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(query_index) * 33 + lane)};
    dense_cuda_search_around<kit_, resident_blocks_>( //
        view.planes, query_span, query_state, view.sampling_bound, visited, gathered, context, lane, fold, rescoring);
    sink.close(fold, context, query_index, lane);
    context.telemetry.tail_drops = context.frontier.dropped();
    context.telemetry.visited_saturations = visited.saturations_;
    context.telemetry.flush<false>(view.telemetry, lane);
}

/** One warp per query over the register frontier - expansions within the ladder - each walk scored under
 *  @p rescorings and written out through @p sink. */
template <dense_cuda_kit kit_, u32_t pool_slots_, typename sink_, typename rescoring_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, pool_slots_ <= 4 ? 4 : 3)
    dense_cuda_walk_kernel(dense_cuda_search_view<kit_> view, sink_ sink, rescoring_ rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ dense_cuda_gathered<typename kit_::dot_t> gathered[dense_cuda_warps_per_block_k];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_cuda_warps_per_block_k) {
        dense_cuda_visited_set_t visited {visited_slots, view.arena_shape.visited_hash_shift,
                                          view.epoch_base + query_index + 1};
        dense_cuda_walk_one<kit_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, sink, rescorings, query_index, query_span, visited, dense_cuda_register_frontier<pool_slots_> {},
            gathered[warp], lane);
    }
}

/** One warp per query past the register ladder's width: the same walk over the arena frontier, whose
 *  budget is the requested expansion itself; ranks cover the frontier's head of 64, so the index
 *  admits this path for searches only within it. */
template <dense_cuda_kit kit_, typename sink_, typename rescoring_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, 4)
    dense_cuda_walk_wide_kernel(dense_cuda_search_view<kit_> view, sink_ sink, rescoring_ rescorings) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ dense_cuda_gathered<typename kit_::dot_t> gathered[dense_cuda_warps_per_block_k];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_arena_frontier_t frontier;
    frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_cuda_warps_per_block_k) {
        dense_cuda_visited_set_t visited {visited_slots, view.arena_shape.visited_hash_shift,
                                          view.epoch_base + query_index + 1};
        dense_cuda_walk_one<kit_, 4u>(view, sink, rescorings, query_index, query_span, visited, frontier,
                                      gathered[warp], lane);
    }
}

/** Rebuilds the norms plane from the vector plane for rows @c [first_row,first_row+rows_count), one
 *  warp per row; norms are derived state, never serialized. */
template <dense_cuda_kit kit_>
__global__ void dense_cuda_derive_norms_kernel(dense_cuda_planes_view<kit_> planes, u32_t first_row, u32_t rows_count) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = first_row + warp_index; row < first_row + rows_count; row += warps) {
        typename kit_::norms_accumulator_t accumulator {};
        for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
            accumulator = kit_::accumulate_norms(planes.vector_for(static_cast<dense_node_id_t>(row), fragment),
                                                 accumulator);
        accumulator = dense_cuda_reduce_norms(accumulator);
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

/** Marks @p seed_count stride-spread rows reached and in the first frontier - the audit's landmark set, spread over
 *  the ID space like the traversal samplers. */
static __global__ void dense_cuda_mark_seeds_kernel(dense_cuda_bitset_t reached, dense_cuda_bitset_t frontier,
                                                    u32_t rows_count, u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    reached.insert(row);
    frontier.insert(row);
}

/** One level of forward reachability: every row of @p frontier marks its targets reached, and the ones it reaches
 *  first join @p next_frontier. Only the frontier expands, so a level is the same set of rows however the warps
 *  interleave, and sweeping until a level expands nothing takes the landmarks' eccentricity in levels. */
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

/** Counts the rows @p reached lacks - the island population - in @p islands_cursor, collecting their IDs into
 *  @p island_ids when it is given. */
static __global__ void dense_cuda_reachability_islands_kernel(dense_cuda_bitset_t reached, u32_t rows_count,
                                                              u32_t* island_ids, u32_t* islands_cursor) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count || reached.contains(row)) return;
    u32_t const slot = cuda_atomic_ref<u32_t>(*islands_cursor).fetch_add(1u, cuda::memory_order_relaxed);
    if (island_ids) island_ids[slot] = row;
}

/** Counts, for every row, the reached rows that point at it - the links a reconnection may take from a reached
 *  resident only while another one remains. */
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

/** Reconnects each island into the first of its reached forward picks with room for it, or into the landmark covering
 *  its ID, bypassing the diversity judgment that dropped it: a free slot when one is left, otherwise a landing spread
 *  across the tail. Every landing is a compare-and-swap that never displaces an unreached sibling, so islands sharing
 *  one target all land, nor a reached resident whose @p in_links would drop to zero, so no reconnection strands
 *  another node; @p reconnected counts the links actually written. Exclusive pass: the launch boundary publishes every
 *  write, so each access is relaxed. */
static __global__ void dense_cuda_reconnect_island_kernel(u32_t* neighbors, u32_t connectivity, u32_t const* island_ids,
                                                          u32_t islands_count, dense_cuda_bitset_t reached,
                                                          u32_t rows_count, u32_t seed_count, u32_t* in_links,
                                                          u32_t* reconnected) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const spread = min_of(16u, connectivity / 2);
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
            // A reached resident gives this link up only while another reached row still points at it.
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

#endif // defined(__CUDACC__)

#if defined(__CUDACC__)

template <nk_dtype_t, nk_kernel_kind_t, cuda_arch_t, typename, typename, typename>
struct dense_vram_index;

/** One stream's device-side scratch for driving a CUDA index: the traversal arena, the batch
 *  staging, and the telemetry counters, riding the stream its allocator names and never opening one.
 *  Two contexts on one index share only the planes and the published counter, never scratch. */
template <cuda_allocator allocator_type_ = cuda_device_allocator_t>
struct dense_cuda_traversal_context {
    using allocator_t = allocator_type_;

    /** Matches the single-query verb lands at most - the arena frontier's head width. */
    inline static constexpr u32_t results_capacity_k = 64;

    /** The single-query verb's landing: ascending matches in inline storage. */
    struct results_t {
        dense_scored_node_t entries_[results_capacity_k];
        std::size_t count_ = 0;

        dense_scored_node_t* data() noexcept { return entries_; }
        dense_scored_node_t const* data() const noexcept { return entries_; }
        dense_scored_node_t& operator[](std::size_t rank) noexcept { return entries_[rank]; }
        dense_scored_node_t const& operator[](std::size_t rank) const noexcept { return entries_[rank]; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** Binds this context to @p allocator's stream and device, a null stream naming the legacy
     *  default one, and allocates only the fixed-size counters; sized buffers wait for the first
     *  verb. */
    dense_cuda_traversal_context(allocator_t allocator = {}) noexcept
        : allocator_ {allocator}, //
          telemetry_ {cuda_make_unique(telemetry_allocator_t {allocator_}, 1)},
          assigned_rows_ {cuda_make_unique(counter_allocator_t {allocator_}, 1)},
          arena_ {nullptr, cuda_deleter<allocator_t> {allocator_, 0}},
          staging_ {nullptr, cuda_deleter<allocator_t> {allocator_, 0}} {
        if (telemetry_) cudaMemsetAsync(telemetry_.get(), 0, sizeof(dense_cuda_telemetry_block_t), stream());
    }

    dense_cuda_traversal_context(dense_cuda_traversal_context const&) = delete;
    dense_cuda_traversal_context& operator=(dense_cuda_traversal_context const&) = delete;

    /** Every owned resource frees in stream order, so a move is a member-wise steal and destruction
     *  needs no drain. */
    dense_cuda_traversal_context(dense_cuda_traversal_context&&) noexcept = default;
    ~dense_cuda_traversal_context() noexcept = default;

    cudaStream_t stream() const noexcept { return allocator_.stream(); }
    int device() const noexcept { return allocator_.device(); }

    /** Whether this context can walk at all: both device-side sinks claimed. */
    explicit operator bool() const noexcept { return telemetry_ && assigned_rows_; }

    /** Waits for every transfer and launch queued on this context's stream - one stream's
     *  completion, the executor's @c drain being the fleet's. */
    status_t drain() noexcept {
        return cudaStreamSynchronize(stream()) == cudaSuccess ? status_t::success_k : status_t::device_error_k;
    }

    /** The telemetry of the last submission, zeroed when it was queued; read after a drain. */
    dense_cuda_telemetry_block_t spent() const noexcept {
        dense_cuda_telemetry_block_t snapshot = {};
        cudaMemcpy(&snapshot, telemetry_.get(), sizeof(snapshot), cudaMemcpyDeviceToHost);
        return snapshot;
    }

  private:

    /** Only the index's verbs reach the raw scratch - callers observe a context through @c results
     *  and @ref spent, never through its buffers. */
    template <nk_dtype_t, nk_kernel_kind_t, cuda_arch_t, typename, typename, typename>
    friend struct dense_vram_index;

    using telemetry_allocator_t = typename allocator_t::template rebind<dense_cuda_telemetry_block_t>::other;
    using counter_allocator_t = typename allocator_t::template rebind<u32_t>::other;
    using bytes_ptr_t = cuda_unique_ptr<byte_t, allocator_t>;

    allocator_t allocator_;

    /** Monotone epoch floor over the arena's visited slots, so the arena is zeroed only when
     *  freshly allocated or when the 32-bit epoch space wraps, never per launch. */
    u64_t epoch_floor_ = 0;

    /** The device-side counters of the last submission, zeroed when it was queued. */
    cuda_unique_ptr<dense_cuda_telemetry_block_t, telemetry_allocator_t> telemetry_;

    /** Rows of the current launch's batch warps have assigned themselves so far. */
    cuda_unique_ptr<u32_t, counter_allocator_t> assigned_rows_;
    bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** Grow-only traversal arena for @p bytes: the exact visited sets and spill tails of
     *  every resident warp, kept across calls. @return The base, or null on exhaustion. */
    byte_t* ensure_arena(std::size_t bytes) noexcept { return ensure_(arena_, arena_bytes_, bytes); }

    /** Grow-only staging for one batch's device transport: an insert's padded vectors, or a
     *  search's queries and result landing. */
    byte_t* ensure_staging(std::size_t bytes) noexcept { return ensure_(staging_, staging_bytes_, bytes); }

    /** Grows @p buffer to @p bytes when @p capacity falls short; @return the base, or null. */
    byte_t* ensure_(bytes_ptr_t& buffer, std::size_t& capacity, std::size_t bytes) noexcept {
        if (bytes <= capacity) return buffer.get();
        // The deleter already carries this context's allocator, so only the run and its count change.
        buffer.reset(allocator_.allocate(bytes));
        buffer.get_deleter().count = bytes;
        capacity = buffer ? bytes : 0;
        return buffer.get();
    }
};

/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  Speaks the RAM-resident dense index's verbs, with two divergences: a batch's IDs are one
 *  contiguous range, and publication is unordered - see @ref dense_vram_index_concurrency.
 *  The first successful @c reserve freezes the geometry and the virtual-address ceiling;
 *  physical slabs map on demand, and fresh neighbors are sentinel-filled before any ID
 *  covering them can be assigned. The planes grow only while @c claiming_ is held, and the
 *  claim counter and the kept status are the only mutable host state; the published-row count
 *  is device memory no host thread dereferences.
 */
template <                                                                      //
    nk_dtype_t vector_dtype_ = nk_i8_k,                                         //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                        //
    cuda_arch_t capability_ = cuda_arch_ampere_k,                               //
    typename vectors_allocator_type_ = cuda_virtual_memory_allocator<byte_t>,   //
    typename neighbors_allocator_type_ = cuda_virtual_memory_allocator<byte_t>, //
    typename traversals_allocator_type_ = cuda_arena_allocator_t>               //
struct dense_vram_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using kit_chain_t = dense_cuda_kit_chain<vector_dtype_, kernel_kind_, capability_>;
    using kit_t = typename kit_chain_t::kit_t;
    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using scalar_t = typename kit_t::scalar_t;
    using dot_t = typename kit_t::dot_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    // The device writes the kit's plain accumulator into these bytes, so the two must share a representation.
    static_assert(sizeof(dot_result_t) == sizeof(dot_t));
    using fragment_t = typename kit_t::fragment_t;
    using norms_t = typename kit_t::norms_t;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_view_t = nk::vector_view<scalar_t>;
    using node_id_t = dense_node_id_t;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using distance_t = dense_distance_t;
    using small_size_t = u32_t;
    using planes_view_t = dense_cuda_planes_view<kit_t>;

    /** The page-locked host bytes a checkpoint moves through, one batch at a time. */
    using bounce_t = growing_buffer<byte_t, cuda_pinned_allocator_t>;

    /** Rows one checkpoint batch carries between a plane and the caller: one slab, the row
     *  granularity the RAM residency grows by. */
    inline static constexpr std::size_t bounce_rows_log2_k = 20;
    using dense_traversal_context_t = dense_cuda_traversal_context<traversals_allocator_t>;

    /** The strongest @c synchronization_t contract this design honors; nothing routes on it.
     *  Batches claim disjoint ID ranges and the publication protocol makes overlapping searches
     *  safe. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;

    /** Capability labels for the shared reports: the GPU picks no CPU SIMD tier. */
    inline static constexpr nk_capability_t picked_quads_capability_k = nk_cap_serial_k;

    /** The tier of the kit the chain picked under @c capability_ - the GPU twin of
     *  @c picked_quads_capability_k, surfaced in the bench reports. */
    inline static constexpr cuda_arch_t picked_capability_k = kit_chain_t::picked_k;
    inline static constexpr bool has_tiles_k = false;
    inline static constexpr nk_capability_t picked_tiles_capability_k = nk_cap_serial_k;

    inline static constexpr std::size_t max_node_count_k = dense_cuda_scored_node_t::id_capacity_k;

    /** Default visited-set slot count; the scratch slice must also fit the candidate cache. */
    inline static constexpr u32_t default_visited_capacity_k = 2048;

    /** The once-frozen geometry word: dimensions in the high half, connectivity in the low, zero
     *  meaning not yet reserved. */
    using packed_geometry_id_t = u64_t;
    inline static constexpr packed_geometry_id_t pack_geometry(std::size_t dimensions,
                                                               std::size_t connectivity) noexcept {
        return (static_cast<packed_geometry_id_t>(dimensions) << 32u) | static_cast<packed_geometry_id_t>(connectivity);
    }
    inline static constexpr u32_t geometry_dimensions(packed_geometry_id_t geometry) noexcept {
        return static_cast<u32_t>(geometry >> 32u);
    }
    inline static constexpr u32_t geometry_connectivity(packed_geometry_id_t geometry) noexcept {
        return static_cast<u32_t>(geometry);
    }

#pragma region Placement

    /** One allocator per plane concern, each carrying its device placement; the index pins itself
     *  to the vectors allocator's device and demands the neighbors allocator agree. */
    dense_vram_index(vectors_allocator_t const& vectors_allocator = {},
                     neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, vectors_plane_(vectors_allocator), norms_plane_(vectors_allocator),
          neighbors_plane_(neighbors_allocator) {}
    dense_vram_index(dense_vram_index const&) = delete;
    dense_vram_index& operator=(dense_vram_index const&) = delete;

#pragma endregion Placement

#pragma region Measures

    /** Claimed rows, the only dense bound under unordered publication; the sampler probes past
     *  unpublished IDs by protocol. */
    std::size_t size() const noexcept {
        return standard_atomic_ref<std::size_t const>(claimed_rows_).load(std::memory_order_acquire);
    }

    /** The first failure that left claimed rows unpublished, kept for the index's lifetime; it
     *  refuses every later mutation while searches keep running. */
    status_t status() const noexcept {
        return standard_atomic_ref<status_t const>(status_).load(std::memory_order_acquire);
    }

    /** Rows mapped across all three planes, the smallest of the three, since slabs are a fixed byte
     *  size and each plane maps a different row count per slab. */
    std::size_t capacity() const noexcept {
        std::size_t const neighbors_row_bytes = connectivity() * sizeof(u32_t);
        std::size_t const vector_row_bytes = row_bytes();
        if (!neighbors_row_bytes || !vector_row_bytes) return 0;
        std::size_t const by_vectors = vectors_plane_.committed_bytes() / vector_row_bytes;
        std::size_t const by_norms = norms_plane_.committed_bytes() / sizeof(norms_t);
        std::size_t const by_neighbors = neighbors_plane_.committed_bytes() / neighbors_row_bytes;
        return min_of(by_vectors, min_of(by_norms, by_neighbors));
    }
    std::size_t connectivity() const noexcept {
        return geometry_connectivity(
            standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }
    std::size_t dimensions() const noexcept {
        return geometry_dimensions(
            standard_atomic_ref<packed_geometry_id_t const>(geometry_).load(std::memory_order_acquire));
    }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    std::size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : connectivity() * 2;
    }

    /** The pinned device's identity and limits, queried once at the first @c reserve. */
    cuda_device_specs_t const& device_specs() const noexcept { return device_; }

    /** Device-side plane views - the audit surface for invariant checkers & tests. */
    planes_view_t planes() const noexcept {
        u32_t const fragments = row_fragments();
        return {reinterpret_cast<fragment_t*>(vectors_plane_.base()),
                reinterpret_cast<norms_t*>(norms_plane_.base()),
                reinterpret_cast<u32_t*>(neighbors_plane_.base()),
                static_cast<u32_t>(connectivity()),
                fragments,
                lanes_per_row(),
                steps_per_row(),
                static_cast<u32_t>(((u64_t {1} << 31) + fragments - 1) / fragments)};
    }

    /** Bytes of one packed logical row, bit-exact for sub-byte scalars. */
    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }

    /** Fragments per vector row, derived from the frozen geometry like every row measure. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), sizeof(typename kit_t::fragment_t));
    }

    /** The vector row's byte length, rounded up to whole fragments; the tail bytes stay zero so
     *  every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept {
        return static_cast<std::size_t>(row_fragments()) * sizeof(typename kit_t::fragment_t);
    }

    /** Lanes cooperating on one row: the smallest power of two covering its fragments, at most a
     *  warp. */
    u32_t lanes_per_row() const noexcept { return min_of(std::bit_ceil(row_fragments()), 32u); }

    /** Steps those lanes take across one row. */
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), lanes_per_row()); }

    /** Bytes of one warp's dynamic-shared slice on the insertion path: the query span while
     *  traversing, then the candidate cache & re-selection spans. */
    u32_t insert_scratch_bytes_per_warp() const noexcept {
        u32_t const selection_slots = min_of(static_cast<u32_t>(connectivity()), //
                                             static_cast<u32_t>(dense_cuda_cache_rows_k));
        u32_t const selection_bytes = selection_slots * static_cast<u32_t>(sizeof(dense_cuda_scored_node_t) + //
                                                                           2 * sizeof(u32_t));
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * dense_cuda_cache_rows_k;
        return max_of(static_cast<u32_t>(row_bytes()), cache_bytes + selection_bytes);
    }

    /** Bytes of one warp's dynamic-shared slice on the refinement merge path: the candidate cache -
     *  whose last row hosts the node's own vector - then the scored span and the kept-row indices. */
    u32_t merge_scratch_bytes_per_warp() const noexcept {
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * dense_cuda_cache_rows_k;
        return cache_bytes +
               dense_cuda_cache_rows_k * static_cast<u32_t>(sizeof(dense_cuda_scored_node_t) + sizeof(u32_t));
    }

    /** Whether a search, multiply or attend launch seats on this device: the query span per warp beside the
     *  static batch buffers. */
    bool walk_scratch_fits() const noexcept {
        return std::size_t {dense_cuda_warps_per_block_k} * (row_bytes() + sizeof(dense_cuda_gathered<dot_t>)) <=
               device_.shared_memory_per_block_bytes;
    }

    /** Whether an insert or refine launch seats on this device: its per-warp candidate cache beside the
     *  static norms and batch buffers - wide rows can serve walks yet refuse building. */
    bool build_scratch_fits() const noexcept {
        std::size_t const statics = std::size_t {dense_cuda_warps_per_block_k} *
                                    (dense_cuda_cache_rows_k * sizeof(norms_t) + sizeof(dense_cuda_gathered<dot_t>));
        return statics + std::size_t {dense_cuda_warps_per_block_k} *
                             max_of(insert_scratch_bytes_per_warp(), merge_scratch_bytes_per_warp()) <=
               device_.shared_memory_per_block_bytes;
    }

    /** The serialized extents: published nodes x logical dimensions & connectivity - the fragment
     *  padding is a storage detail that never reaches a file. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /** Where a save lands: two row-major host planes the caller sized from @ref saved_shape, handed
     *  over at planning so no step names a destination of its own. */
    struct columns_t {

        /** The vectors plane, one packed row per node. */
        bytes_span_t vectors;

        /** The neighbors plane, @c connectivity IDs per node. */
        bytes_span_t neighbors;
    };

    /** Where a load reads from - the read-only twin of @ref columns_t. */
    struct columns_view_t {

        /** The vectors plane, one packed row per node. */
        bytes_view_t vectors;

        /** The neighbors plane, @c connectivity IDs per node. */
        bytes_view_t neighbors;
    };

    /** Bytes this index occupies - the three committed device planes, the published-row count, and
     *  the host struct; traversal scratch belongs to executors and contexts. */
    std::size_t memory_usage() const noexcept {
        return sizeof(dense_vram_index) + vectors_plane_.committed_bytes() + norms_plane_.committed_bytes() +
               neighbors_plane_.committed_bytes() + (published_rows_ ? sizeof(u32_t) : 0);
    }

    /** Copies the stored row of @p node into @p destination with the fragment padding stripped,
     *  through one blocking copy on no stream, so it needs no context; the bulk export is @c save. */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination) const noexcept {
        if (node >= size()) return status_t::config_mismatch_k;
        if (destination.size() != dimensions()) return status_t::config_mismatch_k;
        if (!destination.is_contiguous()) return status_t::unsupported_layout_k;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        return cudaMemcpy(destination.byte_data(), vectors_plane_.base() + static_cast<std::size_t>(node) * row_bytes(),
                          scalar_row_bytes, cudaMemcpyDeviceToHost) == cudaSuccess
                   ? status_t::success_k
                   : status_t::device_error_k;
    }

    /** Reports whether every claimed ID is published - a derived health check, never cached state.
     *  Call it only with no batch in flight. */
    status_t validate() const noexcept {
        u32_t published = 0;
        if (cudaMemcpy(&published, published_rows_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            return status_t::device_error_k;
        return published == size() ? status_t::success_k : status_t::device_error_k;
    }

#pragma endregion Measures

#pragma region Reservation

    /** Which frontier a warp's arena slice serves: the register ladder needs only the exact visited
     *  set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** One warp's arena slice for a walk expanding @p expansion candidates: the exact visited set,
     *  sized well past the expected visit count so probing stays short, plus the spill tail when
     *  @p frontier_kind outgrows registers. */
    static dense_cuda_arena_shape_t arena_shape_for(u32_t expansion, frontier_kind_t frontier_kind) noexcept {
        u32_t const visited_slots = std::bit_ceil(64 * expansion);
        return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)),
                frontier_kind == frontier_kind_t::wide_arena_k ? 32 * expansion : 0};
    }

    /** Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;
    };

    /** Ensures the index can host at least @p config.capacity entries and answers the capacity now
     *  in effect; the first successful call freezes the geometry and the virtual-address ceiling,
     *  later ones only map slabs. */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (neighbors_plane_.device() != device_.ordinal) return {0, status_t::config_mismatch_k};
        if (!config.dimensions || !std::has_single_bit(static_cast<u32_t>(config.connectivity)))
            return {0, status_t::config_mismatch_k};
        // The first success freezes both axes at once; later calls must agree with what they observe.
        packed_geometry_id_t const requested = pack_geometry(config.dimensions,
                                                             static_cast<std::size_t>(config.connectivity));
        packed_geometry_id_t observed = 0;
        if (!standard_atomic_ref<packed_geometry_id_t>(geometry_).compare_exchange_strong(
                observed, requested, std::memory_order_acq_rel, std::memory_order_acquire) &&
            observed != requested)
            return {capacity(), status_t::config_mismatch_k};
        connectivity_t const connectivity = config.connectivity;
        if (config.capacity > max_node_count_k) return {capacity(), status_t::overflow_k};
        if (!vectors_plane_.reserved_bytes()) { // First reservation: bind the device, size the ceiling.
            // The primary context must exist before any driver-API mapping, since the planes live in it.
            if (cudaSetDevice(device_.ordinal) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
                return {0, status_t::device_error_k};
            auto const [specs, status] = cuda_device_specs_t::query(device_.ordinal);
            if (failed(status)) return {0, status};
            device_ = specs;
            // Geometry no walk can seat is refused here; building checks its own wider scratch per call.
            if (!walk_scratch_fits()) return {0, status_t::config_mismatch_k};
            cuInit(0);
            // Virtual address space is free, so the ceiling is generous and independent of the first
            // requested capacity; the reservation itself is the ceiling, no cached row count shadows it.
            std::size_t const bytes_per_row = row_bytes() + sizeof(norms_t) + connectivity * sizeof(u32_t);
            // Device planes budget against VRAM; managed planes oversubscribe it and budget against host
            // memory through their own backend, which keeps the topology symbols out of device instantiations.
            std::size_t budget_bytes = 4 * device_.vram_bytes;
            if constexpr (vectors_allocator_t::address_space_k == cuda_address_space_t::managed_k)
                budget_bytes = vectors_allocator_t::budget_bytes();
            std::size_t const ceiling_rows = min_of(max_node_count_k,
                                                    max_of<std::size_t>(config.capacity, budget_bytes / bytes_per_row));
            if (vectors_plane_.reserve_range(ceiling_rows * row_bytes(), device_.ordinal) != status_t::success_k ||
                norms_plane_.reserve_range(ceiling_rows * sizeof(norms_t), device_.ordinal) != status_t::success_k ||
                neighbors_plane_.reserve_range(ceiling_rows * connectivity * sizeof(u32_t), device_.ordinal) !=
                    status_t::success_k)
                return {0, status_t::bad_alloc_k};
            counter_allocator_t counter_allocator {nullptr, device_.ordinal};
            published_rows_ = cuda_make_unique(counter_allocator, 1);
            if (!published_rows_ || cudaMemset(published_rows_.get(), 0, sizeof(u32_t)) != cudaSuccess)
                return {0, status_t::device_error_k};
        }
        status_t const grown = grow_rows_(config.capacity, nullptr);
        return {capacity(), grown};
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** Breadth of the construction candidate pool; rounded up to a multiple of 32, clamped to
         *  512. Larger pools trade build speed for graph quality. */
        small_size_t expansion_factor = 0;
    };

    /** The bytes an executor's run must hold before a context carved over it can insert @p rows
     *  rows, one aligned term per carve the verb makes; zero before @c reserve freezes the
     *  geometry. */
    std::size_t traversal_memory_usage(std::size_t rows, insert_config_t config = {}) const noexcept {
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32,
                                                                     frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        max_of(insert_blocks_per_multiprocessor_k, search_blocks_per_multiprocessor_k) *
                                        dense_cuda_warps_per_block_k;
        std::size_t const results_bytes = // The widest single-query result landing.
            512 * (sizeof(dense_node_id_t) + sizeof(dense_distance_t));
        return aligned(sizeof(dense_cuda_telemetry_block_t)) + aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes()) + aligned(rows * row_bytes() + results_bytes);
    }

    /** Telemetry of one @c insert() batch. IDs are the contiguous range
     *  @b [first_id,first_id+count) - the assigned order of the incoming rows. */
    struct [[nodiscard]] insert_result_t {
        std::size_t new_size = 0;
        dense_node_id_t first_id = dense_node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;

        /** Neighbors the back-linker abandoned after the bounded lock budget - GPU-only telemetry. */
        std::size_t unlinked_neighbors = 0;

        /** Frontier tail entries shed past their arena slice - GPU-only telemetry. */
        std::size_t tail_drops = 0;

        /** Visited-set claims that ran out of probes and starved a walk - GPU-only telemetry. */
        std::size_t visited_saturations = 0;

        /** Refine-side distance work, folded into @c computed_distances here, so always zero. */
        std::size_t computed_distances_in_refines = 0;

        /** Reverse-refine distance work, folded the same way, so always zero. */
        std::size_t computed_distances_in_reverse_refines = 0;
    };

    /** Inserts @p new_vectors under the contiguous ID range @b [size(),size()+rows), every transfer
     *  and the launch riding the context's stream. Blocking drains that stream and reports
     *  telemetry; deferred returns once queued, and the caller drains before reading anything, the
     *  host batch included. */
    expected<insert_result_t> insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                     dense_traversal_context_t& context, insert_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        expected<insert_result_t> inserted = submit_insert_(new_vectors, assigned_ids, context, config);
        if (completion == completion_t::deferred_k || !inserted || !new_vectors.extent(0)) return inserted;
        if (status_t const drained = context.drain(); failed(drained))
            return {{inserted.value.new_size, inserted.value.first_id}, fail_(drained)};
        dense_cuda_telemetry_block_t const spent = context.spent();
        inserted.value.visited_members = spent.hops;
        inserted.value.computed_distances = spent.scored;
        inserted.value.restarts = spent.restarts;
        inserted.value.unlinked_neighbors = spent.lock_skips;
        inserted.value.tail_drops = spent.tail_drops;
        inserted.value.visited_saturations = spent.visited_saturations;
        return inserted;
    }

#pragma endregion Insertion

#pragma region Search

    /** Optional tuning knobs of @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {

        /** Breadth of the candidate pool; defaults to twice the connectivity, and is raised
         *  automatically when @c count needs a wider pool. */
        small_size_t expansion_factor = 0;

        /** Matches reported per query, up to the widest supported pool of 512. */
        small_size_t count = 10;

        /** The caller's mutation contract; every level shares one code path, since the walk is
         *  insert-safe by protocol, so the field only documents intent. */
        synchronization_t synchronization = synchronization_t::concurrent_k;
    };

    /** Telemetry of one @c search() batch. */
    struct [[nodiscard]] search_result_t {
        std::size_t count = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        std::size_t tail_drops = 0;
        std::size_t visited_saturations = 0;
    };

    /** The bytes an executor's run must hold before a context carved over it can search or multiply
     *  @p rows rows under @p config, landing @c config.count columns per row, one aligned term per
     *  carve the verb makes. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        static_assert(sizeof(dot_t) == sizeof(dense_distance_t));
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        // Search widens the pool to cover its count while multiply walks the bare factor, so the wider one sizes.
        u32_t const expansion = max_of<u32_t>(
            config.expansion_factor > config.count ? config.expansion_factor : config.count * 2,
            static_cast<u32_t>(non_zero_expansion_factor(config)));
        bool const wide = expansion > dense_cuda_sorted_pool<16>::capacity_k;
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(
            wide ? expansion : pool_slots * 32,
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        search_blocks_per_multiprocessor_k * dense_cuda_warps_per_block_k;
        std::size_t const staging_bytes = rows * row_bytes() +
                                          rows * config.count * (sizeof(dense_node_id_t) + sizeof(dense_distance_t)) +
                                          rows * sizeof(u32_t);
        return aligned(sizeof(dense_cuda_telemetry_block_t)) + aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes()) + aligned(staging_bytes);
    }

    /** The bytes a search under the indexer @p heads needs: the plain walk's, plus the staging of the heads and
     *  their weights. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config,
                                       dense_indexer_heads<scalar_t> const& heads) const noexcept {
        // The heads share the staging carve, and aligning their bytes apart can only overestimate it.
        std::size_t const heads_bytes = heads.heads.extent(0) * (row_bytes() + sizeof(f32_t));
        return traversal_memory_usage(rows, config) +
               divide_round_up<std::size_t>(heads_bytes, default_alignment_k) * default_alignment_k;
    }

    /**
     *  @brief Finds the nearest published neighbors of every query row, ascending by distance with
     *      sentinel IDs marking absent matches. Safe to overlap with insertions on other streams.
     *
     *  @param[in] queries The query rows, in the index's stored scalar type.
     *  @param[out] result_ids IDs of the matches, @b [queries,count] with sentinel fills, in host or
     *      device memory.
     *  @param[out] result_distances Distances of the same matches, in host or device memory; an empty
     *      span skips them.
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth and match count.
     *  @param[in] completion Blocking drains the context's stream and reports the telemetry;
     *      deferred returns once queued, and the caller drains before reading either output.
     *  @return The match count per query, and the traversal telemetry this call spent.
     */
    expected<search_result_t> search(vectors_view_t queries, nk::matrix_span<dense_node_id_t> result_ids,
                                     nk::matrix_span<dense_distance_t> result_distances,
                                     dense_traversal_context_t& context, search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        return settle_(submit_search_(queries, dummy_heads_t {}, result_ids, result_distances, context, config),
                       queries.extent(0), context, completion);
    }

    /** The batched @c search ranked by a lightning indexer: a node's distance is `1 − Σₕ wₕ·ReLU(qₕ·k)` over the
     *  @p heads of its query, which must match the queries in count and the index in dimensions, while the walk
     *  navigates by @p queries. */
    expected<search_result_t> search(vectors_view_t queries, dense_indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<dense_node_id_t> result_ids,
                                     nk::matrix_span<dense_distance_t> result_distances,
                                     dense_traversal_context_t& context, search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k && !nk::is_integral_dtype<scalar_t>())
    {
        return settle_(submit_search_(queries, heads, result_ids, result_distances, context, config), queries.extent(0),
                       context, completion);
    }

    /** One query in, the nearest matches ascending in @c context.results - the correctness path; a
     *  one-query grid never fills the machine, the batched overload does. */
    expected<search_result_t> search(vector_view_t query_vector, dense_traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        context.results.count_ = 0;
        if (!size()) return {{}, status_t::success_k}; // An empty index matches nothing, touches nothing.
        // Up to the expansion factor, defaulting to twice the connectivity, clipped to the landing and the graph.
        u32_t const expansion = static_cast<u32_t>(non_zero_expansion_factor(config));
        u32_t const count = min_of<u32_t>(min_of<u32_t>(dense_traversal_context_t::results_capacity_k, expansion),
                                          static_cast<u32_t>(size()));
        dense_node_id_t staged_ids[dense_traversal_context_t::results_capacity_k];
        dense_distance_t staged_distances[dense_traversal_context_t::results_capacity_k];
        vectors_view_t const one_query {query_vector.data(), {1, query_vector.size()}};
        config.count = count;
        expected<search_result_t> result = search(one_query, nk::matrix_span<dense_node_id_t> {staged_ids, {1, count}},
                                                  nk::matrix_span<dense_distance_t> {staged_distances, {1, count}},
                                                  context, config);
        if (!result) return result;
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != dense_node_id_unknown_k; ++live)
            context.results.entries_[live] = dense_scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        result.value.count = live;
        return result;
    }

    /** Filtered search is banned on the device index: a host predicate cannot run inside the walk,
     *  and a device-resident admission bitmap is a design of its own. */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t, predicate_type_&&, dense_traversal_context_t&,
                                        search_config_t = {}) noexcept = delete;

    /**
     *  @brief Graph products of every left-hand row: each row keeps the pairs its walk scored, node
     *      IDs beside native dots, until its output row fills.
     *
     *  @param[in] left The left-hand rows, in the index's stored scalar type.
     *  @param[out] node_ids The node ID of each kept pair, @b [rows,wanted] in walk order, sentinel
     *      past the kept pairs, in host or device memory.
     *  @param[out] products The device's native dot of each kept pair, parallel to @p node_ids and
     *      unspecified past them, in host or device memory.
     *  @param[out] counts Pairs scored per row, above @c wanted when the row was truncated, in host
     *      memory.
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth; @c count plays no part, the outputs' width does.
     *  @param[in] completion Blocking drains the context's stream and reports the telemetry;
     *      deferred returns once queued, and the caller drains before reading any output.
     *  @return The pairs scored over every row once blocking - zero when deferred, where @p counts
     *      carry them after the drain - and the traversal telemetry this call spent.
     */
    expected<search_result_t> multiply(vectors_view_t left, nk::matrix_span<dense_node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       dense_traversal_context_t& context, search_config_t config = {},
                                       completion_t completion = completion_t::blocking_k) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        expected<search_result_t> landed = settle_(submit_multiply_(left, node_ids, products, counts, context, config),
                                                   left.extent(0), context, completion);
        if (landed && completion == completion_t::blocking_k)
            for (u32_t const scored : counts) landed.value.count += scored;
        return landed;
    }

    /**
     *  @brief Softmax-weighted attention of every query row over its @c k most relevant published nodes: the
     *      walk keeps them as @c search keeps its matches, and the warp gathers their value rows after it.
     *
     *  @param[in] queries The query rows, in the index's stored scalar type.
     *  @param[in] values One value row per node ID, stored like the keys, device-resident, at least as many
     *      rows as the index, each starting on a 16-byte boundary.
     *  @param[out] attended The picks, @b [queries,k] nearest first with sentinel IDs past the live ones, in
     *      host or device memory; @c k is its width, within the frontier head @c search admits.
     *  @param[out] outputs The attended rows, @b [queries,values.extent(1)], device-resident.
     *  @param[out] carry Each row's softmax state over its picks in host memory, one per query.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[inout] context The stream every transfer and the launch ride.
     *  @param[in] config The per-call breadth; @c count plays no part, the width of @p attended does.
     *  @param[in] completion As in @c multiply.
     *  @return The picks per query, and the traversal telemetry this call spent. Values or outputs
     *      outside this device refuse with @c device_error_k rather than being uploaded.
     */
    expected<search_result_t> attend(vectors_view_t queries, vectors_view_t values,
                                     nk::matrix_span<dense_node_id_t> attended, nk::matrix_span<f32_t> outputs,
                                     std::span<dense_softmax_state_t> carry, f32_t score_scale,
                                     dense_traversal_context_t& context, search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        return settle_(submit_attend_(queries, values, attended, outputs, carry, score_scale, context, config),
                       queries.extent(0), context, completion);
    }

#pragma endregion Search

#pragma region Refinement

    /** Optional tuning knobs of @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** Breadth of the re-selection pool, rounded up to a multiple of 32 and clamped to 512;
         *  zero takes twice the connectivity, like the CPU's. */
        small_size_t expansion_factor = 0;
    };

    using refine_result_t = dense_refine_result_t;
    using health_result_t = dense_health_result_t;

    /**
     *  @brief Re-selects the neighbors of every node in @p nodes from a fresh walk seeded by the ones it holds, then
     *      heals whatever no search can reach - the same pass the CPU's @c refine_plan_t runs. An empty range heals.
     *
     *  The walk lands shadow neighbors, the scatter buckets every pick under its target, and the merge re-judges
     *  forward and reverse candidates together, backfilling what diversity leaves empty. The heal then sweeps
     *  reachability and, round by round while the islands shrink, re-selects them and reconnects whatever stays
     * unreached into a reached pick or the landmark covering it.
     *
     *  Demands an exclusive caller: the merge writes live neighbors with plain stores. A stopped sequence of passes
     *  leaves the graph correct and merely less improved.
     *
     *  @param[in] nodes The ID range to walk; IDs at or past @c size() are skipped.
     *  @param[inout] context The driving context; its arena grows to the pass's shape.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<refine_result_t> refine(iota_view<dense_node_id_t> nodes, dense_traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
        std::size_t const rows = size();
        refine_result_t result;
        auto add_walks = [&](refine_result_t const& walked) noexcept {
            result.visited_nodes += walked.visited_nodes;
            result.changed_nodes += walked.changed_nodes;
            result.visited_members += walked.visited_members;
            result.computed_distances += walked.computed_distances;
            result.restarts += walked.restarts;
        };
        dense_node_id_t const bound = static_cast<dense_node_id_t>(min_of<std::size_t>(nodes.bound, rows));
        if (nodes.first < bound) {
            auto const [walked, status] = refine_pass_(context, config, {nodes.first, bound}, nullptr);
            if (failed(status)) return {result, status};
            add_walks(walked);
        }
        if (!rows) return {result, status_t::success_k};

        auto [audit, reserve_status] = reserve_audit_(context, rows);
        if (failed(reserve_status)) return {result, reserve_status};
        u32_t islands = 0;
        auto survey = [&]() noexcept {
            auto const [unreached, audit_status] = audit_(context, audit);
            islands = unreached;
            result.sweeps += audit.sweeps;
            return audit_status;
        };
        if (status_t const audited = survey(); failed(audited)) return {result, audited};
        result.islands_found = islands;
        for (u32_t islands_before = islands; islands;) {
            // The island IDs replace the range: offsets below their count index them.
            auto const [healed, healed_status] = refine_pass_(context, config, {0, islands}, audit.island_ids);
            if (failed(healed_status)) return {result, healed_status};
            add_walks(healed);
            if (status_t const audited = survey(); failed(audited)) return {result, audited};
            if (!islands) break;
            // The reached in-links first, so no reconnection takes the last link a reached node is found through.
            u32_t* const neighbors = reinterpret_cast<u32_t*>(neighbors_plane_.base());
            if (cudaMemsetAsync(audit.in_links, 0, rows * sizeof(u32_t), context.stream()) != cudaSuccess)
                return {result, status_t::device_error_k};
            if (status_t const counted = cuda_launch(
                    dense_cuda_reached_in_links_kernel,
                    static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k,
                    block_threads_k, 0, context.stream(), static_cast<u32_t const*>(neighbors),
                    static_cast<u32_t>(rows), static_cast<u32_t>(connectivity()), audit.reached, audit.in_links);
                failed(counted))
                return {result, counted};
            if (status_t const launched = cuda_launch(
                    dense_cuda_reconnect_island_kernel, divide_round_up<u32_t>(islands, block_threads_k),
                    block_threads_k, 0, context.stream(), neighbors, static_cast<u32_t>(connectivity()),
                    audit.island_ids, islands, audit.reached, static_cast<u32_t>(rows),
                    static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows)), audit.in_links, audit.reconnected);
                failed(launched))
                return {result, launched};
            if (status_t const audited = survey(); failed(audited)) return {result, audited};
            // A round that frees no island frees none on the next either, since the picks it needs are unreached.
            if (islands >= islands_before) break;
            islands_before = islands;
        }
        auto const [reconnected, read_status] = read_counter_(audit.reconnected, context.stream());
        if (failed(read_status)) return {result, read_status};
        result.islands_reconnected = reconnected;
        result.reachable_nodes = rows - islands;
        return {result, status_t::success_k};
    }

    /** Measures forward reachability from as many stride-spread landmarks as one search draws -
     *  the connectivity; any shortfall against @c size() is the population no search can discover.
     *  Read-only on the graph, so concurrent insertions may leave fresh nodes uncounted. */
    expected<health_result_t> health(dense_traversal_context_t& context) noexcept {
        std::size_t const rows = size();
        if (!rows) return {{}, status_t::success_k};
        auto [audit, reserve_status] = reserve_audit_(context, 0);
        if (failed(reserve_status)) return {{}, reserve_status};
        auto const [unreached, audit_status] = audit_(context, audit);
        if (failed(audit_status)) return {{}, audit_status};
        return {{rows, rows - unreached, audit.sweeps}, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the stream its copies ride, and the batches
     *      it hands out - one pinned bounce of rows at a time, since a device plane cannot be read
     *      in place.
     *
     *  Vectors travel as logical rows with the fragment padding stripped, neighbors verbatim under
     *  the positional sentinel convention; norms re-derive on load.
     *
     *  One host thread drives a device, so batches are taken one after another, though the tallies
     *  keep the shape a threaded caller expects.
     */
    struct [[nodiscard]] save_plan_t {

        /** Where the rows land, vetted against the watermark at planning. */
        columns_t columns {};

        /** What an ID under the watermark moves by - the row this shard starts at. */
        node_id_t id_shift = 0;

        /** The stream every device-to-host copy queues on, drained before its bytes move. */
        cudaStream_t stream = nullptr;

        /** Page-locked host bytes: one batch of packed vector rows, then its neighbor IDs. */
        bounce_t bounce {cuda_pinned_allocator_t {}};

        /** Rows under the watermark - the claimed rows at planning. */
        node_id_t visible = 0;

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows = 0;

        /** Rows whose vectors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_neighbors = 0;

        save_plan_t() noexcept = default;

        /** The index snapshotted into @p columns on @p stream under the watermark at @p visible, its
         *  IDs shifted by @p id_shift; only @ref plan_save calls it, and sizes the bounce afterwards. */
        save_plan_t(dense_vram_index const& index, columns_t columns, node_id_t id_shift, cudaStream_t stream,
                    node_id_t visible) noexcept
            : columns(columns), id_shift(id_shift), stream(stream),
              bounce(cuda_pinned_allocator_t {stream, index.device_.ordinal}), visible(visible),
              batch_rows(min_of(static_cast<std::size_t>(visible), power_of_two<std::size_t>(bounce_rows_log2_k))),
              index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), id_shift(other.id_shift), stream(other.stream), bounce(std::move(other.bounce)),
              visible(other.visible), batch_rows(other.batch_rows), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;

        /** A save owns nothing but its bounce, so a dying plan leaves the index as it found it. */
        ~save_plan_t() noexcept = default;

        /** One step of the plan: one bounce-sized batch of the rows under the watermark. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_rows;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_rows;

            /** Bytes this step's vector rows occupy inside the plan's vectors column. */
            std::size_t vectors_bytes = 0;

            /** Bytes this step's neighbor rows occupy inside the plan's neighbors column. */
            std::size_t neighbors_bytes = 0;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * batch_rows;
            std::size_t const bound = min_of(first + batch_rows, static_cast<std::size_t>(visible));
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
            step.vectors_bytes = step.index_rows.size() * index_->packed_row_bytes();
            step.neighbors_bytes = step.index_rows.size() * index_->connectivity() * sizeof(node_id_t);
            return step;
        }

        /** The steps the save spans - one bounce-sized batch each. */
        indexed_range<save_plan_t> steps() const noexcept {
            return {this, index_ && batch_rows ? divide_round_up<std::size_t>(visible, batch_rows) : 0};
        }

        /** @brief Copies step @p step 's rows off the device into the plan's own columns: the
         *      vectors with the fragment padding stripped, then the neighbors with every ID under
         *      the watermark shifted by @c id_shift and every other folded to the sentinel.
         *
         *  @param[in] step The step to copy, as @ref steps hands them out.
         *  @param[inout] scratch Unused here, since the bounce is already host memory; it rides
         *      in the signature so a caller drives a device plan exactly as it drives a host one.
         */
        status_t save(step_t const& step, nk::matrix_span<node_id_t>) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const rows = step.index_rows;
            if (rows.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const step_slots = rows.size() * connectivity;
            std::size_t const vectors_first = step.source_rows.first * scalar_row_bytes;
            std::size_t const neighbors_first = step.source_rows.first * neighbors_row_bytes;
            if (step.vectors_bytes != rows.size() * scalar_row_bytes ||
                step.neighbors_bytes != step_slots * sizeof(node_id_t))
                return status_t::config_mismatch_k;
            if (vectors_first > columns.vectors.size() || neighbors_first > columns.neighbors.size() ||
                columns.vectors.size() - vectors_first < step.vectors_bytes ||
                columns.neighbors.size() - neighbors_first < step.neighbors_bytes)
                return status_t::config_mismatch_k;
            if (cudaMemcpy2DAsync(
                    bounce.data(), scalar_row_bytes,
                    index_->vectors_plane_.base() + static_cast<std::size_t>(rows.first) * index_->row_bytes(),
                    index_->row_bytes(), scalar_row_bytes, rows.size(), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return status_t::device_error_k;
            if (cudaStreamSynchronize(stream) != cudaSuccess) return status_t::device_error_k;
            [[maybe_unused]] status_t const copied_vectors = copy_bytes(
                columns.vectors.subspan(vectors_first, step.vectors_bytes),
                bytes_of(bounce.data(), step.vectors_bytes));
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(rows.size(), std::memory_order_acq_rel);

            byte_t* const bounce_neighbors = bounce.data() + batch_rows * scalar_row_bytes;
            if (cudaMemcpyAsync(
                    bounce_neighbors,
                    index_->neighbors_plane_.base() + static_cast<std::size_t>(rows.first) * neighbors_row_bytes,
                    rows.size() * neighbors_row_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
                return status_t::device_error_k;
            if (cudaStreamSynchronize(stream) != cudaSuccess) return status_t::device_error_k;
            node_id_t* const landed = reinterpret_cast<node_id_t*>(bounce_neighbors);
            for (std::size_t word = 0; word != step_slots; ++word)
                landed[word] = landed[word] < visible ? static_cast<node_id_t>(landed[word] + id_shift)
                                                      : dense_node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(
                columns.neighbors.subspan(neighbors_first, step.neighbors_bytes), bytes_of(landed, step_slots));
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(rows.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Pronounces the snapshot complete: every batch's vectors and neighbors landed. A success
         *  spends the plan; a refusal leaves it whole, so the missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != nodes ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** The index being snapshotted; null in the empty plan a refused @ref plan_save returns, and
         *  once @c finish succeeded. */
        dense_vram_index const* index_ = nullptr;
    };

    /**
     *  @brief Opens a save into @p columns under @p synchronization on @p stream: takes the watermark
     *      and sizes one bounce for it.
     *
     *  @c concurrent_k answers @c config_mismatch_k, since the device planes cannot be probed for
     *  settlement; @c exclusive_k reads like @c immutable_k. Columns too short for the rows under it
     *  are refused here, so no step ever writes past them.
     *
     *  @param[out] columns Where the rows land, sized from @ref saved_shape.
     *  @param[in] synchronization The contract the caller certifies.
     *  @param[in] stream The stream every copy queues on.
     *  @param[in] id_shift What an ID under the watermark moves by - the row this shard starts at.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization, cudaStream_t stream = nullptr,
                                    node_id_t id_shift = 0) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return {{}, status_t::config_mismatch_k};
        if (!vectors_plane_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        std::size_t const rows = size();
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t))
            return {{}, status_t::config_mismatch_k};
        save_plan_t plan {*this, columns, id_shift, stream, static_cast<node_id_t>(rows)};
        std::size_t const batch_bytes = plan.batch_rows * (packed_row_bytes() + connectivity() * sizeof(node_id_t));
        if (batch_bytes && failed(plan.bounce.resize_uninitialized(batch_bytes))) return {{}, status_t::bad_alloc_k};
        return {std::move(plan), status_t::success_k};
    }

    /** @brief Serializes every row under the watermark into @p columns on the calling thread - the one-shot over
     *      @ref plan_save for a caller with no streams to fan the steps out over. */
    status_t save(columns_t columns, synchronization_t synchronization, nk::matrix_span<node_id_t> scratch,
                  cudaStream_t stream = nullptr) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization, stream);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows claimed after the published ones, the planes grown to
     *      hold them, and the batches they span, each filled through one pinned bounce and
     *      uploaded on the stream; @ref finish derives the norms and publishes.
     *
     *  Holds the claim for its whole life, so nothing else claims rows while it runs. A source's
     *  IDs are local to the rows it delivers, so every batch shifts them onto the appended range
     *  and folds every other ID onto the sentinel. Repeated plans append, which is how several
     *  parts land in one shard. A plan that dies before @c finish claims nothing, drops the mutex,
     *  and leaves the planes grown for the next one.
     */
    struct [[nodiscard]] load_plan_t {

        /** Where the rows come from, vetted against them at planning. */
        columns_view_t columns {};

        /** The ID space the source's rows speak. */
        iota_view<node_id_t> source_range {};

        /** The stream every host-to-device copy and the norms kernel queue on. */
        cudaStream_t stream = nullptr;

        /** Page-locked host bytes: one batch of packed vector rows, then its neighbor IDs. */
        bounce_t bounce {cuda_pinned_allocator_t {}};

        /** The first appended ID - the row the claim landed at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows one batch carries - a bounce's worth, or every row when fewer. */
        std::size_t batch_rows = 0;

        /** Rows whose vectors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c std::atomic_ref. */
        std::size_t covered_neighbors = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base on @p stream, and the claim held meanwhile - unowned
         *  for a plan of no rows; only @ref plan_load calls it, and sizes the bounce afterwards. */
        load_plan_t(dense_vram_index& index, columns_view_t columns, iota_view<node_id_t> source_range,
                    cudaStream_t stream, std::unique_lock<fu::spin_mutex_t>&& claiming, node_id_t base,
                    std::size_t rows) noexcept
            : columns(columns), source_range(source_range), stream(stream),
              bounce(cuda_pinned_allocator_t {stream, index.device_.ordinal}), base(base), rows(rows),
              batch_rows(min_of(rows, power_of_two<std::size_t>(bounce_rows_log2_k))), index_(&index),
              claiming_(std::move(claiming)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), source_range(other.source_range), stream(other.stream),
              bounce(std::move(other.bounce)), base(other.base), rows(std::exchange(other.rows, 0)),
              batch_rows(other.batch_rows), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), index_(std::exchange(other.index_, nullptr)),
              claiming_(std::move(other.claiming_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;

        /** The lock's own death drops the claim, unless @c finish already did. */
        ~load_plan_t() noexcept = default;

        /** One step of the plan: one bounce-sized batch of the appended rows. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a stream. */
            std::size_t position = 0;

            /** Where the rows land in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_rows;

            /** Where they come from in the caller's buffer, from its first row. */
            iota_view<node_id_t> source_rows;

            /** Bytes this step's vector rows occupy inside the plan's vectors column. */
            std::size_t vectors_bytes = 0;

            /** Bytes this step's neighbor rows occupy inside the plan's neighbors column. */
            std::size_t neighbors_bytes = 0;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = static_cast<std::size_t>(base) + position * batch_rows;
            std::size_t const bound = min_of(first + batch_rows, static_cast<std::size_t>(base) + rows);
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            step.vectors_bytes = step.index_rows.size() * index_->packed_row_bytes();
            step.neighbors_bytes = step.index_rows.size() * index_->connectivity() * sizeof(node_id_t);
            return step;
        }

        /** The steps the append spans - one bounce-sized batch each. */
        indexed_range<load_plan_t> steps() const noexcept {
            return {this, index_ && batch_rows ? divide_round_up(rows, batch_rows) : 0};
        }

        /** @brief Copies step @p step 's rows out of the plan's own columns through the bounce:
         *      the vectors into the fragment-padded plane rows, whose fresh tails arrive zeroed,
         *      then the neighbors with every ID inside @c source_range folded onto the appended
         *      range and every other onto the sentinel.
         *
         *  @param[in] step The step to fill, as @ref steps hands them out.
         */
        status_t load(step_t const& step) noexcept { return load(step, columns, source_range); }

        /** @brief The same copy out of @p columns under @p source_range rather than the plan's
         *      own - what a sharded plan drives, one part's slice at a time, its step naming the
         *      rows in that part's row and ID spaces.
         *
         *  @param[in] step The step to fill; its source rows address @p columns.
         *  @param[in] columns The part's own planes.
         *  @param[in] source_range The ID space the part's rows speak.
         */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty()) return status_t::config_mismatch_k;
            std::size_t const scalar_row_bytes = index_->packed_row_bytes();
            std::size_t const connectivity = index_->connectivity();
            std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
            std::size_t const step_slots = range.size() * connectivity;
            std::size_t const vectors_first = step.source_rows.first * scalar_row_bytes;
            std::size_t const neighbors_first = step.source_rows.first * neighbors_row_bytes;
            if (step.vectors_bytes != range.size() * scalar_row_bytes ||
                step.neighbors_bytes != step_slots * sizeof(node_id_t))
                return status_t::config_mismatch_k;
            if (vectors_first > columns.vectors.size() || neighbors_first > columns.neighbors.size() ||
                columns.vectors.size() - vectors_first < step.vectors_bytes ||
                columns.neighbors.size() - neighbors_first < step.neighbors_bytes)
                return status_t::config_mismatch_k;
            // The part's rows land at this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            // The device indexes its buckets by neighbor ID unchecked, so every rebased ID must land in this plan.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;
            [[maybe_unused]] status_t const staged_vectors = copy_bytes(
                bytes_of(bounce.data(), step.vectors_bytes),
                columns.vectors.subspan(vectors_first, step.vectors_bytes));
            if (cudaMemcpy2DAsync(
                    index_->vectors_plane_.base() + static_cast<std::size_t>(range.first) * index_->row_bytes(),
                    index_->row_bytes(), bounce.data(), scalar_row_bytes, scalar_row_bytes, range.size(),
                    cudaMemcpyHostToDevice, stream) != cudaSuccess)
                return index_->fail_(status_t::device_error_k);
            if (cudaStreamSynchronize(stream) != cudaSuccess) return index_->fail_(status_t::device_error_k);
            standard_atomic_ref<std::size_t> {covered_vectors}.fetch_add(range.size(), std::memory_order_acq_rel);

            byte_t* const bounce_neighbors = bounce.data() + batch_rows * scalar_row_bytes;
            [[maybe_unused]] status_t const staged_neighbors = copy_bytes(
                bytes_of(bounce_neighbors, step.neighbors_bytes),
                columns.neighbors.subspan(neighbors_first, step.neighbors_bytes));
            node_id_t* const landed = reinterpret_cast<node_id_t*>(bounce_neighbors);
            rebase_neighbors(std::span<node_id_t> {landed, step_slots}, source_range, shift, dense_node_id_unknown_k);
            compact_neighbors(std::span<node_id_t> {landed, step_slots}, connectivity, range.first,
                              dense_node_id_unknown_k);
            if (cudaMemcpyAsync(
                    index_->neighbors_plane_.base() + static_cast<std::size_t>(range.first) * neighbors_row_bytes,
                    bounce_neighbors, range.size() * neighbors_row_bytes, cudaMemcpyHostToDevice,
                    stream) != cudaSuccess)
                return index_->fail_(status_t::device_error_k);
            if (cudaStreamSynchronize(stream) != cudaSuccess) return index_->fail_(status_t::device_error_k);
            standard_atomic_ref<std::size_t> {covered_neighbors}.fetch_add(range.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** Publishes the appended rows once every batch's vectors and neighbors landed: the norms
         *  derive on the device, the row count lands last, and the claim lifts. A success spends
         *  the plan, so a second call answers @c uninitialized_k instead of publishing twice; a
         *  refusal leaves it whole, so the missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (standard_atomic_ref<std::size_t> {covered_vectors}.load(std::memory_order_acquire) != rows ||
                standard_atomic_ref<std::size_t> {covered_neighbors}.load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                u32_t const blocks = static_cast<u32_t>(index_->device_.multiprocessor_count) *
                                     search_blocks_per_multiprocessor_k;
                if (failed(cuda_launch(dense_cuda_derive_norms_kernel<kit_t>, blocks, block_threads_k, 0, stream,
                                       index_->planes(), static_cast<u32_t>(base), static_cast<u32_t>(rows))))
                    return index_->fail_(status_t::device_error_k);
                u32_t const restored = static_cast<u32_t>(base + rows);
                if (cudaMemcpyAsync(index_->published_rows_.get(), &restored, sizeof(restored), cudaMemcpyHostToDevice,
                                    stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess)
                    return index_->fail_(status_t::device_error_k);
                standard_atomic_ref<std::size_t>(index_->claimed_rows_).store(base + rows, std::memory_order_release);
                claiming_.unlock();
            }
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** The index appended to; null in the empty plan a refused @ref plan_load returns, and once
         *  @c finish published. */
        dense_vram_index* index_ = nullptr;

        /** The claim held from planning to publication, dropped by its own death otherwise. */
        std::unique_lock<fu::spin_mutex_t> claiming_;
    };

    /**
     *  @brief Opens a load of @p rows from @p columns after the published ones on @p stream: takes
     *      the claim, grows the planes to hold them - the fresh neighbor rows arrive
     *      sentinel-filled - and sizes one bounce.
     *
     *  Refuses an unreserved index, a kept failure, columns shorter than the rows, a claim already
     *  held, and rows past the ID space.
     *
     *  @param[in] columns Where the rows come from, both planes sized for @p rows.
     *  @param[in] rows Rows to append.
     *  @param[in] source_range The ID space the source's rows speak.
     *  @param[in] stream The stream every copy and the norms kernel queue on.
     */
    expected<load_plan_t> plan_load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                                    cudaStream_t stream = nullptr) noexcept {
        if (!vectors_plane_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (columns.vectors.size() < rows * packed_row_bytes() ||
            columns.neighbors.size() < rows * connectivity() * sizeof(node_id_t))
            return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_load(rows, stream);
        if (failed(status)) return {{}, status};
        plan.columns = columns;
        plan.source_range = source_range;
        return {std::move(plan), status};
    }

    /** Opens a load of @p rows the caller sources per step - what a sharded plan drives, its parts
     *  arriving one slice at a time rather than as one pair of columns. */
    expected<load_plan_t> plan_load(std::size_t rows, cudaStream_t stream = nullptr) noexcept {
        if (!vectors_plane_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        if (!rows)
            return {load_plan_t {*this, {}, {}, stream, {}, static_cast<node_id_t>(size()), 0}, status_t::success_k};
        if (!claiming_.lock_if_free()) return {{}, status_t::contended_k};
        std::unique_lock<fu::spin_mutex_t> claiming {claiming_, std::adopt_lock};
        // Read under the claim: a batch claiming between an earlier read and the claim would
        // overlap the loaded rows, and `finish` would then lower the claim beneath it.
        node_id_t const base = static_cast<node_id_t>(size());
        if (rows > max_node_count_k - base) return {{}, status_t::overflow_k};
        load_plan_t plan {*this, {}, {}, stream, std::move(claiming), base, rows};
        std::size_t const batch_bytes = plan.batch_rows * (packed_row_bytes() + connectivity() * sizeof(node_id_t));
        if (failed(plan.bounce.resize_uninitialized(batch_bytes))) return {{}, status_t::bad_alloc_k};
        if (status_t const grown = grow_rows_(base + rows, stream); failed(grown)) return {{}, grown};
        return {std::move(plan), status_t::success_k};
    }

    /** @brief Appends the whole source on the calling thread - the one-shot over @ref plan_load
     *      for a caller with no streams to fan the steps out over. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range,
                  cudaStream_t stream = nullptr) noexcept {
        auto [plan, status] = plan_load(columns, rows, source_range, stream);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

  private:
#pragma region Insertion

    /** The submission half of @c insert: claims, stages, launches, and returns with the IDs, which
     *  are dense and known at claim time. Back-to-back submissions on one context are serialized by
     *  its stream; across devices they overlap freely. */
    expected<insert_result_t> submit_insert_(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                             dense_traversal_context_t& context, insert_config_t config) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        if (!rows) return {{size(), static_cast<dense_node_id_t>(size())}, status_t::success_k};
        if (!vectors_plane_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (new_vectors.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {{}, status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (!build_scratch_fits()) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (new_vectors.stride_bytes(0) < static_cast<std::ptrdiff_t>(scalar_row_bytes))
            return {{}, status_t::unsupported_layout_k};
        cudaStream_t const stream = context.stream();
        // Every fallible preparation precedes the claim: a claimed range that never publishes would
        // starve every later batch's admission ramp.
        byte_t* const staging = context.ensure_staging(rows * row_bytes());
        if (!staging) return {{}, status_t::bad_alloc_k};
        if (scalar_row_bytes != row_bytes() && cudaMemsetAsync(staging, 0, rows * row_bytes(), stream) != cudaSuccess)
            return {{}, status_t::device_error_k};
        // The source pitch is the caller's stride, so a strided view uploads without being gathered first.
        if (cudaMemcpy2DAsync(staging, row_bytes(), new_vectors.byte_data(), new_vectors.stride_bytes(0),
                              scalar_row_bytes, rows, cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const pool_slots = pool_slots_for_(config.expansion_factor, 4);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32,
                                                                     frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        insert_blocks_per_multiprocessor_k * dense_cuda_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};

        // Grow, claim and launch while holding the claiming mutex: the claim is stored only once
        // the launch is accepted, so a rejected grid, which publishes nothing, is un-claimed exactly.
        std::unique_lock claiming {claiming_};
        std::size_t const first_id = standard_atomic_ref<std::size_t>(claimed_rows_).load(std::memory_order_relaxed);
        std::size_t const total_rows = first_id + rows;
        status_t claimed = grow_rows_(total_rows, stream);
        if (claimed == status_t::success_k) {
            // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm arena.
            if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + rows > 0xFFFFFFFEull) {
                if (cudaMemsetAsync(arena, 0, context.arena_bytes_, stream) != cudaSuccess) {
                    claimed = status_t::device_error_k;
                    context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                }
                context.epoch_floor_ = 0;
            }
            u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
            context.epoch_floor_ += rows; // Monotone; an un-claim below only forfeits epoch space.
            if (claimed == status_t::success_k &&
                (cudaMemsetAsync(context.assigned_rows_.get(), 0, sizeof(u32_t), stream) != cudaSuccess ||
                 cudaMemsetAsync(context.telemetry_.get(), 0, sizeof(dense_cuda_telemetry_block_t), stream) !=
                     cudaSuccess))
                claimed = status_t::device_error_k;
            if (claimed == status_t::success_k) {
                dense_cuda_build_view<kit_t> view {planes(), //
                                                   reinterpret_cast<fragment_t const*>(staging),
                                                   static_cast<u32_t>(total_rows),
                                                   insert_scratch_bytes_per_warp(),
                                                   static_cast<dense_node_id_t>(first_id),
                                                   arena,
                                                   arena_shape,
                                                   epoch_base,
                                                   published_rows_.get(),
                                                   context.telemetry_.get(),
                                                   context.assigned_rows_.get()};
                // The launch is the last fallible step; reading its verdict here keeps the claim reversible.
                if (failed(launch_insert_(view, pool_slots, stream))) claimed = status_t::device_error_k;
            }
            if (claimed == status_t::success_k)
                standard_atomic_ref<std::size_t>(claimed_rows_).store(total_rows, std::memory_order_release);
        }
        claiming.unlock();
        if (claimed != status_t::success_k) return {{}, claimed};

        insert_result_t result;
        result.first_id = static_cast<dense_node_id_t>(first_id);
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<dense_node_id_t>(first_id + row);
        result.new_size = total_rows;
        return {result, status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Search

    /** What every walking verb stages ahead of its sink: the device view, the frontier the walks
     *  keep, the staging past the uploads, and the rescoring the walks score under. */
    template <typename rescoring_type_>
    struct walk_launch {

        /** The device view every walk kernel reads. */
        dense_cuda_search_view<kit_t> view;

        /** The register ladder's width, when the ladder serves. */
        u32_t pool_slots;

        /** Which frontier the walks keep. */
        frontier_kind_t frontier;

        /** The staging bytes past the queries and any heads, where the sink writes its outputs. */
        byte_t* outputs;

        /** The rescoring of every walk, over the heads staged beside the queries. */
        rescoring_type_ rescoring;
    };

    /** The device rescoring the walks under @p heads_type_ score with. */
    template <typename heads_type_>
    using walk_rescoring = std::conditional_t<std::is_same_v<heads_type_, dense_indexer_heads<scalar_t>>,
                                              dense_cuda_indexer_rescoring<kit_t>, dense_cuda_no_rescoring>;

    /** Validates and uploads @p queries, then any indexer @p heads and their weights, sizes the arena for
     *  @p expansion, and reserves @p outputs_bytes of staging past them: everything a walking verb stages before
     *  its sink. Until the context drains, the queries, the heads and the context stay live. */
    template <typename heads_type_>
    expected<walk_launch<walk_rescoring<heads_type_>>> stage_walk_(vectors_view_t queries,
                                                                   [[maybe_unused]] heads_type_ const& heads,
                                                                   dense_traversal_context_t& context, u32_t expansion,
                                                                   u32_t minimum_slots,
                                                                   std::size_t outputs_bytes) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, dense_indexer_heads<scalar_t>>;
        std::size_t const queries_count = queries.extent(0);
        if (!vectors_plane_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (queries.extent(1) != dimensions()) return {{}, status_t::config_mismatch_k};
        // Validated with the rest: past here a refusal would already have burned epoch space.
        if (queries.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
            return {{}, status_t::unsupported_layout_k};
        std::size_t heads_rows = 0;
        if constexpr (indexer_k) {
            heads_rows = heads.heads.extent(0);
            if (heads.weights.extent(0) != queries_count || heads_rows != queries_count * heads.weights.extent(1) ||
                heads.heads.extent(1) != dimensions())
                return {{}, status_t::config_mismatch_k};
            if (heads.heads.stride_bytes(0) < static_cast<std::ptrdiff_t>(packed_row_bytes()))
                return {{}, status_t::unsupported_layout_k};
        }
        cudaStream_t const stream = context.stream();

        // Past the register ladder's widest pool the arena-backed walk serves.
        frontier_kind_t const frontier = expansion > dense_cuda_sorted_pool<16>::capacity_k
                                             ? frontier_kind_t::wide_arena_k
                                             : frontier_kind_t::register_ladder_k;
        u32_t const pool_slots = pool_slots_for_(expansion, minimum_slots);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(
            frontier == frontier_kind_t::wide_arena_k ? expansion : pool_slots * 32, frontier);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        search_blocks_per_multiprocessor_k * dense_cuda_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // The staging holds the query rows and any head rows first, since they carry the widest alignment, then
        // the heads' weights, then the outputs; stream order serializes its reuse across calls.
        std::size_t const rows_bytes = (queries_count + heads_rows) * row_bytes();
        std::size_t const weights_bytes = heads_rows * sizeof(f32_t);
        byte_t* const staging = context.ensure_staging(rows_bytes + weights_bytes + outputs_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + queries_count > 0xFFFFFFFEull) {
            if (cudaMemsetAsync(arena, 0, context.arena_bytes_, stream) != cudaSuccess) {
                context.arena_bytes_ = 0; // Spent: a retry must re-zero, never walk epoch garbage.
                return {{}, status_t::device_error_k};
            }
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += queries_count;
        // Pitched uploads: device rows pad to whole fragments, and the pad bytes must be zero for
        // the zero-tail distance identity to hold.
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (scalar_row_bytes != row_bytes() && cudaMemsetAsync(staging, 0, rows_bytes, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};
        auto upload_rows = [&](byte_t* destination, vectors_view_t rows) noexcept {
            return cudaMemcpy2DAsync(destination, row_bytes(), rows.byte_data(), rows.stride_bytes(0), scalar_row_bytes,
                                     rows.extent(0), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        };
        if (!upload_rows(staging, queries)) return {{}, status_t::device_error_k};
        if (cudaMemsetAsync(context.telemetry_.get(), 0, sizeof(dense_cuda_telemetry_block_t), stream) != cudaSuccess)
            return {{}, status_t::device_error_k};

        walk_rescoring<heads_type_> rescoring {};
        if constexpr (indexer_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            byte_t* const staged_weights = staging + rows_bytes;
            std::size_t const weights_row_bytes = heads.weights.extent(1) * sizeof(f32_t);
            if (!upload_rows(staged_heads, heads.heads) ||
                cudaMemcpy2DAsync(staged_weights, weights_row_bytes, heads.weights.byte_data(),
                                  heads.weights.stride_bytes(0), weights_row_bytes, queries_count, cudaMemcpyDefault,
                                  stream) != cudaSuccess)
                return {{}, status_t::device_error_k};
            rescoring = {reinterpret_cast<fragment_t const*>(staged_heads),
                         reinterpret_cast<f32_t const*>(staged_weights), static_cast<u32_t>(heads.weights.extent(1)),
                         row_fragments()};
        }

        dense_cuda_search_view<kit_t> const view {planes(),
                                                  reinterpret_cast<fragment_t const*>(staging),
                                                  static_cast<u32_t>(queries_count),
                                                  static_cast<u32_t>(size()),
                                                  static_cast<u32_t>(row_bytes()),
                                                  expansion,
                                                  arena,
                                                  arena_shape,
                                                  epoch_base,
                                                  context.telemetry_.get()};
        return {{view, pool_slots, frontier, staging + rows_bytes + weights_bytes, rescoring}, status_t::success_k};
    }

    /** Blocking drains the context's stream and adds the telemetry the walks spent; deferred returns
     *  the submission as queued. */
    expected<search_result_t> settle_(expected<search_result_t> submitted, std::size_t rows,
                                      dense_traversal_context_t& context, completion_t completion) noexcept {
        if (completion == completion_t::deferred_k || !submitted || !rows) return submitted;
        if (status_t const drained = context.drain(); failed(drained)) return {{}, drained};
        dense_cuda_telemetry_block_t const spent = context.spent();
        submitted.value.visited_members = spent.hops;
        submitted.value.computed_distances = spent.scored;
        submitted.value.restarts = spent.restarts;
        submitted.value.tail_drops = spent.tail_drops;
        submitted.value.visited_saturations = spent.visited_saturations;
        return submitted;
    }

    /** The submission half of @c search: stages the queries and @p heads, launches, and queues the pitched
     *  writeback. Until the context drains, the queries, the heads, both outputs and the context stay live;
     *  across contexts they must not share a buffer at all. */
    template <typename heads_type_>
    expected<search_result_t> submit_search_(vectors_view_t queries, heads_type_ const& heads,
                                             nk::matrix_span<dense_node_id_t> result_ids,
                                             nk::matrix_span<dense_distance_t> result_distances,
                                             dense_traversal_context_t& context, search_config_t config) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (!queries_count) return {{}, status_t::success_k};
        bool const wants_distances = result_distances.extent(0) != 0;
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count || config.count > 512 ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {{}, status_t::config_mismatch_k};
        if (result_ids.stride_bytes(0) < static_cast<std::ptrdiff_t>(config.count * sizeof(dense_node_id_t)) ||
            (wants_distances &&
             result_distances.stride_bytes(0) < static_cast<std::ptrdiff_t>(config.count * sizeof(dense_distance_t))))
            return {{}, status_t::unsupported_layout_k};

        // Past the register ladder's widest pool the ranks cover the arena frontier's head of 64.
        u32_t const expansion = config.expansion_factor > config.count ? config.expansion_factor : config.count * 2;
        if (expansion > dense_cuda_sorted_pool<16>::capacity_k && config.count > dense_cuda_sorted_pool<2>::capacity_k)
            return {{}, status_t::config_mismatch_k};
        // The pool must cover the requested count, whatever the expansion asked for.
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        std::size_t const results_count = queries_count * config.count;
        std::size_t const ids_bytes = results_count * sizeof(dense_node_id_t);
        std::size_t const distances_bytes = wants_distances ? results_count * sizeof(dense_distance_t) : 0;
        auto const staged = stage_walk_(queries, heads, context, expansion, count_slots > 2 ? count_slots : 2,
                                        ids_bytes + distances_bytes);
        if (!staged) return {{}, staged.status};
        dense_node_id_t* const device_ids = reinterpret_cast<dense_node_id_t*>(staged.value.outputs);
        dense_distance_t* const device_distances =
            wants_distances ? reinterpret_cast<dense_distance_t*>(staged.value.outputs + ids_bytes) : nullptr;
        cudaStream_t const stream = context.stream();
        if (status_t const launched = launch_walk_(
                staged.value, dense_cuda_search_sink<kit_t> {device_ids, device_distances, config.count}, stream);
            failed(launched))
            return {{}, launched};
        // The landing may be wider than the count, so a multi-index can hand every shard its own
        // columns of one results matrix.
        if (cudaMemcpy2DAsync(result_ids.data(), result_ids.stride_bytes(0), device_ids,
                              config.count * sizeof(dense_node_id_t), config.count * sizeof(dense_node_id_t),
                              queries_count, cudaMemcpyDefault, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};
        if (wants_distances &&
            cudaMemcpy2DAsync(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                              config.count * sizeof(dense_distance_t), config.count * sizeof(dense_distance_t),
                              queries_count, cudaMemcpyDefault, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};
        return {{config.count}, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows, launches, and queues the writeback of the
     *  kept pairs and the scored counts. */
    expected<search_result_t> submit_multiply_(vectors_view_t left, nk::matrix_span<dense_node_id_t> node_ids,
                                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                               dense_traversal_context_t& context, search_config_t config) noexcept {
        std::size_t const rows_count = left.extent(0);
        std::size_t const wanted = node_ids.extent(1);
        if (node_ids.extent(0) != rows_count || products.extent(0) != rows_count || products.extent(1) != wanted ||
            counts.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (node_ids.stride_bytes(0) < static_cast<std::ptrdiff_t>(wanted * sizeof(dense_node_id_t)) ||
            products.stride_bytes(0) < static_cast<std::ptrdiff_t>(wanted * sizeof(dot_result_t)))
            return {{}, status_t::unsupported_layout_k};
        std::size_t const ids_bytes = rows_count * wanted * sizeof(dense_node_id_t);
        std::size_t const products_bytes = rows_count * wanted * sizeof(dot_t);
        auto const staged = stage_walk_(left, dummy_heads_t {}, context,
                                        static_cast<u32_t>(non_zero_expansion_factor(config)), 2,
                                        ids_bytes + products_bytes + rows_count * sizeof(u32_t));
        if (!staged) return {{}, staged.status};
        dense_node_id_t* const device_ids = reinterpret_cast<dense_node_id_t*>(staged.value.outputs);
        dot_t* const device_products = reinterpret_cast<dot_t*>(staged.value.outputs + ids_bytes);
        u32_t* const device_counts = reinterpret_cast<u32_t*>(staged.value.outputs + ids_bytes + products_bytes);
        cudaStream_t const stream = context.stream();
        dense_cuda_multiply_sink<kit_t> const sink {device_ids, device_products, device_counts,
                                                    static_cast<u32_t>(wanted)};
        if (status_t const launched = launch_walk_(staged.value, sink, stream); failed(launched)) return {{}, launched};
        if (wanted &&
            (cudaMemcpy2DAsync(node_ids.data(), node_ids.stride_bytes(0), device_ids, wanted * sizeof(dense_node_id_t),
                               wanted * sizeof(dense_node_id_t), rows_count, cudaMemcpyDefault,
                               stream) != cudaSuccess ||
             cudaMemcpy2DAsync(products.data(), products.stride_bytes(0), device_products, wanted * sizeof(dot_t),
                               wanted * sizeof(dot_t), rows_count, cudaMemcpyDefault, stream) != cudaSuccess))
            return {{}, status_t::device_error_k};
        if (cudaMemcpyAsync(counts.data(), device_counts, rows_count * sizeof(u32_t), cudaMemcpyDefault, stream) !=
            cudaSuccess)
            return {{}, status_t::device_error_k};
        return {{}, status_t::success_k};
    }

    /** The submission half of @c attend: validates the picks' width and the values' residency, stages the
     *  queries, launches, and queues the writeback of the picks and the carry. */
    expected<search_result_t> submit_attend_(vectors_view_t queries, vectors_view_t values,
                                             nk::matrix_span<dense_node_id_t> attended, nk::matrix_span<f32_t> outputs,
                                             std::span<dense_softmax_state_t> carry, f32_t score_scale,
                                             dense_traversal_context_t& context, search_config_t config) noexcept {
        std::size_t const rows_count = queries.extent(0);
        if (!rows_count) return {{}, status_t::success_k};
        if (attended.extent(0) != rows_count || attended.extent(1) > 512 || outputs.extent(0) != rows_count ||
            outputs.extent(1) != values.extent(1) || carry.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        u32_t const count = static_cast<u32_t>(attended.extent(1));
        if (attended.stride_bytes(0) < static_cast<std::ptrdiff_t>(count * sizeof(dense_node_id_t)) ||
            outputs.stride_bytes(0) % static_cast<std::ptrdiff_t>(sizeof(f32_t)))
            return {{}, status_t::unsupported_layout_k};
        // The warp loads value rows a whole fragment at a time.
        if ((reinterpret_cast<std::uintptr_t>(values.byte_data()) |
             static_cast<std::uintptr_t>(values.stride_bytes(0))) %
            sizeof(typename kit_t::fragment_t))
            return {{}, status_t::unsupported_layout_k};
        if (values.extent(0) < size()) return {{}, status_t::shape_mismatch_k};
        // The warp gathers values and writes outputs directly, so neither may live off this device.
        if (!is_resident_on(values.byte_data(), device_.ordinal) || !is_resident_on(outputs.data(), device_.ordinal))
            return {{}, status_t::device_error_k};
        // As in search, past the register ladder's widest pool the picks come from the arena frontier's head of 64.
        u32_t const expansion = config.expansion_factor > count ? config.expansion_factor : count * 2;
        if (expansion > dense_cuda_sorted_pool<16>::capacity_k && count > dense_cuda_sorted_pool<2>::capacity_k)
            return {{}, status_t::config_mismatch_k};
        u32_t const count_slots = divide_round_up<u32_t>(count, 32);
        std::size_t const ids_bytes = rows_count * count * sizeof(dense_node_id_t);
        std::size_t const states_bytes = rows_count * sizeof(dense_softmax_state_t);
        auto const staged = stage_walk_(queries, dummy_heads_t {}, context, expansion,
                                        count_slots > 2 ? count_slots : 2, ids_bytes + states_bytes);
        if (!staged) return {{}, staged.status};
        dense_node_id_t* const device_ids = reinterpret_cast<dense_node_id_t*>(staged.value.outputs);
        dense_softmax_state_t* const device_states = reinterpret_cast<dense_softmax_state_t*>(staged.value.outputs +
                                                                                              ids_bytes);
        cudaStream_t const stream = context.stream();
        dense_cuda_attend_sink<kit_t> const sink {
            values,         device_ids,
            outputs.data(), static_cast<std::size_t>(outputs.stride_bytes(0)) / sizeof(f32_t),
            device_states,  score_scale,
            count};
        if (status_t const launched = launch_walk_(staged.value, sink, stream); failed(launched)) return {{}, launched};
        if (count &&
            cudaMemcpy2DAsync(attended.data(), attended.stride_bytes(0), device_ids, count * sizeof(dense_node_id_t),
                              count * sizeof(dense_node_id_t), rows_count, cudaMemcpyDefault, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};
        if (cudaMemcpyAsync(carry.data(), device_states, states_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};
        return {{count}, status_t::success_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** The three-pass refinement engine behind @c refine(): the walk covers @p nodes, or under a device-resident
     *  @p node_ids the listed IDs with @p nodes spanning the array's offsets; the reverse passes re-select the walked
     *  rows and the rows they pick, and leave every other row as it was. */
    expected<refine_result_t> refine_pass_(dense_traversal_context_t& context, refine_config_t config,
                                           iota_view<dense_node_id_t> nodes, u32_t const* node_ids) noexcept {
        std::size_t const nodes_count = size();
        if (!nodes_count) return {{}, status_t::success_k};
        if (!vectors_plane_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        if (!build_scratch_fits()) return {{}, status_t::config_mismatch_k};
        if (status_t const kept = status(); failed(kept)) return {{}, kept};
        // IDs at or past the published count are unwalkable; an ID array is bounded by its own length.
        if (!node_ids) nodes.bound = static_cast<dense_node_id_t>(min_of<std::size_t>(nodes.bound, nodes_count));
        if (nodes.first >= nodes.bound) return {{}, status_t::success_k};
        std::size_t const walk_count = nodes.size();
        cudaStream_t const stream = context.stream();
        // Pass-local planes no reader observes: the shadow neighbors, the reverse buckets, their counters, and the
        // bitset of rows the merge re-selects; the bucket width is the connectivity.
        std::size_t const shadow_bytes = nodes_count * connectivity() * sizeof(u32_t);
        std::size_t const counters_bytes = (nodes_count + divide_round_up<std::size_t>(nodes_count, 32)) *
                                           sizeof(u32_t);
        u32_t const bucket_width = static_cast<u32_t>(connectivity());
        auto shadow = cuda_make_unique(context.allocator_, shadow_bytes * 2 + counters_bytes);
        if (!shadow) return {{}, status_t::bad_alloc_k};
        u32_t* const shadow_neighbors = reinterpret_cast<u32_t*>(shadow.get());
        u32_t* const bucket_ids = reinterpret_cast<u32_t*>(shadow.get() + shadow_bytes);
        u32_t* const bucket_counts = reinterpret_cast<u32_t*>(shadow.get() + shadow_bytes * 2);
        dense_cuda_bitset_t const touched {bucket_counts + nodes_count};
        if (cudaMemsetAsync(bucket_counts, 0, counters_bytes, stream) != cudaSuccess)
            return {{}, status_t::device_error_k};

        // Twice the connectivity by default, like every CPU walk, rounded up to the pool sizes the kernel carries.
        u32_t const pool_slots = pool_slots_for_(static_cast<u32_t>(non_zero_expansion_factor(config)), 4);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32,
                                                                     frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.multiprocessor_count) *
                                        insert_blocks_per_multiprocessor_k * dense_cuda_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // Only a fresh allocation or a wrapped 32-bit epoch space warrants zeroing, never a warm arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + walk_count > 0xFFFFFFFEull) {
            cudaMemsetAsync(arena, 0, context.arena_bytes_, stream);
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += walk_count;
        cudaMemsetAsync(context.assigned_rows_.get(), 0, sizeof(u32_t), stream);
        cudaMemsetAsync(context.telemetry_.get(), 0, sizeof(dense_cuda_telemetry_block_t), stream);

        dense_cuda_refine_view<kit_t> view {planes(), //
                                            shadow_neighbors,
                                            static_cast<u32_t>(nodes.bound),
                                            insert_scratch_bytes_per_warp(),
                                            static_cast<dense_node_id_t>(nodes.first),
                                            arena,
                                            arena_shape,
                                            epoch_base,
                                            context.telemetry_.get(),
                                            context.assigned_rows_.get(),
                                            node_ids,
                                            static_cast<u32_t>(nodes_count)};
        // A walk short of the whole graph starts from the live neighbors copied into the shadow verbatim, so a row
        // it never rewrites but one of its picks lands in merges its own neighbors with the new reverse links.
        if (node_ids || walk_count != nodes_count)
            if (cudaMemcpyAsync(shadow_neighbors, neighbors_plane_.base(), shadow_bytes, cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess)
                return {{}, status_t::device_error_k};
        if (status_t const launched = launch_refine_(view, pool_slots, stream); failed(launched)) return {{}, launched};
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {{}, status_t::device_error_k};
        // The assigned count is the completion oracle: every offset assigned means every shadow row landed.
        u32_t assigned = 0;
        if (cudaMemcpy(&assigned, context.assigned_rows_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
            assigned < walk_count)
            return {{}, status_t::device_error_k};

        // Scatter the walked rows' shadow neighbors in reverse, then merge each touched node's forward & reverse
        // candidates into its final live row. A heal round over island IDs re-selects the islands' own rows alone:
        // re-judging the rows they pick would drop the links that reconnected earlier islands.
        cudaMemsetAsync(context.assigned_rows_.get(), 0, sizeof(u32_t), stream);
        if (status_t const launched = cuda_launch(
                dense_cuda_scatter_links_kernel, device_.multiprocessor_count * insert_blocks_per_multiprocessor_k,
                block_threads_k, 0, stream, static_cast<u32_t const*>(shadow_neighbors),
                static_cast<u32_t>(nodes_count), static_cast<u32_t>(nodes.first), static_cast<u32_t>(walk_count),
                node_ids, static_cast<u32_t>(connectivity()), bucket_width, bucket_ids, bucket_counts, touched);
            failed(launched))
            return {{}, launched};
        dense_cuda_refine_view<kit_t> merge_view = view;
        merge_view.scratch_bytes_per_warp = merge_scratch_bytes_per_warp();
        // The merge sweeps the whole ID space and skips the rows the scatter left untouched.
        merge_view.node_ids = nullptr;
        merge_view.first_id = 0;
        merge_view.end_id = static_cast<u32_t>(nodes_count);
        if (status_t const launched = launch_merge_(merge_view, bucket_ids, bucket_counts, bucket_width, touched,
                                                    stream);
            failed(launched))
            return {{}, launched};
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {{}, status_t::device_error_k};
        if (cudaMemcpy(&assigned, context.assigned_rows_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
            assigned < nodes_count)
            return {{}, status_t::device_error_k};

        dense_cuda_telemetry_block_t const spent = context.spent();
        refine_result_t result;
        result.visited_nodes = walk_count;
        result.changed_nodes = spent.changed_nodes;
        result.visited_members = spent.hops;
        result.computed_distances = spent.scored;
        result.restarts = spent.restarts;
        return {result, status_t::success_k};
    }

    /** Device scratch one reachability audit sweeps in, reused across a heal's rounds - one allocation carved into
     *  the reached set and two frontiers, the expanded, unreached and reconnected counters, and the unreached rows' IDs
     *  and their in-link counts when a caller collects them. */
    struct audit_t {
        typename dense_traversal_context_t::bytes_ptr_t bytes;
        dense_cuda_bitset_t reached, frontier, next_frontier;
        u32_t* expanded = nullptr;
        u32_t* unreached = nullptr;
        u32_t* reconnected = nullptr;
        u32_t* island_ids = nullptr;
        u32_t* in_links = nullptr;
        std::size_t words = 0;

        /** Levels the last audit swept to convergence. */
        std::size_t sweeps = 0;
    };

    /** Allocates an audit over the rows published now, with room for @p island_slots unreached IDs and as many
     *  in-link counts, the reconnected counter zeroed. */
    expected<audit_t> reserve_audit_(dense_traversal_context_t& context, std::size_t island_slots) noexcept {
        audit_t audit {typename dense_traversal_context_t::bytes_ptr_t {
            nullptr, cuda_deleter<typename dense_traversal_context_t::allocator_t> {context.allocator_, 0}}};
        audit.words = divide_round_up<std::size_t>(size(), 32);
        std::size_t const bytes = (audit.words * 3 + 3 + island_slots * 2) * sizeof(u32_t);
        audit.bytes.reset(context.allocator_.allocate(bytes));
        audit.bytes.get_deleter().count = bytes;
        if (!audit.bytes) return {std::move(audit), status_t::bad_alloc_k};
        audit.reached.words = reinterpret_cast<u32_t*>(audit.bytes.get());
        audit.frontier.words = audit.reached.words + audit.words;
        audit.next_frontier.words = audit.frontier.words + audit.words;
        audit.expanded = audit.next_frontier.words + audit.words;
        audit.unreached = audit.expanded + 1;
        audit.reconnected = audit.unreached + 1;
        audit.island_ids = island_slots ? audit.reconnected + 1 : nullptr;
        audit.in_links = island_slots ? audit.island_ids + island_slots : nullptr;
        if (cudaMemsetAsync(audit.reconnected, 0, sizeof(u32_t), context.stream()) != cudaSuccess)
            return {std::move(audit), status_t::device_error_k};
        return {std::move(audit), status_t::success_k};
    }

    /** Sweeps reachability to convergence from as many stride-spread landmarks as one search draws - the
     *  connectivity - then counts the rows it never reached, collecting their IDs when @p audit has room for them. */
    expected<u32_t> audit_(dense_traversal_context_t& context, audit_t& audit) noexcept {
        std::size_t const rows = size();
        if (!vectors_plane_.reserved_bytes()) return {0, status_t::uninitialized_k};
        if (context.device() != device_.ordinal) return {0, status_t::config_mismatch_k};
        cudaStream_t const stream = context.stream();
        std::size_t const seed_count = min_of<std::size_t>(connectivity(), rows);
        // The bitsets, the expanded counter and the unreached counter lie back to back.
        if (cudaMemsetAsync(audit.reached.words, 0, (audit.words * 3 + 2) * sizeof(u32_t), stream) != cudaSuccess)
            return {0, status_t::device_error_k};
        dense_cuda_bitset_t frontier = audit.frontier, next_frontier = audit.next_frontier;
        if (status_t const launched = cuda_launch(
                dense_cuda_mark_seeds_kernel, divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads_k),
                block_threads_k, 0, stream, audit.reached, frontier, static_cast<u32_t>(rows),
                static_cast<u32_t>(seed_count));
            failed(launched))
            return {0, launched};

        // Every row joins one frontier at most, so the levels are bounded by the landmarks' eccentricity, capped
        // defensively.
        constexpr std::size_t sweeps_cap_k = 256;
        u32_t const grid = static_cast<u32_t>(device_.multiprocessor_count) * search_blocks_per_multiprocessor_k;
        for (audit.sweeps = 1; audit.sweeps != sweeps_cap_k; ++audit.sweeps) {
            if (cudaMemsetAsync(audit.expanded, 0, sizeof(u32_t), stream) != cudaSuccess ||
                cudaMemsetAsync(next_frontier.words, 0, audit.words * sizeof(u32_t), stream) != cudaSuccess)
                return {0, status_t::device_error_k};
            if (status_t const launched = cuda_launch(dense_cuda_reachability_sweep_kernel, grid, block_threads_k, 0,
                                                      stream, reinterpret_cast<u32_t const*>(neighbors_plane_.base()),
                                                      static_cast<u32_t>(rows), static_cast<u32_t>(connectivity()),
                                                      audit.reached, frontier, next_frontier, audit.expanded);
                failed(launched))
                return {0, launched};
            auto const [expanded_rows, read_status] = read_counter_(audit.expanded, stream);
            if (failed(read_status)) return {0, read_status};
            if (!expanded_rows) break;
            std::swap(frontier, next_frontier);
        }
        if (status_t const launched = cuda_launch(dense_cuda_reachability_islands_kernel,
                                                  divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads_k),
                                                  block_threads_k, 0, stream, audit.reached, static_cast<u32_t>(rows),
                                                  audit.island_ids, audit.unreached);
            failed(launched))
            return {0, launched};
        return read_counter_(audit.unreached, stream);
    }

    /** One device counter read back once the stream drains. */
    static expected<u32_t> read_counter_(u32_t const* counter, cudaStream_t stream) noexcept {
        u32_t value = 0;
        if (cudaMemcpyAsync(&value, counter, sizeof(u32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return {0, status_t::device_error_k};
        return {value, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Launches

    /** Insertion blocks per multiprocessor: the resident count saturates the path at its registers. */
    inline static constexpr u32_t insert_blocks_per_multiprocessor_k = dense_cuda_resident_blocks_per_multiprocessor_k;

    /** Search blocks per multiprocessor: oversubscribed, so enough queries stay in flight to reach
     *  the memory-level-parallelism knee. */
    inline static constexpr u32_t search_blocks_per_multiprocessor_k = 8;

    /** Threads per block, one warp's worth per resident warp. */
    inline static constexpr u32_t block_threads_k = dense_cuda_warps_per_block_k * 32;

    /** Keeps @p failure as the status unless one is kept already; returns it for the caller's own
     *  report. */
    status_t fail_(status_t failure) noexcept {
        status_t expected = status_t::success_k;
        standard_atomic_ref<status_t>(status_).compare_exchange_strong(expected, failure, std::memory_order_acq_rel,
                                                                       std::memory_order_acquire);
        return failure;
    }

    /** Maps every plane up to @p rows and sentinel-fills the fresh neighbor rows - always ahead of
     *  any launch that could assign new_node_ids into them. Callers hold @c claiming_, or run
     *  exclusively like @c reserve and @c load. */
    status_t grow_rows_(std::size_t rows, cudaStream_t stream) noexcept {
        if (rows <= capacity()) return status_t::success_k;
        std::size_t const neighbors_before = neighbors_plane_.committed_bytes();
        if (status_t status = vectors_plane_.commit(rows * row_bytes()); status != status_t::success_k) return status;
        if (status_t status = norms_plane_.commit(rows * sizeof(norms_t)); status != status_t::success_k) return status;
        if (status_t status = neighbors_plane_.commit(rows * connectivity() * sizeof(u32_t));
            status != status_t::success_k)
            return status;
        // Past a successful commit a failure must be kept: `capacity()` has advanced, so a retry
        // would early-return and assign IDs into rows never sentinel-filled.
        std::size_t const fresh_bytes = neighbors_plane_.committed_bytes() - neighbors_before;
        if (fresh_bytes && cuMemsetD32Async(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>( //
                                                neighbors_plane_.base() + neighbors_before)),
                                            dense_node_id_unknown_k, fresh_bytes / 4, stream) != CUDA_SUCCESS)
            return fail_(status_t::device_error_k);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return fail_(status_t::device_error_k);
        return status_t::success_k;
    }

    /** Rounds the requested expansion to the closed pool ladder: 64/128/256/512. */
    static u32_t pool_slots_for_(u32_t expansion_factor, u32_t default_slots) noexcept {
        if (!expansion_factor) return default_slots;
        u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion_factor, 32));
        return min_of(max_of(slots, 2u), 16u);
    }

    status_t launch_insert_(dense_cuda_build_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        status_t launched = status_t::success_k;
        dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
            launched = dense_cuda_launch<dense_cuda_insert_kernel<kit_t, slots.value>>( //
                view, device_.multiprocessor_count * insert_blocks_per_multiprocessor_k, stream);
        });
        return launched;
    }
    status_t launch_refine_(dense_cuda_refine_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        status_t launched = status_t::success_k;
        dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
            launched = dense_cuda_launch<dense_cuda_refine_kernel<kit_t, slots.value>>( //
                view, device_.multiprocessor_count * insert_blocks_per_multiprocessor_k, stream);
        });
        return launched;
    }
    status_t launch_merge_(dense_cuda_refine_view<kit_t> view, u32_t const* bucket_ids, u32_t const* bucket_counts,
                           u32_t bucket_width, dense_cuda_bitset_t touched, cudaStream_t stream) noexcept {
        return dense_cuda_launch<dense_cuda_merge_links_kernel<kit_t>>( //
            view, device_.multiprocessor_count * insert_blocks_per_multiprocessor_k, stream, bucket_ids, bucket_counts,
            bucket_width, touched);
    }

    /** Launches one walk per query of @p launch, each written out through @p sink, and reports the
     *  launch's own verdict. */
    template <typename sink_, typename rescoring_>
    status_t launch_walk_(walk_launch<rescoring_> const& launch, sink_ sink, cudaStream_t stream) noexcept {
        u32_t const blocks = device_.multiprocessor_count * search_blocks_per_multiprocessor_k;
        if (launch.frontier == frontier_kind_t::wide_arena_k)
            return dense_cuda_launch<dense_cuda_walk_wide_kernel<kit_t, sink_, rescoring_>>(launch.view, blocks, stream,
                                                                                            sink, launch.rescoring);
        status_t launched = status_t::success_k;
        dense_cuda_with_pool_slots(launch.pool_slots, [&](auto slots) {
            launched = dense_cuda_launch<dense_cuda_walk_kernel<kit_t, slots.value, sink_, rescoring_>>(
                launch.view, blocks, stream, sink, launch.rescoring);
        });
        return launched;
    }

#pragma endregion Launches

    /** The vector and norm planes' backend, rebound to bytes. */
    using vectors_plane_allocator_t = typename vectors_allocator_t::template rebind<byte_t>::other;

    /** The neighbor plane's backend, rebound to bytes. */
    using neighbors_plane_allocator_t = typename neighbors_allocator_t::template rebind<byte_t>::other;

    /** The published-row counter is plain stream-ordered device memory on the legacy stream. */
    using counter_allocator_t = cuda_device_allocator<u32_t>;

    /** Hardware identity: the ordinal comes from the vectors allocator at construction, the
     *  remaining specs fill at the first @c reserve. */
    cuda_device_specs_t device_;

    /** The frozen geometry word: dimensions high, connectivity low, zero when unreserved. */
    packed_geometry_id_t geometry_ = 0;

    /** The vector and norm planes; each allocator is its plane, owning the base and the committed
     *  extent, so capacities are derived, never cached. */
    vectors_plane_allocator_t vectors_plane_, norms_plane_;

    /** The neighbor plane, on the same terms. */
    neighbors_plane_allocator_t neighbors_plane_;

    /** The device-resident published-row count, allocated by the first @c reserve(); every driving
     *  context polls it, none owns it. */
    cuda_unique_ptr<u32_t, counter_allocator_t> published_rows_;

    /** Held by the one batch growing and claiming at a time. */
    fu::spin_mutex_t claiming_;

    /** Each batch's dense ID range, release-published once its launch is accepted. */
    std::size_t claimed_rows_ = 0;

    /** The first post-claim device failure: IDs past it may be unpublished or unfilled. */
    status_t status_ = status_t::success_k;
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
