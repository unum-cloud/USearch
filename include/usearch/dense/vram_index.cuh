/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/vram_index.cuh
 *  @date July 18, 2026
 */
#pragma once

#include <atomic>  // `std::atomic` row claiming across concurrent inserting contexts
#include <thread>  // `std::this_thread::yield` while another context grows the planes
#include <bit>     // `std::bit_width` for hash-prefix widths
#include <cstddef> // `std::byte` for raw scratch & arena spans
#include <cstring> // `std::memcpy`

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
#include "usearch/dense/vram_metric.cuh"

namespace unum {
namespace usearch {

/** @brief Warps per thread block across every kernel of this design - four schedulers per
 *      SM on every modern NVIDIA architecture, one warp each. */
inline constexpr u32_t dense_cuda_warps_per_block_k = 4;
/** @brief Resident blocks per SM - a residency contract shared by the launch grids and the
 *      kernels' launch bounds, so the register allocator sizes to it instead of trading
 *      spills for occupancy the launches never ask for. Search kernels with small pools fit
 *      a four-block contract spill-free and earn a third more resident warps; the wider
 *      pool ladder and every insert kernel stay at three. */
inline constexpr u32_t dense_cuda_resident_blocks_per_sm_k = 3;
/** @brief Rows of the per-warp shared candidate cache; also bounds the diversity-selection
 *      pool prefix and the staging arrays. */
inline constexpr u32_t dense_cuda_cache_rows_k = 64;
/** @brief Floor & jitter of the @c __nanosleep backoff while retrying a held slot-0 lock -
 *      measured to keep hub convoys tolerable on young graphs. */
inline constexpr u32_t lock_backoff_floor_ns_k = 512;
inline constexpr u32_t lock_backoff_jitter_mask_k = 1023;
/** @brief Acquire attempts before a traversal drops a transiently locked candidate. */
inline constexpr u32_t traversal_peek_retries_k = 16;
/** @brief Acquire attempts before a back-linker abandons one reverse edge - the bounded-wait
 *      policy; abandonments are telemetered and measured below one in a million. */
inline constexpr u32_t back_link_lock_attempts_k = 256;
/** @brief Admission-ramp floor: the in-flight window starts here and doubles with the
 *      published set, bounding bootstrap lock contention. */
inline constexpr u32_t admission_ramp_floor_k = 32;

/** @brief Device-resident index control block - the state warps consult across SMs and
 *      across launches. One per index, allocated through the traversals allocator; every
 *      driving context polls it, none owns it. All fields are all-RMW, no compare-and-swap.
 *      @ref dense_vram_index_concurrency */
struct dense_cuda_control_t {
    /** @brief Published-node count: advanced by a fire-and-forget @c atomicAdd at each
     *      slot-0 release, polled by the admission ramp. Advisory - it implies nothing
     *      about @b which ids are published. */
    u32_t published_count;
};

/** @brief Device-resident telemetry sink, aggregated once per traversal. One per driving
 *      context, never shared - two concurrent launches diffing one block would
 *      cross-contaminate their before/after snapshots. */
struct dense_cuda_telemetry_block_t {
    u64_t hops, scored, lock_attempts, lock_skips, full_lists, restarts, locked_drops;
    /** @brief Scored candidates split by plane residency - the counters that make every
     *      multi-device throughput number self-explaining. Zero unless the planes view
     *      names a resident range. */
    u64_t local_scored, remote_scored;
};

#if defined(__CUDACC__)

/**
 *  @brief Typed views over the three device planes, @b [rows,*] each: packed vectors,
 *      per-row norms, and fixed-capacity neighbor lists. Under the flat virtual-memory
 *      arena the node id @b is the location - one multiply, no chunk directory.
 */
template <dense_cuda_kit kit_>
struct dense_cuda_planes_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** @brief Vector plane, @b [rows,row_fragments] fragments, one aligned row per node. */
    fragment_t* vectors;
    /** @brief Norms plane, @b [rows] records, written once pre-publication. */
    norms_t* norms;
    /** @brief Neighbor plane, @b [rows,connectivity] ids; slot 0 doubles as the lock word. */
    u32_t* neighbors;
    /** @brief Neighbor-row width: the stride of @c neighbors and the link budget per node. */
    u32_t connectivity;
    /** @brief Fragments per vector row - the stride of @c vectors; the row's byte length is
     *      the dimension count rounded up to whole fragments, tail bytes zeroed. */
    u32_t row_fragments;
    /** @brief Lanes cooperating on one row while scoring: the smallest power of two
     *      covering @c row_fragments, capped at the warp width. */
    u32_t lanes_per_row;
    /** @brief Scoring steps each of those lanes takes to cover one full row. */
    u32_t steps_per_row;
    /** @brief Fixed-point reciprocal of @c row_fragments - `ceil(2^31 / row_fragments)` -
     *      so flattened [row,fragment] word indices split without hardware division. The
     *      2^31 base fits any fragment width in 32 bits, and exactness holds whenever
     *      word x row_fragments stays under 2^31 - true for every cache-fill index here. */
    u32_t row_fragments_magic;
    /** @brief Rows physically resident on the executing device, feeding the locality
     *      telemetry - both zero when the view spans one device and the split would only
     *      state the obvious. */
    u32_t resident_begin = 0, resident_end = 0;

    /** @brief Splits a flattened word index over a @b [rows,row_fragments] grid - one
     *      32x32 widening multiply and a shift. */
    __device__ __forceinline__ u32_t row_of_word(u32_t word) const noexcept {
        return static_cast<u32_t>((static_cast<u64_t>(word) * row_fragments_magic) >> 31);
    }
    __device__ __forceinline__ u32_t fragment_of_word(u32_t word, u32_t row) const noexcept {
        return word - row * row_fragments;
    }

    /** @brief Lane's fragment of the vector stored at @p node_id. */
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
 *  @brief Typed view over one node's neighbor row - the only legal way to touch a list
 *      that concurrent warps may be reading, mirroring the CPU's @c neighbors_list_ref_t
 *      method-for-method over @c cuda::atomic_ref at device scope.
 *
 *  Slot 0 is the combined lock & first-neighbor word: the all-ones sentinel while
 *  unpublished or locked, the node's own id when published empty, otherwise the first
 *  neighbor. One deliberate divergence from the CPU: the lock is @c lock_if_free with a
 *  bounded attempt budget - unbounded spinning against a descheduled owner warp is a
 *  livelock hazard unique to the GPU, and skipped reverse edges are the accepted,
 *  telemetered cost.
 */
struct dense_cuda_neighbors_ref_t {
    u32_t* row_; // Non-const even for readers: `cuda::atomic_ref` cannot bind const words pre-P3323.
    u32_t capacity_;

    using word_ref_t = cuda::atomic_ref<u32_t, cuda::thread_scope_device>;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }
    /** @brief Warp-ballot mask of the lanes whose index is a valid slot of this row -
     *      collective row reads report the sentinel on the lanes outside it. */
    __device__ __forceinline__ unsigned lane_mask() const noexcept {
        return capacity_ >= 32 ? 0xFFFFFFFFu : (1u << capacity_) - 1u;
    }

    /** @brief Relaxed atomic read of slot @p index; slot 0 belongs to @c peek_first / @c lock_if_free. */
    __device__ __forceinline__ dense_node_id_t load_slot(u32_t index) const noexcept {
        return word_ref_t(row_[index]).load(cuda::memory_order_relaxed);
    }
    /** @brief Relaxed atomic write of slot @p index - only for the unpublished owner or the
     *      current lock holder. */
    __device__ __forceinline__ void store_slot(u32_t index, dense_node_id_t value) const noexcept {
        word_ref_t(row_[index]).store(value, cuda::memory_order_relaxed);
    }
    /** @brief Acquire-load of the lock word: the sentinel means unpublished or locked; any
     *      other value is the first neighbor id, with everything its last (un)locker wrote
     *      now visible to us. */
    __device__ __forceinline__ dense_node_id_t peek_first() const noexcept {
        return word_ref_t(row_[0]).load(cuda::memory_order_acquire);
    }
    /** @brief Bounded lock acquisition: exchanges the sentinel in with acquire order, up to
     *      @p max_attempts times with jittered @c __nanosleep backoff. An exchange that
     *      receives the sentinel back did @b not acquire the lock - that rule lives here,
     *      in one place. @return The displaced first-neighbor value, or the sentinel. */
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
    /** @brief Release-stores @p first_value into slot 0 - unlocking and simultaneously
     *      (re)publishing the edited list to every future acquiring reader. */
    __device__ __forceinline__ void unlock(dense_node_id_t first_value) const noexcept {
        word_ref_t(row_[0]).store(first_value, cuda::memory_order_release);
    }
    /** @brief Warp-collective row read for @b unlocked rows: the leader lane acquires
     *      slot 0, the others load their slots relaxed - one ordered load per row, not
     *      thirty-two. The sentinel in the returned lane-0 value means the row is
     *      transiently locked or unpublished. A lock holder must @b never call this -
     *      slot 0 holds the sentinel while locked; pass the displaced first value from
     *      @c lock_if_free through @c locked_row instead. */
    __device__ __forceinline__ dense_node_id_t load_row(int lane) const noexcept {
        if (static_cast<u32_t>(lane) >= capacity_) return dense_node_id_unknown_k;
        return lane ? load_slot(lane) : peek_first();
    }
    /** @brief Warp-collective row read for the @b lock holder: lane 0 reports the
     *      displaced @p first_value, the others load their slots relaxed. */
    __device__ __forceinline__ dense_node_id_t locked_row(int lane, dense_node_id_t first_value) const noexcept {
        if (static_cast<u32_t>(lane) >= capacity_) return dense_node_id_unknown_k;
        return lane ? load_slot(lane) : first_value;
    }
};

/** @brief Per-warp register telemetry, folded into the global counters once per node.
 *      Counters are 32-bit in registers - one walk's hops cap at a few thousand and every
 *      other counter is bounded by hops times the connectivity - and widen at the flush. */
struct dense_cuda_telemetry_t {
    u32_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, full_lists = 0, restarts = 0, locked_drops = 0;
    u32_t local_scored = 0, remote_scored = 0;

    /** @brief Folds the live counters into the device block; the search path never touches
     *      the lock or list counters, so its instantiation skips four dead atomics. */
    template <bool inserting_>
    __device__ __forceinline__ void flush(dense_cuda_telemetry_block_t* counters, int lane) noexcept {
        // @c atomicAdd spells 64-bit as `unsigned long long`; @c u64_t is layout-identical.
        static_assert(sizeof(u64_t) == sizeof(unsigned long long));
        if (lane != 0) return;
        using wide_t = unsigned long long;
        atomicAdd(reinterpret_cast<wide_t*>(&counters->hops), static_cast<wide_t>(hops));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->scored), static_cast<wide_t>(scored));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->restarts), static_cast<wide_t>(restarts));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->locked_drops), static_cast<wide_t>(locked_drops));
        if (local_scored | remote_scored) {
            atomicAdd(reinterpret_cast<wide_t*>(&counters->local_scored), static_cast<wide_t>(local_scored));
            atomicAdd(reinterpret_cast<wide_t*>(&counters->remote_scored), static_cast<wide_t>(remote_scored));
        }
        if constexpr (!inserting_) return;
        atomicAdd(reinterpret_cast<wide_t*>(&counters->lock_attempts), static_cast<wide_t>(lock_attempts));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->lock_skips), static_cast<wide_t>(lock_skips));
        atomicAdd(reinterpret_cast<wide_t*>(&counters->full_lists), static_cast<wide_t>(full_lists));
    }
};

/**
 *  @brief Each warp performing scans over the graph owns one context like this - the
 *      device twin of the CPU's per-thread traversal context: the candidate pool, the
 *      lane's PRNG stream, and the running telemetry.
 */
template <typename frontier_>
struct dense_cuda_warp_context {
    frontier_ frontier;
    splitmix64_t random_generator;
    dense_cuda_telemetry_t telemetry;
};

/**
 *  @brief Scores one staged batch of up to 32 candidate rows as a single flattened load
 *      gather batch. Rows are gathered in groups: a sub-warp of @c lanes_per_row lanes owns
 *      one row, stepping @c steps_per_row times across its fragments, while the group tile
 *      keeps up to @c max_gathers_in_flight_k independent loads per lane. The query lives in
 *      a shared-memory span and is re-read by broadcast each step, so no per-lane query
 *      registers cap the dimension count.
 *
 *  Mutable-plane discipline: vector & norm loads here are plain loads, made safe by the
 *  acquire chain - every staged id was discovered through an acquiring slot-0 read that
 *  synchronizes with its publisher's release. @return This lane's scored entry, or the
 *  sentinel on lanes past @p gathered_count.
 */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_sm_k>
__device__ __forceinline__ dense_cuda_scored_node_t dense_cuda_score_batch( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, u32_t const (&gathered_ids)[dense_cuda_cache_rows_k],
    dense_cuda_scored_node_t (&gathered_scored)[32], u32_t gathered_count, int lane, u32_t first_row = 0) noexcept {

    using norms_t = typename kit_::norms_t;
    // The gather window targets the SM's outstanding-request knee: 128 concurrent random
    // gathers saturate the HBM path, so window = 128 / resident warps. On the 4-block
    // kernels this knowingly oversubscribes the register file by a few registers - ptxas
    // spills a handful of COLD loop-carried bytes to keep every HOT gather slot live.
    constexpr u32_t gathers_in_flight_k = min_of(kit_::max_gathers_in_flight_k,
                                                 128u / (dense_cuda_warps_per_block_k * resident_blocks_));
    u32_t const lanes_log2 = 31 - __clz(planes.lanes_per_row);
    u32_t const rows_per_gather = 32u >> lanes_log2;
    u32_t const lane_in_row = static_cast<u32_t>(lane) & (planes.lanes_per_row - 1);
    u32_t const subwarp = static_cast<u32_t>(lane) >> lanes_log2;
    bool const leader = lane_in_row == 0;

    // Row groups are independent; the pass loop tiles them so at most `gathers_in_flight_k` loads -
    // the gather register budget - coexist per lane.
#pragma unroll 1
    for (u32_t group_base = 0; group_base < planes.lanes_per_row; group_base += gathers_in_flight_k) {
        norms_t batch_norms[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (kit_::has_norms_k && leader && group_base + slot < planes.lanes_per_row && row < gathered_count)
                batch_norms[slot] = planes.norm_for(gathered_ids[first_row + row]);
        }
        typename kit_::accumulator_t accumulators[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) accumulators[slot] = kit_::zero_accumulator();
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
                    loaded[slot] = planes.vector_for(gathered_ids[first_row + row], fragment_index);
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
            typename kit_::dot_t const dot = dense_cuda_reduce(kit_::partial_to_dot(accumulators[slot]),
                                                               planes.lanes_per_row);
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (leader && group_base + slot < planes.lanes_per_row && row < gathered_count) {
                dense_distance_t const distance = kit_::distance(dot, query_state, batch_norms[slot]);
                gathered_scored[row] = dense_cuda_scored_node_t::make(distance, gathered_ids[first_row + row]);
            }
        }
    }
    __syncwarp();
    return lane < static_cast<int>(gathered_count) ? gathered_scored[lane] : dense_cuda_scored_node_t::sentinel();
}

/** @brief Scores one gathered batch against the query and folds it into the pool,
 *      dropping the entries the pool already holds. */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_sm_k, typename frontier_>
__device__ __forceinline__ void dense_cuda_score_into_frontier( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, u32_t const (&gathered_ids)[dense_cuda_cache_rows_k],
    dense_cuda_scored_node_t (&gathered_scored)[32], u32_t gathered_count, dense_cuda_warp_context<frontier_>& context,
    int lane) noexcept {
    context.telemetry.scored += gathered_count;
    if (planes.resident_end) {
        bool const counted = static_cast<u32_t>(lane) < gathered_count;
        u32_t const candidate = counted ? gathered_ids[lane] : 0;
        unsigned const local_mask = __ballot_sync(
            0xFFFFFFFFu, counted && candidate >= planes.resident_begin && candidate < planes.resident_end);
        context.telemetry.local_scored += __popc(local_mask);
        context.telemetry.remote_scored += gathered_count - __popc(local_mask);
    }
    dense_cuda_scored_node_t const staged = dense_cuda_score_batch<kit_, resident_blocks_>( //
        planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, lane);
    // No dedup: the exact visited set upstream guarantees every id stages once.
    context.frontier.push_batch(staged, lane);
}

/**
 *  @brief Samples one 32-wide batch of distinct, published landmarks below @p sampling_bound
 *      into @p gathered_ids - uniform draws gated on the slot-0 publication word, probing
 *      forward on unpublished hits so out-of-order publication only costs retries.
 *  @return The number of staged landmarks; zero when the visited set saturated the bound.
 */
template <dense_cuda_kit kit_>
__device__ __forceinline__ u32_t dense_cuda_sample_landmarks( //
    dense_cuda_planes_view<kit_> const& planes, dense_node_id_t sampling_bound, dense_cuda_visited_set_t visited,
    splitmix64_t& random_generator, u32_t (&gathered_ids)[dense_cuda_cache_rows_k], int lane) noexcept {

    dense_node_id_t draw = uniform_below<dense_node_id_t> {sampling_bound}(random_generator);
    bool published = false;
    for (int attempt = 0; attempt != 4 && !published; ++attempt) {
        dense_cuda_neighbors_ref_t const row {planes.neighbors_for(draw), planes.connectivity};
        published = row.peek_first() != dense_node_id_unknown_k;
        if (!published) draw = draw + 1 == sampling_bound ? 0 : draw + 1;
    }
    bool const fresh = visited.test_and_set(draw, lane, published);
    unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
    u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
    if (fresh) gathered_ids[ordinal] = draw;
    __syncwarp();
    return __popc(fresh_mask);
}

/**
 *  @brief Continues a greedy best-first NSW walk toward the query over whatever the
 *      frontier already holds - one warp, one traversal. Entries staged before the call
 *      compete with the landmark seeds drawn here, so a caller can pre-seed the walk with
 *      known candidates; every pre-staged id must already be in the visited set. Read-only
 *      on the graph: takes no locks, drops transiently locked candidates after a bounded
 *      retry, and leaves the nearest published candidates ranked in @c context.frontier.
 */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_sm_k, typename frontier_>
__device__ __forceinline__ void dense_cuda_search_ahead( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, dense_node_id_t sampling_bound, dense_cuda_visited_set_t visited,
    u32_t (&gathered_ids)[dense_cuda_cache_rows_k], dense_cuda_scored_node_t (&gathered_scored)[32],
    dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    constexpr u32_t restarts_cap_k = 8;
    frontier_& frontier = context.frontier;
    dense_cuda_telemetry_t& telemetry = context.telemetry;
    u32_t const hops_cap = frontier.hops_budget();

    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = dense_cuda_sample_landmarks<kit_>( //
                planes, sampling_bound, visited, context.random_generator, gathered_ids, lane);
            if (gathered_count)
                dense_cuda_score_into_frontier<kit_, resident_blocks_>( //
                    planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
            seeding = false;
        }
        else {
            dense_cuda_scored_node_t const candidate = frontier.pop_nearest(lane);
            if (candidate == dense_cuda_scored_node_t::sentinel()) {
                // Starvation restart: a drained frontier before an exhausted walk means
                // nothing reachable from its seeds remains - re-seed and continue.
                if (frontier.exhausted() || telemetry.restarts >= restarts_cap_k || telemetry.hops >= hops_cap) break;
                seeding = true;
                ++telemetry.restarts;
                continue;
            }
            // Once the nearest remaining candidate lies beyond the frontier's horizon,
            // nothing deeper can improve the result set.
            if (frontier.beyond_radius(candidate)) break;
            if (telemetry.hops >= hops_cap) break;
            ++telemetry.hops;

            // Read the candidate's neighbor row; a transiently locked row is retried
            // briefly, then dropped rather than convoying behind the re-linker. Only the
            // leader lane polls the racing lock word: per-lane polls can observe different
            // values of a concurrent unlock, and a value-dependent `continue` would fork
            // the warp across loop iterations - fatal for every collective downstream.
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
            if (first_value == dense_node_id_unknown_k) {
                ++telemetry.locked_drops;
                continue;
            }
            // The first warp-width of slots - the whole row when the connectivity fits one
            // warp. Keeping this sweep loop-free keeps the hot traversal path compact.
            dense_node_id_t const neighbor_id = lane == 0                                   ? first_value
                                                : static_cast<u32_t>(lane) < row.capacity() ? row.load_slot(lane)
                                                                                            : dense_node_id_unknown_k;
            bool const usable = neighbor_id != dense_node_id_unknown_k && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(neighbor_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) gathered_ids[ordinal] = neighbor_id;
            __syncwarp();
            u32_t const gathered_count = __popc(fresh_mask);
            if (gathered_count)
                dense_cuda_score_into_frontier<kit_, resident_blocks_>( //
                    planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
            // Rows wider than a warp continue a warp-width of slots at a time; the bound is
            // warp-uniform, so every collective executes at one program point on every lane.
            for (u32_t slot_base = 32; slot_base < planes.connectivity; slot_base += 32) {
                u32_t const slot = slot_base + lane;
                dense_node_id_t const wide_id = slot < planes.connectivity ? row.load_slot(slot)
                                                                           : dense_node_id_unknown_k;
                bool const wide_usable = wide_id != dense_node_id_unknown_k && wide_id != candidate.id();
                bool const wide_fresh = visited.test_and_set(wide_id, lane, wide_usable);
                unsigned const wide_mask = __ballot_sync(0xFFFFFFFFu, wide_fresh);
                u32_t const wide_ordinal = __popc(wide_mask & ((1u << lane) - 1));
                if (wide_fresh) gathered_ids[wide_ordinal] = wide_id;
                __syncwarp();
                u32_t const wide_count = __popc(wide_mask);
                if (wide_count)
                    dense_cuda_score_into_frontier<kit_, resident_blocks_>( //
                        planes, query_span, query_state, gathered_ids, gathered_scored, wide_count, context, lane);
            }
        }
    }
}

/**
 *  @brief Greedy best-first NSW walk from random published landmarks toward the query -
 *      one warp, one traversal: a fresh frontier handed to @ref dense_cuda_search_ahead.
 */
template <dense_cuda_kit kit_, u32_t resident_blocks_ = dense_cuda_resident_blocks_per_sm_k, typename frontier_>
__device__ __forceinline__ void dense_cuda_search_around( //
    dense_cuda_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::query_state_t query_state, dense_node_id_t sampling_bound, dense_cuda_visited_set_t visited,
    u32_t (&gathered_ids)[dense_cuda_cache_rows_k], dense_cuda_scored_node_t (&gathered_scored)[32],
    dense_cuda_warp_context<frontier_>& context, int lane) noexcept {
    context.frontier.reset(lane);
    dense_cuda_search_ahead<kit_, resident_blocks_>( //
        planes, query_span, query_state, sampling_bound, visited, gathered_ids, gathered_scored, context, lane);
}

/** @brief Runtime-strided view over the warp's shared vector cache - rows of
 *      @c row_fragments_ fragments each, aliasing the scratch slice. */
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

/**
 *  @brief Warp-cooperative ascending buffer of scored nodes over a shared-memory span -
 *      the device twin of the CPU's sorted buffer. Chunks arrive pre-sorted from the
 *      32-wide bitonic network and merge in place: every lane ranks its incoming entry
 *      against the whole span, existing entries count the incoming values ahead of them,
 *      and one barrier separates the staged reads from the scattered writes. Entries
 *      ranked past @c capacity_ are dropped - the insert-with-limit contract.
 */
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

/** @brief Shared-memory spans backing one warp's link re-selection, carved after the
 *      vector cache: the selected target ids, the ascending per-link scores, and the
 *      refined cache-row indices. */
struct dense_cuda_relink_scratch_t {
    u32_t* selected_ids;
    dense_cuda_scored_node_t* link_entries;
    u32_t* refined_rows;

    /** @brief Carves the three spans past the vector cache's @p cache_bytes: the ascending
     *      link scores first - the widest alignment - then the selected target ids, then
     *      the refined cache-row indices. The layout rule lives here, with the type. */
    static __device__ __forceinline__ dense_cuda_relink_scratch_t carve(std::byte* scratch, u32_t connectivity,
                                                                        std::size_t cache_bytes) noexcept {
        u32_t const slots = min_of(connectivity, static_cast<u32_t>(dense_cuda_cache_rows_k));
        std::byte* const base = scratch + cache_bytes;
        return {reinterpret_cast<u32_t*>(base + sizeof(dense_cuda_scored_node_t) * slots),
                reinterpret_cast<dense_cuda_scored_node_t*>(base),
                reinterpret_cast<u32_t*>(base + (sizeof(dense_cuda_scored_node_t) + sizeof(u32_t)) * slots)};
    }
};

/**
 *  @brief Greedy diversity selection over the ascending pool prefix - the classic NSW
 *      pruning heuristic: a candidate is kept only when it is closer to the query than to
 *      every already-kept link, so links spread across directions. Candidate vectors are
 *      cached in the warp's shared scratch; the kept list lives in the shared selected
 *      span, ascending by construction since candidates arrive ascending.
 *  @return The kept-link count; @p selected_ids holds that many node ids on return.
 */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ u32_t dense_cuda_select_diverse( //
    dense_cuda_planes_view<kit_> const& planes, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], u32_t (&gathered_ids)[dense_cuda_cache_rows_k],
    u32_t* selected_ids, dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    frontier_& frontier = context.frontier;

    // Stage the ascending prefix ids & cache their vectors and norms.
    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < dense_cuda_cache_rows_k; base += 32) {
        dense_cuda_scored_node_t const entry = base + lane < 64 ? frontier.rank_entry(base + lane)
                                                                : dense_cuda_scored_node_t::sentinel();
        unsigned const live = __ballot_sync(0xFFFFFFFFu, !(entry == dense_cuda_scored_node_t::sentinel()));
        if (!(entry == dense_cuda_scored_node_t::sentinel())) gathered_ids[base + lane] = entry.id();
        candidates_count += __popc(live);
        if (live != 0xFFFFFFFFu) break;
    }
    __syncwarp();
    for (u32_t word = lane; word < candidates_count * planes.row_fragments; word += 32) {
        u32_t const row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, row);
        __pipeline_memcpy_async(&cache.at(row, fragment), planes.vector_span_for(gathered_ids[row]) + fragment,
                                sizeof(typename kit_::fragment_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    if (kit_::has_norms_k)
        for (u32_t row = lane; row < candidates_count; row += 32) cache_norms[row] = planes.norm_for(gathered_ids[row]);
    __syncwarp();

    u32_t selected_count = 0;
    for (u32_t candidate_index = 0; candidate_index < candidates_count && selected_count < planes.connectivity;
         ++candidate_index) {
        f32_t const to_query = frontier.rank_entry(candidate_index).distance();
        bool diverse = true;
        for (u32_t base = 0; base < selected_count && diverse; base += 32) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < selected_count) {
                u32_t const kept_row = selected_ids[base + lane];
                typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    accumulator = kit_::accumulate_dot(cache.at(candidate_index, fragment),
                                                       cache.at(kept_row, fragment), accumulator);
                f32_t const between = kit_::distance(kit_::partial_to_dot(accumulator),
                                                     kit_::make_query_state(cache_norms[candidate_index]),
                                                     cache_norms[kept_row]);
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
    // Rewrite the kept rows as node ids in place - the caller links to nodes, not rows.
    __syncwarp();
    for (u32_t slot = lane; slot < selected_count; slot += 32) selected_ids[slot] = gathered_ids[selected_ids[slot]];
    __syncwarp();
    return selected_count;
}

/**
 *  @brief Wires the reverse edges: locks each selected neighbor's row via the bounded
 *      slot-0 exchange, appends when there is room, and re-runs the diversity heuristic
 *      over the full list plus the newcomer when there is none - rewriting the row under
 *      the lock. Cheaper replacement policies measurably degrade the graph.
 *
 *  Locking discipline: at most one neighbor lock is held at any moment, only after this
 *  node's own publication, and never while waiting on anything else - lock holders are
 *  always runnable. A lock that cannot be acquired within the attempt budget skips the
 *  edge; approximate search tolerates the missing reverse link by construction.
 */
template <dense_cuda_kit kit_>
__device__ __forceinline__ void dense_cuda_back_link_neighbors( //
    dense_cuda_planes_view<kit_> const& planes, dense_node_id_t new_node_id, typename kit_::norms_t new_norms,
    dense_cuda_relink_scratch_t relink, u32_t selected_count, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], u32_t (&gathered_ids)[dense_cuda_cache_rows_k],
    dense_cuda_scored_node_t (&gathered_scored)[32], splitmix64_t& random_generator,
    dense_cuda_telemetry_t& telemetry, int lane) noexcept {

    // The cache hosts every existing link, the newcomer on the last row, and the re-linked
    // target itself on the row before it - the scored batch reads the target as its query.
    constexpr u32_t newcomer_row_k = dense_cuda_cache_rows_k - 1;
    constexpr u32_t target_row_k = dense_cuda_cache_rows_k - 2;
    // Those two reserved rows also bound how many links one warp can re-select in shared
    // memory; wider rows keep sentinel tails until the arena lifts the cache ceiling.
    u32_t const connectivity = min_of(planes.connectivity, dense_cuda_cache_rows_k - 2u);

    for (u32_t link = 0; link < selected_count; ++link) {
        dense_node_id_t const target = relink.selected_ids[link];
        dense_cuda_neighbors_ref_t const row {planes.neighbors_for(target), connectivity};
        dense_node_id_t first_value = dense_node_id_unknown_k;
        if (lane == 0) first_value = row.lock_if_free(back_link_lock_attempts_k, random_generator);
        first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
        ++telemetry.lock_attempts;
        if (first_value == dense_node_id_unknown_k) { // Bounded-wait policy: skip the edge.
            ++telemetry.lock_skips;
            continue;
        }
        if (first_value == target) { // Published-empty self-loop: the newcomer is the first neighbor.
            if (lane == 0) row.unlock(new_node_id);
            continue;
        }

        // Read the whole locked row a warp-width at a time, remembering the first empty
        // slot; the same sweep stages every live link id for the potential re-selection.
        u32_t empty_slot = connectivity;
        for (u32_t base = 0; base < connectivity; base += 32) {
            u32_t const slot = base + lane;
            dense_node_id_t const slot_value = slot == 0             ? first_value
                                               : slot < connectivity ? row.load_slot(slot)
                                                                     : dense_node_id_unknown_k;
            if (slot < connectivity) gathered_ids[slot] = slot_value;
            unsigned const empty_mask = __ballot_sync(0xFFFFFFFFu,
                                                      slot < connectivity && slot_value == dense_node_id_unknown_k);
            if (empty_mask && empty_slot == connectivity) empty_slot = base + __ffs(empty_mask) - 1;
        }
        __syncwarp();
        if (empty_slot != connectivity) { // Fast path: append into that slot, unlock unchanged.
            if (lane == 0) row.store_slot(empty_slot, new_node_id);
            __syncwarp();
            if (lane == 0) row.unlock(first_value);
            continue;
        }

        // Full list: score every link & the newcomer against the target itself, then walk
        // all of them ascending, re-running the diversity heuristic, and rewrite the row.
        ++telemetry.full_lists;
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            cache.at(target_row_k, word) = planes.vector_for(target, word);
        __syncwarp();
        typename kit_::query_state_t const target_state = kit_::make_query_state(planes.norm_for(target));

        // The newcomer's distance to the target, from its freshly-owned vector row.
        typename kit_::accumulator_t accumulator = kit_::zero_accumulator();
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            accumulator = kit_::accumulate_dot(planes.vector_for(new_node_id, word), cache.at(target_row_k, word),
                                               accumulator);
        typename kit_::dot_t const self_dot = dense_cuda_reduce(kit_::partial_to_dot(accumulator), 32u);
        f32_t const newcomer_distance = kit_::distance(__shfl_sync(0xFFFFFFFFu, self_dot, 0), target_state, new_norms);
        telemetry.scored += connectivity + 1;

        // Score the links a warp-width at a time, pre-sort each batch with the bitonic
        // network, and merge every batch into the ascending sorted buffer - the ported
        // insert-with-limit, so any link count the cache can hold walks in order.
        dense_cuda_sorted_buffer_t links_sorted {relink.link_entries, 0, connectivity};
        for (u32_t base = 0; base < connectivity; base += 32) {
            u32_t const batch_count = min_of(connectivity - base, 32u);
            dense_cuda_scored_node_t const scored = dense_cuda_score_batch<kit_>( //
                planes, cache.row(target_row_k), target_state, gathered_ids, gathered_scored, batch_count, lane, base);
            dense_cuda_scored_node_t const sorted_batch = dense_cuda_sort32(
                static_cast<u32_t>(lane) < batch_count
                    ? dense_cuda_scored_node_t::make(scored.distance(), base + lane) // Id = slot index.
                    : dense_cuda_scored_node_t::sentinel(),
                lane);
            links_sorted.insert_batch(sorted_batch, batch_count, lane);
        }

        // Cache the links' vectors (their norms ride the staging) plus the newcomer at the
        // tail row, then merge-walk all `connectivity + 1` ascending by distance-to-target.
        // The fill rides the async-copy pipeline: every fragment is a global-to-shared
        // `cp.async` that holds no registers, so the runtime word count costs no pressure.
        u32_t const link_words = connectivity * planes.row_fragments;
        for (u32_t word = lane; word < link_words; word += 32) {
            u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
            __pipeline_memcpy_async(&cache.at(cache_row, fragment),
                                    planes.vector_span_for(gathered_ids[cache_row]) + fragment,
                                    sizeof(typename kit_::fragment_t));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        for (u32_t word = lane; word < planes.row_fragments; word += 32)
            cache.at(newcomer_row_k, word) = planes.vector_for(new_node_id, word);
        for (u32_t cache_row = lane; cache_row < connectivity; cache_row += 32)
            cache_norms[cache_row] = planes.norm_for(gathered_ids[cache_row]);
        if (lane == 0) cache_norms[newcomer_row_k] = new_norms;
        __syncwarp();

        u32_t existing_index = 0;
        bool newcomer_pending = true;
        u32_t refined_count = 0;
        for (u32_t merge_index = 0; merge_index != connectivity + 1 && refined_count != connectivity; ++merge_index) {
            dense_cuda_scored_node_t const next_link = existing_index < links_sorted.size_
                                                           ? links_sorted.at(existing_index)
                                                           : dense_cuda_scored_node_t::sentinel();
            bool const take_newcomer = newcomer_pending && newcomer_distance < next_link.distance();
            f32_t const candidate_distance = take_newcomer ? newcomer_distance : next_link.distance();
            u32_t const candidate_row = take_newcomer ? newcomer_row_k : next_link.id();
            if (!take_newcomer && existing_index >= links_sorted.size_) break;
            if (take_newcomer) newcomer_pending = false;
            else ++existing_index;

            bool diverse = true;
            for (u32_t base = 0; base < refined_count && diverse; base += 32) { // Warp-uniform bounds.
                bool lane_diverse = true;
                if (base + lane < refined_count) {
                    u32_t const kept_row = relink.refined_rows[base + lane];
                    typename kit_::accumulator_t pair_accumulator = kit_::zero_accumulator();
                    for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                        pair_accumulator = kit_::accumulate_dot(cache.at(candidate_row, fragment),
                                                                cache.at(kept_row, fragment), pair_accumulator);
                    f32_t const between = kit_::distance(kit_::partial_to_dot(pair_accumulator),
                                                         kit_::make_query_state(cache_norms[candidate_row]),
                                                         cache_norms[kept_row]);
                    lane_diverse = between >= candidate_distance;
                    ++telemetry.scored;
                }
                diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
            }
            if (diverse) {
                if (lane == 0) relink.refined_rows[refined_count] = candidate_row;
                ++refined_count;
                __syncwarp();
            }
        }

        // Rewrite the row: tail slots relaxed, shrink sentinel-terminated, head unlocks.
        __syncwarp();
        for (u32_t slot = lane; slot < connectivity; slot += 32) {
            if (slot == 0) continue;
            dense_node_id_t linked = dense_node_id_unknown_k;
            if (slot < refined_count) {
                u32_t const kept_row = relink.refined_rows[slot];
                linked = kept_row == newcomer_row_k ? new_node_id : gathered_ids[kept_row];
            }
            row.store_slot(slot, linked);
        }
        __syncwarp();
        if (lane == 0) {
            u32_t const head_row = relink.refined_rows[0];
            row.unlock(head_row == newcomer_row_k ? new_node_id : gathered_ids[head_row]);
        }
    }
}

/**
 *  @brief Layout of one warp's slice of the global traversal arena - the single owner of
 *      that arithmetic: the host computes the shape, the views carry it, and the kernels
 *      ask it for typed spans. Every slice holds the exact visited set; wide walks add
 *      the frontier's spill tail after it.
 */
struct dense_cuda_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    constexpr std::size_t slice_bytes() const noexcept {
        return static_cast<std::size_t>(visited_slots) * sizeof(u64_t) +
               static_cast<std::size_t>(tail_entries) * sizeof(dense_cuda_scored_node_t);
    }
    __device__ __forceinline__ u64_t* visited_span(std::byte* base, std::size_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(base + warp_index * slice_bytes());
    }
    __device__ __forceinline__ dense_cuda_scored_node_t* tail_span(std::byte* base,
                                                                   std::size_t warp_index) const noexcept {
        return reinterpret_cast<dense_cuda_scored_node_t*>( //
            base + warp_index * slice_bytes() + static_cast<std::size_t>(visited_slots) * sizeof(u64_t));
    }
};

/** @brief One warp's launch-uniform slice - its coordinates, its dynamic-shared scratch,
 *      and its arena visited span: the shared prologue of every traversal kernel, all
 *      affine arithmetic compiling into the exact SASS it replaces. */
struct dense_cuda_warp_slice_t {
    int lane, warp;
    std::size_t warp_index;
    std::byte* scratch;
    u64_t* visited_slots;
};

template <typename view_>
__device__ __forceinline__ dense_cuda_warp_slice_t dense_cuda_carve_slice(view_ const& view,
                                                                          std::byte* dynamic_scratch) noexcept {
    int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    std::size_t const warp_index = blockIdx.x * static_cast<std::size_t>(dense_cuda_warps_per_block_k) + warp;
    std::byte* const scratch = dynamic_scratch + static_cast<std::size_t>(warp) * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch, view.arena_shape.visited_span(view.arena, warp_index)};
}

/** @brief Geometry & plane pointers one insertion launch operates on. Every plane row a
 *      dispensable new_node_id can touch is mapped and sentinel-filled ahead of the launch, so
 *      growth hazards cannot reach the kernel. */
template <dense_cuda_kit kit_>
struct dense_cuda_build_view {
    dense_cuda_planes_view<kit_> planes;
    /** @brief Incoming vectors, @b [rows,row_fragments] in dispensed-id order, device-resident. */
    typename kit_::fragment_t const* staging;
    /** @brief Ids strictly below this bound may be claimed & inserted by this launch. */
    u32_t end_id;
    /** @brief Bytes of the per-warp dynamic-shared slice: the query span, grown to fit
     *      the candidate cache and the re-selection spans. */
    u32_t scratch_bytes_per_warp;
    /** @brief The batch's base id: staging row 0 lands there, and the kernel's per-launch
     *      cursor hands out offsets within @b [first_id,end_id). */
    dense_node_id_t first_id;
    /** @brief The traversal arena and each warp's slice of it - the exact visited set. */
    std::byte* arena;
    dense_cuda_arena_shape_t arena_shape;
    /** @brief The context's monotone epoch floor at launch: batch-local epochs ride above
     *      it, so stale arena slots from earlier launches always compare older and the
     *      host never re-zeroes a warm arena. */
    u32_t epoch_base;
    /** @brief The index's shared control block - the admission ramp's cross-batch signal. */
    dense_cuda_control_t* control;
    dense_cuda_telemetry_block_t* telemetry;
    /** @brief This launch's work-queue head: offsets within the batch, zeroed per launch. */
    u32_t* batch_cursor;
};

/** @brief Inputs & outputs of one search launch: a batch of queries in, the nearest ids
 *      and - when requested - their distances out, both ascending per query. */
template <dense_cuda_kit kit_>
struct dense_cuda_search_view {
    dense_cuda_planes_view<kit_> planes;
    typename kit_::fragment_t const* queries; // @b [queries,row_fragments]
    dense_node_id_t* result_ids;              // @b [queries,count]
    dense_distance_t* result_distances;       // @b [queries,count]; null skips the writes
    u32_t queries_count;
    u32_t count;
    /** @brief Published-node ceiling for landmark sampling - the host's snapshot of
     *      @c size() at launch; the walk itself gates on slot-0 publication words. */
    u32_t sampling_bound;
    /** @brief Bytes of the per-warp dynamic-shared slice - the query span. */
    u32_t scratch_bytes_per_warp;
    /** @brief Requested expansion; past the register ladder it becomes the wide walk's
     *      expansion budget and sizes the arena's spill tail. */
    u32_t expansion;
    /** @brief The traversal arena and each warp's slice of it - the exact visited set,
     *      then the wide frontier's spill tail. */
    std::byte* arena;
    dense_cuda_arena_shape_t arena_shape;
    /** @brief The context's monotone epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    dense_cuda_telemetry_block_t* telemetry;
};

/** @brief Default dynamic-shared opt-in threshold: launches asking for more must raise
 *      the kernel's attribute before the first launch. */
inline constexpr u32_t dense_cuda_static_scratch_limit_k = 48 * 1024;

/**
 *  @brief Configures & launches one traversal kernel: raises the dynamic-shared ceiling
 *      when the warp slices outgrow the static window, then launches @p blocks of the
 *      standard warp complement on @p stream - the one launch shape every kernel here
 *      shares, usable by single- and multi-index drivers alike.
 */
template <auto kernel_, typename view_type_, typename... extras_types_>
void dense_cuda_launch(view_type_ view, u32_t blocks, cudaStream_t stream, extras_types_... extras) noexcept {
    u32_t const scratch_bytes = dense_cuda_warps_per_block_k * view.scratch_bytes_per_warp;
    if (scratch_bytes > dense_cuda_static_scratch_limit_k)
        cudaFuncSetAttribute(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch_bytes);
    kernel_<<<blocks, dense_cuda_warps_per_block_k * 32, scratch_bytes, stream>>>(view, extras...);
}

/** @brief Lowers a runtime pool width onto the closed 2/4/8/16 register ladder, invoking
 *      @p callback with the width as an integral constant - the one dispatch every
 *      pooled kernel family shares. */
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
 *  @brief Inserts the node identified by @p new_node_id: lands the payload & norms, walks the
 *      published graph, selects diverse links, publishes with one release store, then
 *      wires the reverse edges - the whole CPU insertion flow, one warp per node.
 */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_cuda_insert_one( //
    dense_cuda_build_view<kit_> const& view, dense_node_id_t new_node_id, dense_cuda_visited_set_t visited,
    typename kit_::fragment_t* query_span, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_relink_scratch_t relink, typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k],
    u32_t (&gathered_ids)[dense_cuda_cache_rows_k], dense_cuda_scored_node_t (&gathered_scored)[32],
    dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;

    // Land the payload: copy the vector fragments, derive both norms from the owned bytes.
    // The same pass parks the query in its shared span for the traversal to re-read.
    typename kit_::fragment_t const* staging_row =
        view.staging + static_cast<std::size_t>(new_node_id - view.first_id) * planes.row_fragments;
    typename kit_::fragment_t* owned_row = planes.vector_span_for(new_node_id);
    typename kit_::norms_accumulator_t norms_accumulator = kit_::zero_norms_accumulator();
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
    typename kit_::query_state_t const query_state = kit_::make_query_state(new_norms);
    __syncwarp();

    // Walk the published prefix - a virgin index publishes an empty node instead.
    if (new_node_id != 0)
        dense_cuda_search_around<kit_>( //
            planes, query_span, query_state, new_node_id, visited, gathered_ids, gathered_scored, context, lane);
    else context.frontier.reset(lane);

    // Select diverse links & write the own row - no locks: no reader arrives before the
    // slot-0 release below flips it from the sentinel.
    u32_t const selected_count = dense_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered_ids, relink.selected_ids, context, lane);
    dense_cuda_neighbors_ref_t const own_row {planes.neighbors_for(new_node_id), planes.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += 32)
        if (slot > 0) own_row.store_slot(slot, relink.selected_ids[slot]);
    __syncwarp();
    dense_node_id_t const first_link = selected_count ? relink.selected_ids[0] : new_node_id;
    if (lane == 0) {
        own_row.unlock(first_link);                    // A self-loop encodes published-empty.
        atomicAdd(&view.control->published_count, 1u); // Fire-and-forget reduction.
    }

    // Only now - published, no locks held - wire the reverse edges.
    dense_cuda_back_link_neighbors<kit_>( //
        planes, new_node_id, new_norms, relink, selected_count, cache, cache_norms, gathered_ids, gathered_scored,
        context.random_generator, context.telemetry, lane);
}

/**
 *
 *  @section dense_vram_index_concurrency Concurrency & Publication
 *
 *  One warp inserts one node, and no two warps ever synchronize beyond per-neighbor
 *  slot-0 locks: there is no watermark, no publication order, and no compare-and-swap.
 *  Id dispensing is batch-local: the host pre-claims a dense range under its growth
 *  latch, and each launch's per-context cursor hands out offsets within it - a warp's
 *  fetch_add result @b is its node id, sibling batches can never cross-claim, and an
 *  aborted launch's over-dispensed offsets die with its cursor, no re-crediting anywhere.
 *  A node is usable exactly when its slot 0 acquire-loads non-sentinel; a self-loop id
 *  there means "published, zero neighbors". Landmark sampling draws below the claimed
 *  new_node_id and probes past unpublished hits, so out-of-order publication costs retries,
 *  never correctness. The admission ramp - new_node_id @b t waits until
 *  t < published + max(32, published) - bounds early-graph lock contention and keeps
 *  the sampler's hit rate above one half; past bootstrap, warps never wait on each other.
 *  Every give-up path still publishes (a self-loop tombstone), so dispensed ids never
 *  stall anything, and a post-mortem sweep can tombstone the stragglers of a killed
 *  launch precisely because publication order is free.
 */
template <dense_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_sm_k)
    dense_cuda_insert_kernel(dense_cuda_build_view<kit_> view) {
    // One launch-sized dynamic slice per warp hosts two disjoint lifetimes over the same
    // bytes: the query span while traversing, then the candidate-vector cache while
    // selecting & back-linking - the host sizes the slice to the larger shape. The exact
    // visited set lives in the warp's arena slice, not in shared memory.
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_cuda_relink_scratch_t const relink = dense_cuda_relink_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_cuda_cache_rows_k);
    while (true) {
        u32_t new_node_id;
        // The work queue is batch-local: the host pre-claims [first_id,end_id) before
        // the launch, so concurrent batches from sibling contexts can never cross-claim,
        // and an aborted launch's over-dispensed offsets die with its cursor.
        if (lane == 0) new_node_id = view.first_id + atomicAdd(view.batch_cursor, 1u);
        new_node_id = __shfl_sync(0xFFFFFFFFu, new_node_id, 0);
        if (new_node_id >= view.end_id) return;

        while (true) { // Admission ramp - the only bootstrap throttle.
            u32_t published;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(published) : "l"(&view.control->published_count));
            u32_t const window = max_of(published, admission_ramp_floor_k);
            if (new_node_id < published + window) break;
            __nanosleep(2048);
        }

        dense_cuda_warp_context<dense_cuda_register_frontier<pool_slots_>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(new_node_id) * 33 + lane)};
        dense_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + (new_node_id - view.first_id) + 1};
        dense_cuda_insert_one<kit_>( //
            view, new_node_id, visited, query_span, cache, relink, cache_norms[warp], gathered_ids[warp],
            gathered_scored[warp], context, lane);
        context.telemetry.flush<true>(view.telemetry, lane);
    }
}

/** @brief Geometry & plane pointers one refinement launch operates on: the live planes it
 *      reads, and the shadow neighbor plane it writes - same @b [rows,connectivity] shape,
 *      never observed by any concurrent reader. */
template <dense_cuda_kit kit_>
struct dense_cuda_refine_view {
    dense_cuda_planes_view<kit_> planes;
    /** @brief Refined neighbor rows land here, @b [rows,connectivity] ids - plain stores,
     *      published wholesale by the host after the launch drains. */
    u32_t* shadow_neighbors;
    /** @brief Ids strictly below this bound are refined by this launch. */
    u32_t end_id;
    /** @brief Bytes of the per-warp dynamic-shared slice - the insertion path's shape. */
    u32_t scratch_bytes_per_warp;
    /** @brief The pass's base id: the kernel's cursor hands out offsets in @b [first_id,end_id). */
    dense_node_id_t first_id;
    /** @brief The traversal arena and each warp's slice of it - the exact visited set. */
    std::byte* arena;
    dense_cuda_arena_shape_t arena_shape;
    /** @brief The context's monotone epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    dense_cuda_telemetry_block_t* telemetry;
    /** @brief This launch's work-queue head: offsets within the pass, zeroed per launch. */
    u32_t* batch_cursor;
    /** @brief Optional dense id list to refine instead of the @b [first_id,end_id) range -
     *      the island-healing path; null refines the range. @c end_id then bounds the
     *      list length, and dispensed offsets index the list. */
    u32_t const* node_list = nullptr;
    /** @brief Published-node ceiling for the walk's landmark sampling - always the graph
     *      size, never @c end_id: under a node list @c end_id is a LIST length, and a
     *      sampler bounded by it would draw landmarks from a sliver of the id space. */
    u32_t sampling_bound = 0;
};

/**
 *  @brief Re-selects the links of the node identified by @p node_id: stages its own stored
 *      vector as the query, seeds the walk with its current row - so every edge the
 *      incremental build wired, forward or reverse, competes in the re-selection instead
 *      of being forgotten - walks the graph wider, and writes the diverse selection into
 *      the shadow row. Read-only on the live planes: no locks, no publication.
 */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_cuda_refine_one( //
    dense_cuda_refine_view<kit_> const& view, dense_node_id_t node_id, dense_cuda_visited_set_t visited,
    typename kit_::fragment_t* query_span, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_relink_scratch_t relink, typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k],
    u32_t (&gathered_ids)[dense_cuda_cache_rows_k], dense_cuda_scored_node_t (&gathered_scored)[32],
    dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;

    // The query is the node's own stored vector; its norms are already derived state.
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
        query_span[fragment] = planes.vector_for(node_id, fragment);
    typename kit_::query_state_t const query_state = kit_::make_query_state(planes.norm_for(node_id));
    __syncwarp();

    // The node itself never joins its own candidate pool - visited before anything stages.
    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane, true);

    // Seed the frontier with the current row: a published non-empty row stages its links
    // exactly like a traversal expansion, minus the hop bookkeeping.
    dense_cuda_neighbors_ref_t const own_row {planes.neighbors_for(node_id), planes.connectivity};
    dense_node_id_t first_value = dense_node_id_unknown_k;
    if (lane == 0) first_value = own_row.peek_first();
    first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
    if (first_value != dense_node_id_unknown_k && first_value != node_id)
        for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) {
            u32_t const slot = slot_base + lane;
            dense_node_id_t const seed_id = slot == 0                    ? first_value
                                            : slot < planes.connectivity ? own_row.load_slot(slot)
                                                                         : dense_node_id_unknown_k;
            bool const usable = seed_id != dense_node_id_unknown_k && seed_id != node_id;
            bool const fresh = visited.test_and_set(seed_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) gathered_ids[ordinal] = seed_id;
            __syncwarp();
            u32_t const gathered_count = __popc(fresh_mask);
            if (gathered_count)
                dense_cuda_score_into_frontier<kit_>( //
                    planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
        }

    // Walk the graph past the seeds, then re-select the links from the whole pool.
    dense_cuda_search_ahead<kit_>( //
        planes, query_span, query_state, view.sampling_bound, visited, gathered_ids, gathered_scored, context, lane);
    u32_t const selected_count = dense_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered_ids, relink.selected_ids, context, lane);

    // Land the refined row in the shadow plane - the insertion row convention: links
    // ascending, sentinel tail, a self-loop in slot 0 encoding published-empty.
    u32_t* const shadow_row = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot = lane; slot < planes.connectivity; slot += 32)
        shadow_row[slot] = slot < selected_count ? relink.selected_ids[slot]
                           : slot == 0           ? node_id
                                                 : dense_node_id_unknown_k;
    __syncwarp();
}

/** @brief One warp refines one node, dispensing node ids from the pass cursor - the bulk
 *      post-processing sweep over a fully published graph. Grid-safe by construction:
 *      warps only read the live planes and write disjoint shadow rows. */
template <dense_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_sm_k)
    dense_cuda_refine_kernel(dense_cuda_refine_view<kit_> view) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_cuda_relink_scratch_t const relink = dense_cuda_relink_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_cuda_cache_rows_k);
    while (true) {
        u32_t offset;
        if (lane == 0) offset = atomicAdd(view.batch_cursor, 1u);
        offset = __shfl_sync(0xFFFFFFFFu, offset, 0);
        // Under a node list, end_id bounds the list; the dispensed offset indexes it.
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_list ? view.node_list[offset] : view.first_id + offset;

        dense_cuda_warp_context<dense_cuda_register_frontier<pool_slots_>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(node_id) * 33 + lane)};
        dense_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + offset + 1};
        dense_cuda_refine_one<kit_>( //
            view, node_id, visited, query_span, cache, relink, cache_norms[warp], gathered_ids[warp],
            gathered_scored[warp], context, lane);
        context.telemetry.flush<false>(view.telemetry, lane);
    }
}

/** @brief Lands a published-empty shadow row for a node whose live row offers no seeds -
 *      unreachable either way, and the merge pass may still wire reverse edges into it. */
template <dense_cuda_kit kit_>
__device__ __forceinline__ void dense_cuda_shadow_self_loop(dense_cuda_refine_view<kit_> const& view,
                                                            dense_node_id_t node_id, int lane) noexcept {
    u32_t* const shadow_row = view.shadow_neighbors + static_cast<std::size_t>(node_id) * view.planes.connectivity;
    for (u32_t slot = lane; slot < view.planes.connectivity; slot += 32)
        shadow_row[slot] = slot == 0 ? node_id : dense_node_id_unknown_k;
    __syncwarp();
}

/**
 *  @brief Re-selects the links of @p node_id from its two-hop neighborhood alone - the
 *      local join: every neighbor and neighbor-of-neighbor is scored against the node's
 *      own vector, the pool keeps the nearest, and the diverse selection lands in the
 *      shadow row. No walk, no landmarks - the pass streams rows instead of chasing
 *      pointers, so it costs a fraction of a walk-seeded refinement and iterates well:
 *      each pass propagates candidates one hop further, NN-descent's convergence engine.
 */
template <dense_cuda_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_cuda_join_one( //
    dense_cuda_refine_view<kit_> const& view, dense_node_id_t node_id, dense_cuda_visited_set_t visited,
    typename kit_::fragment_t* query_span, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_relink_scratch_t relink, typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k],
    u32_t (&gathered_ids)[dense_cuda_cache_rows_k], dense_cuda_scored_node_t (&gathered_scored)[32],
    dense_cuda_warp_context<frontier_>& context, int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;

    // The query is the node's own stored vector; its norms are already derived state.
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
        query_span[fragment] = planes.vector_for(node_id, fragment);
    typename kit_::query_state_t const query_state = kit_::make_query_state(planes.norm_for(node_id));
    __syncwarp();

    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane, true);

    // Hop one: stage the node's own row, remembering its ids for the second hop in the
    // selected-ids span - free scratch until the diverse selection claims it below.
    u32_t const hop_capacity = min_of(planes.connectivity, static_cast<u32_t>(dense_cuda_cache_rows_k));
    u32_t hop1_count = 0;
    dense_cuda_neighbors_ref_t const own_row {planes.neighbors_for(node_id), planes.connectivity};
    dense_node_id_t first_value = dense_node_id_unknown_k;
    if (lane == 0) first_value = own_row.peek_first();
    first_value = __shfl_sync(0xFFFFFFFFu, first_value, 0);
    if (first_value == dense_node_id_unknown_k || first_value == node_id) {
        dense_cuda_shadow_self_loop(view, node_id, lane);
        return;
    }
    for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) {
        u32_t const slot = slot_base + lane;
        dense_node_id_t const seed_id = slot == 0                    ? first_value
                                        : slot < planes.connectivity ? own_row.load_slot(slot)
                                                                     : dense_node_id_unknown_k;
        bool const usable = seed_id != dense_node_id_unknown_k && seed_id != node_id;
        bool const fresh = visited.test_and_set(seed_id, lane, usable);
        unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
        u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
        if (fresh) gathered_ids[ordinal] = seed_id;
        if (fresh && hop1_count + ordinal < hop_capacity) relink.selected_ids[hop1_count + ordinal] = seed_id;
        __syncwarp();
        u32_t const gathered_count = __popc(fresh_mask);
        hop1_count = min_of(hop1_count + gathered_count, hop_capacity);
        if (gathered_count)
            dense_cuda_score_into_frontier<kit_>( //
                planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
    }

    // Hop two: expand each first-hop row, staging only the fresh ids.
    for (u32_t hop1_index = 0; hop1_index != hop1_count; ++hop1_index) {
        dense_node_id_t const neighbor_id = relink.selected_ids[hop1_index];
        dense_cuda_neighbors_ref_t const row {planes.neighbors_for(neighbor_id), planes.connectivity};
        dense_node_id_t neighbor_first = dense_node_id_unknown_k;
        if (lane == 0) neighbor_first = row.peek_first();
        neighbor_first = __shfl_sync(0xFFFFFFFFu, neighbor_first, 0);
        if (neighbor_first == dense_node_id_unknown_k || neighbor_first == neighbor_id) continue;
        for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) {
            u32_t const slot = slot_base + lane;
            dense_node_id_t const candidate_id = slot == 0                    ? neighbor_first
                                                 : slot < planes.connectivity ? row.load_slot(slot)
                                                                              : dense_node_id_unknown_k;
            bool const usable = candidate_id != dense_node_id_unknown_k && candidate_id != node_id;
            bool const fresh = visited.test_and_set(candidate_id, lane, usable);
            unsigned const fresh_mask = __ballot_sync(0xFFFFFFFFu, fresh);
            u32_t const ordinal = __popc(fresh_mask & ((1u << lane) - 1));
            if (fresh) gathered_ids[ordinal] = candidate_id;
            __syncwarp();
            u32_t const gathered_count = __popc(fresh_mask);
            if (gathered_count)
                dense_cuda_score_into_frontier<kit_>( //
                    planes, query_span, query_state, gathered_ids, gathered_scored, gathered_count, context, lane);
        }
    }

    // Re-select diversely from the pool & land the shadow row - the refinement convention.
    u32_t const selected_count = dense_cuda_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered_ids, relink.selected_ids, context, lane);
    u32_t* const shadow_row = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot = lane; slot < planes.connectivity; slot += 32)
        shadow_row[slot] = slot < selected_count ? relink.selected_ids[slot]
                           : slot == 0           ? node_id
                                                 : dense_node_id_unknown_k;
    __syncwarp();
}

/** @brief One warp joins one node's two-hop neighborhood, dispensing node ids from the
 *      pass cursor - the local sibling of @ref dense_cuda_refine_kernel. The pool is
 *      fixed at the diverse selection's own width: deeper pools would stage candidates
 *      the selection can never read. */
template <dense_cuda_kit kit_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_sm_k)
    dense_cuda_join_kernel(dense_cuda_refine_view<kit_> view) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {query_span, view.planes.row_fragments};
    dense_cuda_relink_scratch_t const relink = dense_cuda_relink_scratch_t::carve(
        slice.scratch, view.planes.connectivity,
        static_cast<std::size_t>(view.planes.row_fragments) * sizeof(typename kit_::fragment_t) *
            dense_cuda_cache_rows_k);
    while (true) {
        u32_t node_id;
        if (lane == 0) node_id = view.first_id + atomicAdd(view.batch_cursor, 1u);
        node_id = __shfl_sync(0xFFFFFFFFu, node_id, 0);
        if (node_id >= view.end_id) return;

        dense_cuda_warp_context<dense_cuda_register_frontier<2>> context;
        context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(node_id) * 33 + lane)};
        dense_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + (node_id - view.first_id) + 1};
        dense_cuda_join_one<kit_>( //
            view, node_id, visited, query_span, cache, relink, cache_norms[warp], gathered_ids[warp],
            gathered_scored[warp], context, lane);
        context.telemetry.flush<false>(view.telemetry, lane);
    }
}

/**
 *  @brief Scatters the reverse edges of the shadow rows: every forward pick @b u→v lands
 *      @b u in @b v's reverse bucket - the lock-free bulk twin of the insertion path's
 *      back-linker. Bucket appends past @p bucket_width are counted, never written; the
 *      merge pass below re-selects diversely from what landed, so overflow only sheds
 *      candidates a full row would likely prune anyway.
 */
static __global__ void dense_cuda_scatter_links_kernel(u32_t const* shadow_neighbors, u32_t rows_count,
                                                       u32_t connectivity, u32_t bucket_width, u32_t* bucket_ids,
                                                       u32_t* bucket_counts) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps)
        for (u32_t slot_base = 0; slot_base < connectivity; slot_base += 32) {
            u32_t const slot = slot_base + lane;
            u32_t const target = slot < connectivity ? shadow_neighbors[row * connectivity + slot]
                                                     : dense_node_id_unknown_k;
            if (target == dense_node_id_unknown_k || target == static_cast<u32_t>(row)) continue;
            u32_t const landing = atomicAdd(&bucket_counts[target], 1u);
            if (landing < bucket_width)
                bucket_ids[static_cast<std::size_t>(target) * bucket_width + landing] = static_cast<u32_t>(row);
        }
}

/**
 *  @brief Selects one node's final row from its forward picks and its reverse bucket -
 *      the classic diversity heuristic over the union, ascending by distance to the node,
 *      exactly the selection a CPU-faithful full-list back-link would have produced. The
 *      refined row lands in the live neighbor plane with plain stores: the pass owns the
 *      graph exclusively, and every write is a complete, well-formed row.
 */
template <dense_cuda_kit kit_>
__device__ __forceinline__ void dense_cuda_merge_links_one( //
    dense_cuda_refine_view<kit_> const& view, u32_t const* bucket_ids, u32_t const* bucket_counts, u32_t bucket_width,
    dense_node_id_t node_id, dense_cuda_vector_cache<typename kit_::fragment_t> cache,
    dense_cuda_scored_node_t* sorted_span, u32_t* refined_rows,
    typename kit_::norms_t (&cache_norms)[dense_cuda_cache_rows_k], u32_t (&gathered_ids)[dense_cuda_cache_rows_k],
    dense_cuda_scored_node_t (&gathered_scored)[32], dense_cuda_telemetry_t& telemetry, int lane) noexcept {

    dense_cuda_planes_view<kit_> const& planes = view.planes;
    // The last cache row hosts the node's own vector - the query every candidate scores
    // against - so at most the rows before it can host candidates.
    constexpr u32_t query_row_k = dense_cuda_cache_rows_k - 1;
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
        cache.at(query_row_k, fragment) = planes.vector_for(node_id, fragment);
    typename kit_::query_state_t const query_state = kit_::make_query_state(planes.norm_for(node_id));
    __syncwarp();

    // Stage the forward picks; the shadow row is this pass's own output, already distinct.
    u32_t candidates_count = 0;
    u32_t const* const shadow_row = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += 32) {
        u32_t const slot = slot_base + lane;
        u32_t const forward_id = slot < planes.connectivity ? shadow_row[slot] : dense_node_id_unknown_k;
        bool const usable = forward_id != dense_node_id_unknown_k && forward_id != node_id;
        unsigned const usable_mask = __ballot_sync(0xFFFFFFFFu, usable);
        u32_t const ordinal = __popc(usable_mask & ((1u << lane) - 1));
        if (usable) gathered_ids[candidates_count + ordinal] = forward_id;
        candidates_count += __popc(usable_mask);
        __syncwarp();
    }
    u32_t const forward_count = candidates_count;

    // Stage the reverse bucket, dropping ids the forward prefix already holds - mutual
    // picks land in both - and clamping the union to the candidate cache.
    u32_t const landed = min_of(bucket_counts[node_id], bucket_width);
    for (u32_t slot_base = 0; slot_base < landed && candidates_count < query_row_k; slot_base += 32) {
        u32_t const slot = slot_base + lane;
        u32_t const reverse_id = slot < landed ? bucket_ids[static_cast<std::size_t>(node_id) * bucket_width + slot]
                                               : dense_node_id_unknown_k;
        bool usable = reverse_id != dense_node_id_unknown_k && reverse_id != node_id;
        for (u32_t forward_index = 0; usable && forward_index != forward_count; ++forward_index)
            usable = gathered_ids[forward_index] != reverse_id;
        unsigned const usable_mask = __ballot_sync(0xFFFFFFFFu, usable);
        u32_t const ordinal = __popc(usable_mask & ((1u << lane) - 1));
        if (usable && candidates_count + ordinal < query_row_k) gathered_ids[candidates_count + ordinal] = reverse_id;
        candidates_count = min_of(candidates_count + __popc(usable_mask), query_row_k);
        __syncwarp();
    }

    // Score every candidate against the node & sort ascending; ids are cache-row indices.
    telemetry.scored += candidates_count;
    dense_cuda_sorted_buffer_t links_sorted {sorted_span, 0, query_row_k};
    for (u32_t base = 0; base < candidates_count; base += 32) {
        u32_t const batch_count = min_of(candidates_count - base, 32u);
        dense_cuda_scored_node_t const scored = dense_cuda_score_batch<kit_>( //
            planes, cache.row(query_row_k), query_state, gathered_ids, gathered_scored, batch_count, lane, base);
        dense_cuda_scored_node_t const sorted_batch = dense_cuda_sort32(
            static_cast<u32_t>(lane) < batch_count
                ? dense_cuda_scored_node_t::make(scored.distance(), base + lane) // Id = candidate index.
                : dense_cuda_scored_node_t::sentinel(),
            lane);
        links_sorted.insert_batch(sorted_batch, batch_count, lane);
    }

    // Cache the candidate vectors & norms for the pairwise diversity checks.
    u32_t const candidate_words = candidates_count * planes.row_fragments;
    for (u32_t word = lane; word < candidate_words; word += 32) {
        u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
        __pipeline_memcpy_async(&cache.at(cache_row, fragment),
                                planes.vector_span_for(gathered_ids[cache_row]) + fragment,
                                sizeof(typename kit_::fragment_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    for (u32_t cache_row = lane; cache_row < candidates_count; cache_row += 32)
        cache_norms[cache_row] = planes.norm_for(gathered_ids[cache_row]);
    __syncwarp();

    // The greedy diversity walk, ascending by distance to the node.
    u32_t refined_count = 0;
    for (u32_t merge_index = 0; merge_index != links_sorted.size_ && refined_count != planes.connectivity;
         ++merge_index) {
        dense_cuda_scored_node_t const entry = links_sorted.at(merge_index);
        f32_t const to_query = entry.distance();
        u32_t const candidate_row = entry.id();
        bool diverse = true;
        for (u32_t base = 0; base < refined_count && diverse; base += 32) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < refined_count) {
                u32_t const kept_row = refined_rows[base + lane];
                typename kit_::accumulator_t pair_accumulator = kit_::zero_accumulator();
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    pair_accumulator = kit_::accumulate_dot(cache.at(candidate_row, fragment),
                                                            cache.at(kept_row, fragment), pair_accumulator);
                f32_t const between = kit_::distance(kit_::partial_to_dot(pair_accumulator),
                                                     kit_::make_query_state(cache_norms[candidate_row]),
                                                     cache_norms[kept_row]);
                lane_diverse = between >= to_query;
                ++telemetry.scored;
            }
            diverse = __all_sync(0xFFFFFFFFu, lane_diverse);
        }
        if (diverse) {
            if (lane == 0) refined_rows[refined_count] = candidate_row;
            ++refined_count;
            __syncwarp();
        }
    }

    // Fill the remaining slots with the nearest pruned candidates, ascending - the
    // incremental path reaches the same density by appending reverse edges without a
    // diversity test until the row fills; a strictly diverse row would stay ~20% thinner
    // and measurably poorer to search.
    for (u32_t merge_index = 0; merge_index != links_sorted.size_ && refined_count != planes.connectivity;
         ++merge_index) {
        u32_t const candidate_row = links_sorted.at(merge_index).id();
        bool kept = false;
        for (u32_t base = 0; base < refined_count; base += 32) { // Warp-uniform bounds.
            bool const matches = base + lane < refined_count && refined_rows[base + lane] == candidate_row;
            kept |= __any_sync(0xFFFFFFFFu, matches);
        }
        if (kept) continue;
        if (lane == 0) refined_rows[refined_count] = candidate_row;
        ++refined_count;
        __syncwarp();
    }

    // Land the final row in the live plane - the insertion row convention.
    __syncwarp();
    u32_t* const live_row = planes.neighbors_for(node_id);
    for (u32_t slot = lane; slot < planes.connectivity; slot += 32)
        live_row[slot] = slot < refined_count ? gathered_ids[refined_rows[slot]]
                         : slot == 0          ? node_id
                                              : dense_node_id_unknown_k;
    __syncwarp();
}

/** @brief One warp merges one node's forward & reverse candidates into its final row,
 *      dispensing node ids from the pass cursor. */
template <dense_cuda_kit kit_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, dense_cuda_resident_blocks_per_sm_k)
    dense_cuda_merge_links_kernel(dense_cuda_refine_view<kit_> view, u32_t const* bucket_ids,
                                  u32_t const* bucket_counts, u32_t bucket_width) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ typename kit_::norms_t cache_norms[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    dense_cuda_vector_cache<typename kit_::fragment_t> const cache {
        reinterpret_cast<typename kit_::fragment_t*>(slice.scratch), view.planes.row_fragments};
    // Past the candidate cache: the ascending scored span, then the kept-row indices.
    std::size_t const cache_bytes = static_cast<std::size_t>(view.planes.row_fragments) *
                                    sizeof(typename kit_::fragment_t) * dense_cuda_cache_rows_k;
    dense_cuda_scored_node_t* const sorted_span = reinterpret_cast<dense_cuda_scored_node_t*>(slice.scratch +
                                                                                              cache_bytes);
    u32_t* const refined_rows = reinterpret_cast<u32_t*>(slice.scratch + cache_bytes +
                                                         sizeof(dense_cuda_scored_node_t) * dense_cuda_cache_rows_k);
    while (true) {
        u32_t node_id;
        if (lane == 0) node_id = view.first_id + atomicAdd(view.batch_cursor, 1u);
        node_id = __shfl_sync(0xFFFFFFFFu, node_id, 0);
        if (node_id >= view.end_id) return;

        dense_cuda_telemetry_t telemetry;
        dense_cuda_merge_links_one<kit_>( //
            view, bucket_ids, bucket_counts, bucket_width, node_id, cache, sorted_span, refined_rows, cache_norms[warp],
            gathered_ids[warp], gathered_scored[warp], telemetry, lane);
        telemetry.flush<false>(view.telemetry, lane);
    }
}

/**
 *  @brief One query's full search - stages the query into its shared span, walks the
 *      graph, folds the ranks, and stores the nearest @c count ids (and distances)
 *      ascending. The kernels below are thin drivers: they differ only in the frontier
 *      they hand in. Safe against concurrent insertions: the walk gates every row on its
 *      slot-0 word and never writes to the graph.
 */
template <dense_cuda_kit kit_, u32_t resident_blocks_, typename frontier_>
__device__ __forceinline__ void dense_cuda_search_one( //
    dense_cuda_search_view<kit_> const& view, u32_t query_index, typename kit_::fragment_t* query_span,
    dense_cuda_visited_set_t visited, frontier_ frontier, u32_t (&gathered_ids)[dense_cuda_cache_rows_k],
    dense_cuda_scored_node_t (&gathered_scored)[32], int lane) noexcept {

    typename kit_::fragment_t const* query_row = view.queries +
                                                 static_cast<std::size_t>(query_index) * view.planes.row_fragments;
    typename kit_::norms_accumulator_t norms_accumulator = kit_::zero_norms_accumulator();
    for (u32_t fragment = lane; fragment < view.planes.row_fragments; fragment += 32) {
        typename kit_::fragment_t const value = query_row[fragment];
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = dense_cuda_reduce_norms(norms_accumulator);
    norms_accumulator = dense_cuda_broadcast_norms(norms_accumulator);
    typename kit_::norms_t const query_norms = kit_::partials_to_norms(norms_accumulator);
    typename kit_::query_state_t const query_state = kit_::make_query_state(query_norms);
    __syncwarp();

    dense_cuda_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(query_index) * 33 + lane)};
    dense_cuda_search_around<kit_, resident_blocks_>( //
        view.planes, query_span, query_state, view.sampling_bound, visited, gathered_ids, gathered_scored, context,
        lane);
    // The rank extraction shuffles - every lane participates, only in-range ones store.
    // Ranks past the live prefix surface as sentinel ids, marking absent matches.
    context.frontier.finalize_ranks(lane);
    for (u32_t rank_base = 0; rank_base < view.count; rank_base += 32) {
        u32_t const rank = rank_base + lane;
        dense_cuda_scored_node_t const ranked_entry = context.frontier.rank_entry(rank);
        if (rank < view.count) {
            std::size_t const slot = static_cast<std::size_t>(query_index) * view.count + rank;
            view.result_ids[slot] = ranked_entry.id();
            if (view.result_distances) view.result_distances[slot] = kit_::convert_to_distance(ranked_entry.distance());
        }
    }
    context.telemetry.flush<false>(view.telemetry, lane);
}

/** @brief One warp per query over the register frontier - expansions within the ladder. */
template <dense_cuda_kit kit_, u32_t pool_slots_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, pool_slots_ <= 4 ? 4 : 3)
    dense_cuda_search_kernel(dense_cuda_search_view<kit_> view) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

    dense_cuda_warp_slice_t const slice = dense_cuda_carve_slice(view, dynamic_scratch);
    int const warp = slice.warp, lane = slice.lane;
    std::size_t const warp_index = slice.warp_index;
    u64_t* const visited_slots = slice.visited_slots;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    for (u32_t query_index = static_cast<u32_t>(warp_index); query_index < view.queries_count;
         query_index += gridDim.x * dense_cuda_warps_per_block_k) {
        dense_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + query_index + 1};
        dense_cuda_search_one<kit_, (pool_slots_ <= 4 ? 4u : 3u)>( //
            view, query_index, query_span, visited, dense_cuda_register_frontier<pool_slots_> {}, gathered_ids[warp],
            gathered_scored[warp], lane);
    }
}

/** @brief One warp per query past the register ladder's width: the same walk over the
 *      arena frontier, whose expansion budget @b is the requested expansion factor -
 *      unbounded by any register shape. Results cover the frontier's head, the exact
 *      nearest 64, so the facade admits this path only for counts within it. */
template <dense_cuda_kit kit_>
__global__ void __launch_bounds__(dense_cuda_warps_per_block_k * 32, 4)
    dense_cuda_search_wide_kernel(dense_cuda_search_view<kit_> view) {
    extern __shared__ std::byte dynamic_scratch[];
    __shared__ u32_t gathered_ids[dense_cuda_warps_per_block_k][dense_cuda_cache_rows_k];
    __shared__ dense_cuda_scored_node_t gathered_scored[dense_cuda_warps_per_block_k][32];

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
        dense_cuda_visited_set_t const visited {visited_slots, view.arena_shape.visited_hash_shift,
                                                view.epoch_base + query_index + 1};
        dense_cuda_search_one<kit_, 4u>(view, query_index, query_span, visited, frontier, gathered_ids[warp],
                                        gathered_scored[warp], lane);
    }
}

/** @brief Rebuilds the norms plane from the vector plane, one warp per row - norms are
 *      derived state and never serialized, so every load re-computes them where the
 *      vectors already are, instead of shipping redundant bytes. */
template <dense_cuda_kit kit_>
__global__ void dense_cuda_derive_norms_kernel(dense_cuda_planes_view<kit_> planes, u32_t rows_count) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        typename kit_::norms_accumulator_t accumulator = kit_::zero_norms_accumulator();
        for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += 32)
            accumulator = kit_::accumulate_norms(planes.vector_for(static_cast<dense_node_id_t>(row), fragment),
                                                 accumulator);
        accumulator = dense_cuda_reduce_norms(accumulator);
        if (lane == 0) planes.norm_ref_for(static_cast<dense_node_id_t>(row)) = kit_::partials_to_norms(accumulator);
    }
}

/** @brief Post-drain structural audit of the neighbor plane: every row published, every id
 *      in range, self-loop rows childless, no duplicate edges within a row. One warp scans
 *      one row, a warp-width of slots at a time; each bad row bumps @p violations once. */
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
            // Duplicate-edge scan: rotate every not-earlier batch past this lane's value.
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
            atomicAdd(reinterpret_cast<unsigned long long*>(violations), 1ull);
    }
}

/** @brief Marks @p seed_count stride-spread row bits in the visited bitset - the audit's
 *      landmark set, mirroring how the traversal samplers spread over the id space. */
static __global__ void dense_cuda_mark_seeds_kernel(u32_t* visited_words, u32_t rows_count, u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    atomicOr(&visited_words[row >> 5], 1u << (row & 31u));
}

/**
 *  @brief One level-synchronous sweep of forward reachability over the neighbor plane:
 *      every visited-but-unexpanded row marks its targets visited. Bits only ever turn
 *      on, and a row expands exactly once, so re-sweeping until the change counter stays
 *      zero converges to the exact forward-reachable set of the seeds - an audit over
 *      exact bitsets, never a lossy shortcut.
 */
static __global__ void dense_cuda_reachability_sweep_kernel(u32_t const* neighbors, u32_t rows_count,
                                                            u32_t connectivity, u32_t* visited_words,
                                                            u32_t* expanded_words, u32_t* changed) {
    std::size_t const warp_index = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) / 32;
    int const lane = threadIdx.x % 32;
    std::size_t const warps = gridDim.x * static_cast<std::size_t>(blockDim.x) / 32;
    for (std::size_t row = warp_index; row < rows_count; row += warps) {
        u32_t const bit = 1u << (row & 31u);
        if (!(visited_words[row >> 5] & bit) || (expanded_words[row >> 5] & bit)) continue;
        if (lane == 0) {
            atomicOr(&expanded_words[row >> 5], bit);
            atomicAdd(changed, 1u);
        }
        for (u32_t slot_base = 0; slot_base < connectivity; slot_base += 32) {
            u32_t const slot = slot_base + lane;
            u32_t const target = slot < connectivity ? neighbors[row * connectivity + slot] : dense_node_id_unknown_k;
            if (target != dense_node_id_unknown_k && target != static_cast<u32_t>(row) && target < rows_count)
                atomicOr(&visited_words[target >> 5], 1u << (target & 31u));
        }
    }
}

/** @brief Post-mortem for a launch that published fewer rows than it claimed: publishes every
 *      still-sentinel row of the claimed range as a published-empty self-loop.
 *
 *  The claim cannot be revoked once the admission latch is released - sibling batches may
 *  already own ids above it - so the range is made real instead of hidden, restoring the two
 *  invariants @c size() rests on: every id below the claim frontier is published, and the
 *  publication count catches up so no later batch's admission ramp stalls behind dead rows.
 *  The compare-and-swap targets the same slot-0 word the insertion path releases, so a warp
 *  that published concurrently always wins. Publication order is free, which is exactly what
 *  makes this sweep legal.
 */
static __global__ void dense_cuda_tombstone_kernel(u32_t* neighbors, u32_t connectivity, u32_t first_id, u32_t rows,
                                                   dense_cuda_control_t* control) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    u32_t const node_id = first_id + row;
    u32_t* const slot_zero = neighbors + static_cast<std::size_t>(node_id) * connectivity;
    if (atomicCAS(slot_zero, dense_node_id_unknown_k, node_id) == dense_node_id_unknown_k)
        atomicAdd(&control->published_count, 1u);
}

/** @brief Collects the rows whose visited bit stayed clear - the island population -
 *      into a dense id list; @p list_cursor returns its length. */
static __global__ void dense_cuda_reachability_islands_kernel(u32_t const* visited_words, u32_t rows_count,
                                                              u32_t* island_ids, u32_t* list_cursor) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count || (visited_words[row >> 5] & (1u << (row & 31u)))) return;
    island_ids[atomicAdd(list_cursor, 1u)] = row;
}

/**
 *  @brief Forces one in-edge per listed island: the island id lands in its nearest
 *      forward target's row - the first empty slot, or the last slot when full -
 *      bypassing the diversity judgment that kept dropping it. The heuristics build good
 *      graphs; this guarantees connected ones. Exclusive pass, plain stores.
 */
static __global__ void dense_cuda_force_link_kernel(u32_t* neighbors, u32_t connectivity, u32_t const* island_ids,
                                                    u32_t islands_count, u32_t const* visited_words) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const* const island_row = neighbors + static_cast<std::size_t>(island) * connectivity;
    // The nearest REACHABLE forward pick: an island's nearest neighbors are often its own
    // sibling islanders, and wiring into those moves nothing. A node with no reachable
    // pick waits for the next round, when a connected sibling can carry it.
    u32_t target = dense_node_id_unknown_k;
    for (u32_t slot = 0; slot != connectivity && target == dense_node_id_unknown_k; ++slot) {
        u32_t const pick = island_row[slot];
        if (pick == dense_node_id_unknown_k || pick == island) continue;
        if (visited_words[pick >> 5] & (1u << (pick & 31u))) target = pick;
    }
    if (target == dense_node_id_unknown_k) return;
    u32_t* const target_row = neighbors + static_cast<std::size_t>(target) * connectivity;
    // Full-row landings spread across the tail: near-duplicate island clusters share one
    // nearest mainland target, and a fixed landing slot would let each round keep only
    // its last writer - one healed island per target per round, unbounded rounds.
    u32_t const spread = min_of(16u, connectivity / 2);
    u32_t landing = connectivity - 1 - island_index % spread;
    for (u32_t slot = 0; slot != connectivity; ++slot) {
        u32_t const resident = target_row[slot];
        if (resident == island) return; // Already wired; the audit raced a heal.
        if (resident == dense_node_id_unknown_k || resident == target) {
            landing = slot;
            break;
        }
    }
    target_row[landing] = island;
}

/** @brief Folds the visited bitset into a reachable-row count. */
static __global__ void dense_cuda_reachability_count_kernel(u32_t const* visited_words, u32_t words_count,
                                                            u32_t rows_count, unsigned long long* reachable) {
    u32_t const word_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_index >= words_count) return;
    u32_t word = visited_words[word_index];
    // The bitset tail past the last row must not count.
    if (word_index == words_count - 1 && (rows_count & 31u)) word &= (1u << (rows_count & 31u)) - 1u;
    if (word) atomicAdd(reachable, static_cast<unsigned long long>(__popc(word)));
}

#endif // defined(__CUDACC__)

#if defined(__CUDACC__)

/**
 *  @brief One agent's device-side scratch for driving a CUDA index - the GPU twin of the
 *      CPU's caller-owned @c dense_traversal_context. The CPU context is a thread's
 *      NUMA-local scratch; this is a @b stream's pool-local scratch: it owns the ordering
 *      stream, the traversal arena, the batch staging, and the telemetry sink, all drawn
 *      from one allocator and grown once to the widest launch, so the index's verbs issue
 *      launches and never per-call allocations.
 *
 *  Cardinality inverts against the CPU: a context is a concurrent submission lane, two to
 *  four per index - one per stream - never one per task. Two contexts on one index share
 *  the read-only planes and the published counter, and nothing else: their arenas, staging,
 *  and telemetry are disjoint by construction, so an insert stream and a search stream
 *  never alias scratch.
 */
template <nk_dtype_t, nk_kernel_kind_t, cuda_arch_t, typename, typename, typename>
struct dense_vram_index;

template <cuda_allocator allocator_type_ = cuda_device_allocator<std::byte>>
struct dense_cuda_traversal_context {
    using allocator_t = allocator_type_;

    /** @brief The single-query verb lands this many nearest matches at most - the arena
     *      frontier's exact head width, and far past any interactive top-k. */
    inline static constexpr u32_t results_capacity_k = 64;

    /** @brief The single-query verb's landing: ascending @c dense_scored_node_t matches,
     *      the CPU context's @c results in miniature - inline storage, no allocation. */
    struct results_t {
        dense_scored_node_t entries_[results_capacity_k];
        std::size_t count_ = 0;

        dense_scored_node_t* data() noexcept { return entries_; }
        dense_scored_node_t const* data() const noexcept { return entries_; }
        std::size_t size() const noexcept { return count_; }
    };

    results_t results;

    /** @brief Creates the context's ordering stream, binds a working allocator to it on
     *      @p allocator's device - the argument carries placement only - and allocates the
     *      fixed-size telemetry sink; the sized buffers wait for the first verb, so
     *      construction is cheap and @c noexcept. */
    dense_cuda_traversal_context(allocator_t allocator = {}) noexcept
        : allocator_ {stream_, allocator.device()},
          telemetry_ {cuda_make_unique(telemetry_allocator_t {allocator_}, 1)},
          batch_cursor_ {cuda_make_unique(cursor_allocator_t {allocator_}, 1)} {
        if (telemetry_) cudaMemsetAsync(telemetry_.get(), 0, sizeof(dense_cuda_telemetry_block_t), stream_);
    }
    dense_cuda_traversal_context(dense_cuda_traversal_context const&) = delete;
    dense_cuda_traversal_context& operator=(dense_cuda_traversal_context const&) = delete;
    /** @brief Every owned resource is a stream handle or a @c cuda_unique_ptr, so moves are
     *      member-wise steals and destruction is declaration-driven: the owning pointers -
     *      declared after @c stream_ - free onto the still-live stream, then the stream
     *      wrapper drains and destroys it. */
    dense_cuda_traversal_context(dense_cuda_traversal_context&&) noexcept = default;
    ~dense_cuda_traversal_context() noexcept = default;

    cudaStream_t stream() const noexcept { return stream_; }
    int device() const noexcept { return allocator_.device(); }

    /** @brief Blocking host copy of this context's telemetry sink. */
    dense_cuda_telemetry_block_t telemetry_snapshot() const noexcept {
        dense_cuda_telemetry_block_t snapshot = {};
        cudaMemcpyAsync(&snapshot, telemetry_.get(), sizeof(snapshot), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        return snapshot;
    }

  private:
    /** @brief Only the index's verbs reach the raw scratch - callers observe a context
     *      through @c results and the snapshots, never through its buffers. */
    template <nk_dtype_t, nk_kernel_kind_t, cuda_arch_t, typename, typename, typename>
    friend struct dense_vram_index;

    using telemetry_allocator_t = typename allocator_t::template rebind<dense_cuda_telemetry_block_t>::other;
    using cursor_allocator_t = typename allocator_t::template rebind<u32_t>::other;
    using bytes_ptr_t = cuda_unique_ptr<std::byte, allocator_t>;

    cuda_stream_t stream_;
    allocator_t allocator_;
    /** @brief Monotone epoch floor over the arena's visited slots: every launch's epochs
     *      ride above it, so the arena is zeroed only when freshly allocated or when the
     *      32-bit epoch space wraps - never per launch. */
    u64_t epoch_floor_ = 0;
    cuda_unique_ptr<dense_cuda_telemetry_block_t, telemetry_allocator_t> telemetry_;
    cuda_unique_ptr<u32_t, cursor_allocator_t> batch_cursor_;
    bytes_ptr_t arena_;
    std::size_t arena_bytes_ = 0;
    bytes_ptr_t staging_;
    std::size_t staging_bytes_ = 0;

    /** @brief Grow-only traversal arena for @p bytes - the exact visited sets and spill
     *      tails of every resident warp; kept across calls, like the CPU context keeps
     *      its larger link scratches. @return The base, or null on exhaustion. */
    std::byte* ensure_arena(std::size_t bytes) noexcept { return ensure_(arena_, arena_bytes_, bytes); }
    /** @brief Grow-only staging for one batch's device transport - an insert's
     *      fragment-padded vectors, or a search's queries & result landing. */
    std::byte* ensure_staging(std::size_t bytes) noexcept { return ensure_(staging_, staging_bytes_, bytes); }

    std::byte* ensure_(bytes_ptr_t& buffer, std::size_t& capacity, std::size_t bytes) noexcept {
        if (bytes <= capacity) return buffer.get();
        buffer = cuda_make_unique(allocator_, bytes);
        capacity = buffer ? bytes : 0;
        return buffer.get();
    }
};

/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  Mirrors the RAM-resident dense index verb-for-verb - @c reserve, @c insert,
 *  @c search - with two deliberate divergences, each documented where it lives: a batch's
 *  assigned ids are the contiguous new_node_id range (no per-vector id span), and
 *  publication is unordered - see @ref dense_vram_index_concurrency.
 *
 *  @section dense_vram_index_growth Growth
 *
 *  The first successful @c reserve() freezes the virtual-address ceiling; physical slabs
 *  map on demand ahead of every launch, and fresh neighbor rows are sentinel-filled
 *  before any new_node_id covering them can be dispensed.
 */
template <                                                                         //
    nk_dtype_t vector_dtype_ = nk_i8_k,                                            //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                           //
    cuda_arch_t arch_ceiling_ = cuda_arch_blackwell_k,                             //
    typename vectors_allocator_type_ = cuda_virtual_memory_allocator<std::byte>,   //
    typename neighbors_allocator_type_ = cuda_virtual_memory_allocator<std::byte>, //
    typename traversals_allocator_type_ = cuda_device_allocator<std::byte>>        //
struct dense_vram_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using kit_chain_t = dense_cuda_kit_chain<vector_dtype_, kernel_kind_, arch_ceiling_>;
    using kit_t = typename kit_chain_t::kit_t;
    using vectors_allocator_t = vectors_allocator_type_;
    using neighbors_allocator_t = neighbors_allocator_type_;
    using traversals_allocator_t = traversals_allocator_type_;
    using scalar_t = typename kit_t::scalar_t;
    using fragment_t = typename kit_t::fragment_t;
    using norms_t = typename kit_t::norms_t;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_view_t = nk::vector_view<scalar_t>;
    using node_id_t = dense_node_id_t;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using distance_t = dense_distance_t;
    using small_size_t = u32_t;
    using header_t = dense_index_header_t;
    using planes_view_t = dense_cuda_planes_view<kit_t>;
    using dense_traversal_context_t = dense_cuda_traversal_context<traversals_allocator_t>;

    /** @brief The strongest @c synchronization_t contract this design honors - purely
     *      documentation, nothing routes on it. One graph, many concurrent agents: batches
     *      claim disjoint dense id ranges under the admission latch, dispense offsets from
     *      per-launch cursors, and the all-RMW publication protocol makes overlapping
     *      searches safe. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    /** @brief Capability labels for the shared reports: the GPU picks no CPU SIMD tier. */
    inline static constexpr nk_capability_t picked_quads_capability_k = nk_cap_serial_k;
    /** @brief The generation row the kit chain selected under this instantiation's ceiling -
     *      the GPU twin of @c picked_quads_capability_k, surfaced in the bench reports. */
    inline static constexpr cuda_arch_t picked_arch_k = kit_chain_t::picked_arch_k;
    inline static constexpr bool has_tiles_k = false;
    inline static constexpr nk_capability_t picked_tiles_capability_k = nk_cap_serial_k;

    inline static constexpr std::size_t max_node_count_k = dense_cuda_scored_node_t::id_capacity_k;
    /** @brief Default visited-set slot count: a runtime knob now, this is only the
     *      heuristic floor - the scratch slice must also fit the candidate cache. */
    inline static constexpr u32_t default_visited_capacity_k = 2048;

    /** @brief The once-frozen geometry word's type, the CPU index's exact packing:
     *      dimensions in the high half, connectivity in the low, zero meaning "not yet
     *      reserved" - so one atomic word answers both "what shape" and "is it frozen". */
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

    /** @brief Bytes of one packed logical row, counted bit-exactly through NumKong's own
     *      reflection - sub-byte types pack several dimensions per storage value, and
     *      @c bits_per_dimension already knows every packing. */
    std::size_t packed_row_bytes() const noexcept {
        return divide_round_up<std::size_t>(dimensions() * scalar_t::bits_per_dimension(), 8u);
    }
    /** @brief Fragments per vector row - derived from the frozen dimensions, like every
     *      row measure below: the geometry word is the only stored source of truth. */
    u32_t row_fragments() const noexcept {
        return divide_round_up<u32_t>(static_cast<u32_t>(packed_row_bytes()), kit_t::fragment_bytes_k);
    }
    /** @brief The vector row's byte length: the dimension count rounded up to whole
     *      fragments, tail bytes kept zero so every dot-derived metric is unaffected. */
    std::size_t row_bytes() const noexcept {
        return static_cast<std::size_t>(row_fragments()) * kit_t::fragment_bytes_k;
    }
    /** @brief Runtime scoring geometry: lanes cooperating on one row (the smallest power
     *      of two covering its fragments, at most a warp), and their steps across it. */
    u32_t lanes_per_row() const noexcept { return min_of(std::bit_ceil(row_fragments()), 32u); }
    u32_t steps_per_row() const noexcept { return divide_round_up(row_fragments(), lanes_per_row()); }

    /** @brief Bytes of one warp's dynamic-shared slice on the insertion path: the query
     *      span while traversing, then the candidate cache & re-selection spans. */
    u32_t insert_scratch_bytes_per_warp() const noexcept {
        u32_t const relink_slots = min_of(static_cast<u32_t>(connectivity()), //
                                          static_cast<u32_t>(dense_cuda_cache_rows_k));
        u32_t const relink_bytes = relink_slots * static_cast<u32_t>(sizeof(dense_cuda_scored_node_t) + //
                                                                     2 * sizeof(u32_t));
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * dense_cuda_cache_rows_k;
        return max_of(static_cast<u32_t>(row_bytes()), cache_bytes + relink_bytes);
    }

    /** @brief Bytes of one warp's dynamic-shared slice on the refinement merge path: the
     *      candidate cache - whose last row hosts the node's own vector - then the
     *      ascending scored span and the kept-row indices. */
    u32_t merge_scratch_bytes_per_warp() const noexcept {
        u32_t const cache_bytes = static_cast<u32_t>(row_bytes()) * dense_cuda_cache_rows_k;
        return cache_bytes +
               dense_cuda_cache_rows_k * static_cast<u32_t>(sizeof(dense_cuda_scored_node_t) + sizeof(u32_t));
    }

    /** @brief Which frontier a warp's arena slice serves: the register ladder needs only
     *      the exact visited set, the wide arena frontier adds its spill tail. */
    enum class frontier_kind_t : u32_t { register_ladder_k, wide_arena_k };

    /** @brief One warp's arena slice for a walk expanding @p expansion candidates: the
     *      exact visited set - sized well past the expected visit count, so linear
     *      probing stays short - plus the spill tail when @p frontier_kind outgrows
     *      registers. */
    static dense_cuda_arena_shape_t arena_shape_for(u32_t expansion, frontier_kind_t frontier_kind) noexcept {
        u32_t const visited_slots = std::bit_ceil(64 * expansion);
        return {visited_slots, 32 - static_cast<u32_t>(std::bit_width(visited_slots - 1)),
                frontier_kind == frontier_kind_t::wide_arena_k ? 32 * expansion : 0};
    }

    /** @brief Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        std::size_t dimensions = 0;
    };
    /** @brief Outcome of a @c reserve() call: the capacity & connectivity actually in effect. */
    struct reserve_result_t {
        std::size_t new_capacity = 0;
        connectivity_t connectivity = connectivity_unknown_k;
        status_t status = status_t::success_k;
    };
    /** @brief Optional tuning knobs; zeros mean "derive from connectivity". */
    struct insert_config_t {
        /** @brief Breadth of the construction candidate pool; rounded up to a multiple of 32,
         *      clamped to 512. Larger pools trade build speed for graph quality. */
        small_size_t expansion_factor = 0;
    };
    /** @brief Outcome & telemetry of one @c insert() batch. Ids are the contiguous range
     *      @b [first_id,first_id+count) - the dispensed order of the incoming rows. */
    struct insert_result_t {
        status_t status = status_t::success_k;
        std::size_t new_size = 0;
        dense_node_id_t first_id = dense_node_id_unknown_k;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        /** @brief Reverse edges abandoned after the bounded lock budget - GPU-only telemetry. */
        std::size_t abandoned_back_links = 0;
        /** @brief The CPU splits refine-side distance work out of @c computed_distances;
         *      the GPU folds it all into one counter, so these report zero. */
        std::size_t computed_distances_in_refines = 0;
        std::size_t computed_distances_in_reverse_refines = 0;
        explicit operator bool() const noexcept { return succeeded(status); }
    };
    /** @brief Where a refinement pass draws its candidates from. */
    enum class refine_scope_t : u32_t {
        /** @brief A fresh graph walk per node, seeded by its current row - the insertion
         *      traversal re-run over the finished graph; @c expansion_factor applies. */
        global_k,
        /** @brief The node's two-hop neighborhood alone - no walk, a fraction of the
         *      cost, converging over iterated passes like NN-descent. */
        local_k,
        /** @brief No fresh candidates at all: the existing forward & reverse edges are
         *      re-judged together per node - the cheapest pass, wiring in the reverse
         *      links the bounded-wait back-linker abandoned and re-balancing full rows. */
        repair_k,
    };
    /** @brief Optional tuning knobs of @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {
        /** @brief Breadth of the re-selection candidate pool - the insertion semantic:
         *      rounded up to a multiple of 32, clamped to 512. Ignored by local passes,
         *      whose candidate set is the two-hop neighborhood itself. */
        small_size_t expansion_factor = 0;
        refine_scope_t scope = refine_scope_t::global_k;
    };
    /** @brief Outcome & telemetry of one @c refine() pass over the whole graph. */
    struct refine_result_t {
        status_t status = status_t::success_k;
        std::size_t refined_nodes = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        explicit operator bool() const noexcept { return succeeded(status); }
    };
    struct search_config_t {
        /** @brief Breadth of the candidate pool; defaults to twice the connectivity, and is
         *      raised automatically when @c count needs a wider pool. */
        small_size_t expansion_factor = 0;
        /** @brief Matches reported per query, up to the widest supported pool of 512. */
        small_size_t count = 10;
        /** @brief The caller's mutation contract. The GPU walk is insert-safe by protocol
         *      - every row gates on its slot-0 acquire - so all three levels share one
         *      code path and the field only documents intent. */
        synchronization_t synchronization = synchronization_t::concurrent_k;
    };
    struct search_result_t {
        status_t status = status_t::success_k;
        std::size_t count = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        std::size_t restarts = 0;
        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /** @brief Mirrors the CPU index's construction: one allocator per plane concern, each
     *      carrying its device placement; the index pins itself to the vectors allocator's
     *      GPU and demands the neighbors allocator agree. Allocates nothing here. */
    dense_vram_index(vectors_allocator_t const& vectors_allocator = {},
                     neighbors_allocator_t const& neighbors_allocator = {}) noexcept
        : device_ {vectors_allocator.device()}, vectors_plane_(vectors_allocator), norms_plane_(vectors_allocator),
          neighbors_plane_(neighbors_allocator) {}
    dense_vram_index(dense_vram_index const&) = delete;
    dense_vram_index& operator=(dense_vram_index const&) = delete;

    /** @brief Claimed rows - the id frontier. The CPU's @c size() reads its publication
     *      watermark instead; GPU publication is unordered, so the claim frontier is the
     *      only dense bound, and the sampler probes past unpublished ids by protocol. */
    std::size_t size() const noexcept { return admission_.claimed_rows.load(std::memory_order_acquire); }
    std::size_t connectivity() const noexcept {
        return geometry_connectivity(geometry_.load(std::memory_order_acquire));
    }
    std::size_t dimensions() const noexcept { return geometry_dimensions(geometry_.load(std::memory_order_acquire)); }
    /** @brief Device-side plane views - the audit surface for invariant checkers & tests. */
    /** @brief The pinned device's identity & limits - the index group sizes launches and
     *      routes streams by it, instead of re-querying the driver per pointer. */
    cuda_device_specs_t const& device_specs() const noexcept { return device_; }

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

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries. The first
     *      successful call freezes the geometry and the virtual-address ceiling; later
     *      calls with matching geometry only map further physical slabs.
     */
    reserve_result_t reserve(reserve_capacity_t config) noexcept {
        // Both plane allocators must place their memory on the index's pinned device.
        if (neighbors_plane_.device() != device_.ordinal)
            return {0, connectivity_unknown_k, status_t::config_mismatch_k};
        if (!config.dimensions || !config.connectivity) return {0, connectivity_unknown_k, status_t::config_mismatch_k};
        // The type's only compare-and-swap, on this cold path: the first success freezes
        // both flexible axes at once, exactly like the CPU index's geometry word; later
        // calls must agree with whatever they observe.
        packed_geometry_id_t const requested = pack_geometry(config.dimensions,
                                                             static_cast<std::size_t>(config.connectivity));
        packed_geometry_id_t observed = 0;
        if (!geometry_.compare_exchange_strong(observed, requested, std::memory_order_acq_rel) && observed != requested)
            return {mapped_rows(), static_cast<connectivity_t>(geometry_connectivity(observed)),
                    status_t::config_mismatch_k};
        connectivity_t const connectivity = config.connectivity;
        if (config.capacity > max_node_count_k) return {mapped_rows(), connectivity, status_t::overflow_k};
        if (!vectors_plane_.reserved_bytes()) { // First reservation: bind the device, size the ceiling.
            // Bind the runtime's primary context before any driver-API mapping - the VMM
            // planes must live in the same context the kernels launch into - and read the
            // device specs the launch grids and the ceiling policy consult.
            if (cudaSetDevice(device_.ordinal) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
                return {0, connectivity, status_t::device_error_k};
            expected<cuda_device_specs_t> const specs = cuda_device_specs_t::query(device_.ordinal);
            if (specs.status != status_t::success_k) return {0, connectivity, specs.status};
            device_ = specs.value;
            cuInit(0);
            // Virtual address space is free, so the ceiling is generous and independent of
            // the first requested capacity: geometry freezes here, row capacity stays
            // grow-only up to physical memory - a small first reserve() must not cap growth.
            // The reservation itself is the stored ceiling; no cached row count shadows it.
            std::size_t const bytes_per_row = row_bytes() + sizeof(norms_t) + connectivity * sizeof(u32_t);
            // Device-resident planes budget against VRAM; managed planes oversubscribe it,
            // so their backend budgets against host memory instead - the pager, not the
            // reservation, is what bounds a bigger-than-VRAM build. The constexpr branch
            // keeps device-plane instantiations free of the topology symbols.
            std::size_t budget_bytes = 4 * device_.vram_bytes;
            if constexpr (vectors_allocator_t::address_space_k == cuda_address_space_t::managed_k)
                budget_bytes = vectors_allocator_t::budget_bytes();
            std::size_t const ceiling_rows = min_of(max_node_count_k,
                                                    max_of<std::size_t>(config.capacity, budget_bytes / bytes_per_row));
            if (vectors_plane_.reserve_range(ceiling_rows * row_bytes(), device_.ordinal) != status_t::success_k ||
                norms_plane_.reserve_range(ceiling_rows * sizeof(norms_t), device_.ordinal) != status_t::success_k ||
                neighbors_plane_.reserve_range(ceiling_rows * connectivity * sizeof(u32_t), device_.ordinal) !=
                    status_t::success_k)
                return {0, connectivity, status_t::bad_alloc_k};
            control_allocator_t control_allocator {nullptr, device_.ordinal};
            control_ = cuda_make_unique(control_allocator, 1);
            if (!control_ || cudaMemset(control_.get(), 0, sizeof(dense_cuda_control_t)) != cudaSuccess)
                return {0, connectivity, status_t::device_error_k};
        }
        status_t const grown = grow_rows(config.capacity, nullptr);
        return {mapped_rows(), connectivity, grown};
    }

    /**
     *  @brief Inserts a batch of @p new_vectors, assigning them the contiguous id range
     *      @b [size(),size()+rows) in row order. Blocks until the batch is fully
     *      published; @p stream orders the transfers & the launch.
     */
    insert_result_t insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids, dense_traversal_context_t& context,
                           insert_config_t config = {}) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        dense_cuda_telemetry_block_t const before = context.telemetry_ ? context.telemetry_snapshot()
                                                                       : dense_cuda_telemetry_block_t {};
        insert_result_t result = insert_nowait(new_vectors, assigned_ids, context, config);
        if (!result || !rows) return result;
        status_t const confirmed = confirm_batch(context, result, rows);
        if (confirmed != status_t::success_k) return {confirmed, result.new_size, result.first_id};
        dense_cuda_telemetry_block_t const after = context.telemetry_snapshot();
        result.visited_members = after.hops - before.hops;
        result.computed_distances = after.scored - before.scored;
        result.restarts = after.restarts - before.restarts;
        result.abandoned_back_links = after.lock_skips - before.lock_skips;
        return result;
    }

    /**
     *  @brief Attests the context's most recent launch and repairs it if it fell short.
     *
     *  Drains the stream, then reads the per-launch batch cursor: every offset dispensed means
     *  every claimed id published, since give-up paths publish self-loop tombstones. A shortfall
     *  cannot un-claim the range - sibling batches may already own ids above it - so the still
     *  sentinel rows are tombstoned, keeping @c size() honest about a dense id space, and the
     *  caller is still told the batch failed. Scope is one launch: the cursor is per-context and
     *  re-zeroed per launch, so back-to-back @c insert_nowait calls attest only the last.
     *
     *  @param[inout] context The context whose stream and cursor the batch rode.
     *  @param[in] batch The result @c insert_nowait returned, for its claimed id range.
     *  @param[in] rows The row count that batch was given.
     */
    status_t confirm_batch(dense_traversal_context_t& context, insert_result_t const& batch,
                           std::size_t rows) noexcept {
        if (!rows) return status_t::success_k;
        if (cudaStreamSynchronize(context.stream()) != cudaSuccess) return status_t::device_error_k;
        u32_t dispensed = 0;
        if (cudaMemcpy(&dispensed, context.batch_cursor_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            return status_t::device_error_k;
        if (dispensed >= rows) return status_t::success_k;
        u32_t const block_threads_k = 256;
        u32_t const blocks = static_cast<u32_t>(divide_round_up<std::size_t>(rows, block_threads_k));
        dense_cuda_tombstone_kernel<<<blocks, block_threads_k, 0, context.stream()>>>(
            reinterpret_cast<u32_t*>(neighbors_plane_.base()), static_cast<u32_t>(connectivity()), batch.first_id,
            static_cast<u32_t>(rows), control_.get());
        [[maybe_unused]] cudaError_t const swept = cudaStreamSynchronize(context.stream());
        return status_t::device_error_k;
    }

    /**
     *  @brief Reports whether every claimed id is published - a derived health check, never
     *      cached state. Call it only with no batch in flight: a shortfall then names a
     *      poisoned context, the one failure no post-mortem sweep can reach.
     */
    [[nodiscard]] status_t validate() const noexcept {
        u32_t published = 0;
        if (cudaMemcpy(&published, &control_.get()->published_count, sizeof(u32_t), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return status_t::device_error_k;
        return published == size() ? status_t::success_k : status_t::device_error_k;
    }

    /**
     *  @brief The index group's ingest verb: @c insert without the trailing synchronization -
     *      every transfer and the launch queue on the context's stream, and the CALLER
     *      synchronizes that stream before reading results or reusing the host batch.
     *      That synchronization is where @c confirm_batch belongs: this verb sees only the
     *      failures that surface before the launch returns, so a caller that never confirms
     *      cannot learn that a launch ran short.
     *      One host thread drives a whole device group this way: back-to-back calls on
     *      one context stay safe because stream order serializes the staging reuse, and
     *      calls across contexts on different devices overlap freely. Ids are dense and
     *      known at claim time, so @p assigned_ids fills immediately; the traversal
     *      telemetry deltas are not reported - snapshot around the caller's sync when
     *      they matter.
     */
    insert_result_t insert_nowait(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                  dense_traversal_context_t& context, insert_config_t config = {}) noexcept {
        std::size_t const rows = new_vectors.extent(0);
        if (!rows) return {status_t::success_k, size(), static_cast<dense_node_id_t>(size())};
        if (!vectors_plane_.reserved_bytes()) return {status_t::uninitialized_k};
        if (new_vectors.extent(1) != dimensions()) return {status_t::config_mismatch_k};
        if (assigned_ids.size() && assigned_ids.size() != rows) return {status_t::config_mismatch_k};
        if (context.device() != device_.ordinal || !context.batch_cursor_ || !context.telemetry_)
            return {status_t::config_mismatch_k};
        cudaStream_t const stream = context.stream();
        // Every fallible preparation precedes the id claim: a claimed range that never
        // publishes would starve every later batch's admission ramp forever, so nothing
        // that can fail is allowed to run between the claim and the launch. The staged
        // batch and the arena depend only on batch-local offsets, never on first_id.
        std::byte* const staging = context.ensure_staging(rows * row_bytes());
        if (!staging) return {status_t::bad_alloc_k};
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (scalar_row_bytes != row_bytes()) cudaMemsetAsync(staging, 0, rows * row_bytes(), stream);
        if (cudaMemcpy2DAsync(staging, row_bytes(), new_vectors.byte_data(), scalar_row_bytes, scalar_row_bytes, rows,
                              cudaMemcpyHostToDevice, stream) != cudaSuccess)
            return {status_t::device_error_k};

        // Every inserting warp owns an arena slice for its exact visited set.
        u32_t const pool_slots = pool_slots_for(config.expansion_factor, 4);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(pool_slots * 32,
                                                                     frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.sm_count) * insert_blocks_per_sm_k *
                                        dense_cuda_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        std::byte* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {status_t::bad_alloc_k};

        // Grow-then-claim-then-launch under the cold admission latch. The claim stays
        // invisible until the launch is accepted: while the latch is held no sibling has
        // read `claimed_rows`, so a rejected grid - which publishes nothing - is un-claimed
        // exactly. That is the whole batch-atomicity argument, and it costs no counter: the
        // id frontier is simply not stored until it is earned. The latch already spans a
        // full stream drain inside `grow_rows`, so enclosing an async launch is cheaper
        // than what it already tolerates.
        while (admission_.latch.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
        std::size_t const first_id = admission_.claimed_rows.load(std::memory_order_relaxed);
        std::size_t const total_rows = first_id + rows;
        status_t claimed = grow_rows(total_rows, stream);
        if (claimed == status_t::success_k) {
            // Epoch discipline: zero marks never-used slots, and this launch's epochs ride
            // above the context's monotone floor - so only a fresh allocation or a wrapped
            // 32-bit epoch space warrants zeroing, never a warm arena.
            if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + rows > 0xFFFFFFFEull) {
                if (cudaMemsetAsync(arena, 0, context.arena_bytes_, stream) != cudaSuccess)
                    claimed = status_t::device_error_k;
                context.epoch_floor_ = 0;
            }
            u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
            context.epoch_floor_ += rows; // Monotone; an un-claim below only forfeits epoch space.
            if (claimed == status_t::success_k &&
                cudaMemsetAsync(context.batch_cursor_.get(), 0, sizeof(u32_t), stream) != cudaSuccess)
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
                                                   control_.get(),
                                                   context.telemetry_.get(),
                                                   context.batch_cursor_.get()};
                launch_insert(view, pool_slots, stream);
                // The launch is the last fallible step and it reports synchronously. Reading
                // the error here, rather than after the stream drains, is what keeps the
                // claim reversible - past this point no id can be handed back.
                if (cudaGetLastError() != cudaSuccess) claimed = status_t::device_error_k;
            }
            if (claimed == status_t::success_k) admission_.claimed_rows.store(total_rows, std::memory_order_release);
        }
        admission_.latch.clear(std::memory_order_release);
        if (claimed != status_t::success_k) return {claimed};

        insert_result_t result;
        result.first_id = static_cast<dense_node_id_t>(first_id);
        for (std::size_t row = 0; row != assigned_ids.size(); ++row)
            assigned_ids[row] = static_cast<dense_node_id_t>(first_id + row);
        result.new_size = total_rows;
        return result;
    }

    /**
     *  @brief Re-selects every node's neighbor list from a fresh walk over the finished
     *      graph - the bulk post-processing pass: build cheap & incremental, then recover
     *      the quality of an expensive build in one lock-free sweep. Each node's walk is
     *      seeded by its current row, so existing edges compete in the re-selection
     *      rather than being forgotten; a scatter-and-merge pair then re-wires the
     *      reverse edges the way the insertion path's back-linker would have - forward
     *      picks and reverse candidates judged together, diversely, per node. Demands an
     *      exclusive caller: the merge writes live rows with plain stores no concurrent
     *      search may observe half-written.
     */
    refine_result_t refine(dense_traversal_context_t& context, refine_config_t config = {}) noexcept {
        return refine_pass_(context, config, nullptr, 0);
    }

  private:
    /** @brief The three-pass refinement engine behind @c refine() and @c relink_islands():
     *      an optional device-resident id list narrows the walk to those nodes - over a
     *      live-copied shadow - while the reverse passes always sweep the whole graph. */
    refine_result_t refine_pass_(dense_traversal_context_t& context, refine_config_t config, u32_t const* node_list,
                                 u32_t list_count) noexcept {
        std::size_t const nodes = size();
        if (!nodes) return {};
        if (!vectors_plane_.reserved_bytes()) return {status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context.batch_cursor_ || !context.telemetry_)
            return {status_t::config_mismatch_k};
        cudaStream_t const stream = context.stream();
        // Pass-local planes, never observed by readers: the shadow rows of the forward
        // selection, the reverse buckets the scatter fills, and their landing counters.
        // The bucket width matches the connectivity - what a full final row can keep.
        std::size_t const shadow_bytes = nodes * connectivity() * sizeof(u32_t);
        u32_t const bucket_width = static_cast<u32_t>(connectivity());
        auto shadow = cuda_make_unique(context.allocator_, shadow_bytes * 2 + nodes * sizeof(u32_t));
        if (!shadow) return {status_t::bad_alloc_k};
        u32_t* const shadow_neighbors = reinterpret_cast<u32_t*>(shadow.get());
        u32_t* const bucket_ids = reinterpret_cast<u32_t*>(shadow.get() + shadow_bytes);
        u32_t* const bucket_counts = reinterpret_cast<u32_t*>(shadow.get() + shadow_bytes * 2);
        if (cudaMemsetAsync(bucket_counts, 0, nodes * sizeof(u32_t), stream) != cudaSuccess)
            return {status_t::device_error_k};

        u32_t const pool_slots = pool_slots_for(config.expansion_factor, 4);
        // A local pass visits at most the two-hop neighborhood; its visited set derives
        // from that ceiling instead of the walk expansion.
        u32_t const two_hop = static_cast<u32_t>(connectivity() * (connectivity() + 1));
        bool const local = config.scope == refine_scope_t::local_k;
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(
            local ? divide_round_up(two_hop, 16u) : pool_slots * 32, frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.sm_count) * insert_blocks_per_sm_k *
                                        dense_cuda_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        std::byte* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {status_t::bad_alloc_k};
        // Epoch discipline: zero marks never-used slots, and this pass's epochs ride above
        // the context's monotone floor - see the insertion verb's twin.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + nodes > 0xFFFFFFFEull) {
            cudaMemsetAsync(arena, 0, context.arena_bytes_, stream);
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += nodes;
        cudaMemsetAsync(context.batch_cursor_.get(), 0, sizeof(u32_t), stream);
        dense_cuda_telemetry_block_t const before = context.telemetry_snapshot();

        dense_cuda_refine_view<kit_t> view {planes(), //
                                            shadow_neighbors,
                                            static_cast<u32_t>(nodes),
                                            insert_scratch_bytes_per_warp(),
                                            0,
                                            arena,
                                            arena_shape,
                                            epoch_base,
                                            context.telemetry_.get(),
                                            context.batch_cursor_.get(),
                                            nullptr,
                                            static_cast<u32_t>(nodes)};
        // A repair pass draws no fresh candidates, and a listed pass re-walks only its
        // listed nodes: both start from the live rows copied into the shadow verbatim,
        // and the scatter-merge pair below re-judges whatever is wired.
        bool const repair = config.scope == refine_scope_t::repair_k;
        if (repair || node_list) {
            if (cudaMemcpyAsync(shadow_neighbors, neighbors_plane_.base(), shadow_bytes, cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess)
                return {status_t::device_error_k};
        }
        if (node_list) {
            view.node_list = node_list;
            view.end_id = list_count;
        }
        std::size_t const walk_count = node_list ? list_count : nodes;
        if (!repair) {
            if (local) launch_join(view, stream);
            else launch_refine(view, pool_slots, stream);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};
        // The pass cursor is the completion oracle: every offset dispensed means every
        // node's shadow row landed. A repair pass dispenses nothing before the merge.
        u32_t dispensed = 0;
        if (!repair && (cudaMemcpy(&dispensed, context.batch_cursor_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) !=
                            cudaSuccess ||
                        dispensed < walk_count))
            return {status_t::device_error_k};

        // Scatter the reverse edges, then merge each node's forward & reverse candidates
        // into its final live row - stream-ordered, one more cursor cycle.
        cudaMemsetAsync(context.batch_cursor_.get(), 0, sizeof(u32_t), stream);
        dense_cuda_scatter_links_kernel<<<device_.sm_count * insert_blocks_per_sm_k, block_threads_k, 0, stream>>>(
            shadow_neighbors, static_cast<u32_t>(nodes), static_cast<u32_t>(connectivity()), bucket_width, bucket_ids,
            bucket_counts);
        dense_cuda_refine_view<kit_t> merge_view = view;
        merge_view.scratch_bytes_per_warp = merge_scratch_bytes_per_warp();
        // The reverse passes always sweep the whole graph, whatever the walk covered.
        merge_view.node_list = nullptr;
        merge_view.end_id = static_cast<u32_t>(nodes);
        launch_merge(merge_view, bucket_ids, bucket_counts, bucket_width, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};
        if (cudaMemcpy(&dispensed, context.batch_cursor_.get(), sizeof(u32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
            dispensed < nodes)
            return {status_t::device_error_k};

        dense_cuda_telemetry_block_t const after = context.telemetry_snapshot();
        refine_result_t result;
        result.refined_nodes = nodes;
        result.visited_members = after.hops - before.hops;
        result.computed_distances = after.scored - before.scored;
        result.restarts = after.restarts - before.restarts;
        return result;
    }

  public:
    /** @brief Outcome of one @c audit_reachability() sweep set. */
    struct reachability_result_t {
        status_t status = status_t::success_k;
        /** @brief Rows forward-reachable from the landmark seeds - the searcher's view. */
        std::size_t reachable_nodes = 0;
        /** @brief Level-synchronous sweeps to convergence - a diameter proxy. */
        std::size_t sweeps = 0;
        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /**
     *  @brief Measures forward reachability from @p seed_count stride-spread landmarks -
     *      the connectivity audit: any shortfall against @c size() is the population no
     *      search can discover, however wide its expansion. Read-only on the graph and
     *      safe against concurrent searches; concurrent insertions may leave fresh rows
     *      uncounted.
     */
    reachability_result_t audit_reachability(dense_traversal_context_t& context,
                                             std::size_t seed_count = 1024) noexcept {
        std::size_t const nodes = size();
        if (!nodes) return {};
        typename dense_traversal_context_t::bytes_ptr_t plan;
        std::size_t words = 0;
        reachability_result_t result;
        if (status_t swept = sweep_reachability_(context, seed_count, plan, words, result.sweeps);
            swept != status_t::success_k)
            return {swept};
        cudaStream_t const stream = context.stream();
        u32_t* const visited_words = reinterpret_cast<u32_t*>(plan.get() + sizeof(unsigned long long));
        auto* const reachable = reinterpret_cast<unsigned long long*>(plan.get());
        u32_t const count_blocks = static_cast<u32_t>(divide_round_up(words, std::size_t {block_threads_k}));
        dense_cuda_reachability_count_kernel<<<count_blocks, block_threads_k, 0, stream>>>(
            visited_words, static_cast<u32_t>(words), static_cast<u32_t>(nodes), reachable);
        unsigned long long reachable_rows = 0;
        if (cudaMemcpyAsync(&reachable_rows, reachable, sizeof(reachable_rows), cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return {status_t::device_error_k};
        result.reachable_nodes = static_cast<std::size_t>(reachable_rows);
        return result;
    }

    /** @brief Outcome of one @c relink_islands() healing pass. */
    struct relink_result_t {
        status_t status = status_t::success_k;
        /** @brief Rows the audit found unreachable - all re-walked and re-wired. */
        std::size_t islands_found = 0;
        /** @brief Rows the diverse heal left unreached, each then force-linked into its
         *      nearest forward target - zero here means the heal alone sufficed. */
        std::size_t islands_forced = 0;
        std::size_t visited_members = 0;
        std::size_t computed_distances = 0;
        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /**
     *  @brief Audits reachability, then re-links every unreached node: each island member
     *      re-walks the graph and re-selects its row over a live-copied shadow, and the
     *      reverse merge wires fresh in-edges pointing at it - the connectivity guarantee
     *      pass. Demands an exclusive caller, like @c refine().
     */
    relink_result_t relink_islands(dense_traversal_context_t& context, refine_config_t config = {}) noexcept {
        std::size_t const nodes = size();
        if (!nodes) return {};
        typename dense_traversal_context_t::bytes_ptr_t plan;
        std::size_t words = 0, sweeps = 0;
        if (status_t swept = sweep_reachability_(context, 1024, plan, words, sweeps); swept != status_t::success_k)
            return {swept};
        cudaStream_t const stream = context.stream();
        u32_t* const visited_words = reinterpret_cast<u32_t*>(plan.get() + sizeof(unsigned long long));
        auto islands = cuda_make_unique(context.allocator_, nodes * sizeof(u32_t) + sizeof(u32_t));
        if (!islands) return {status_t::bad_alloc_k};
        u32_t* const island_ids = reinterpret_cast<u32_t*>(islands.get());
        u32_t* const list_cursor = island_ids + nodes;
        cudaMemsetAsync(list_cursor, 0, sizeof(u32_t), stream);
        u32_t const blocks = static_cast<u32_t>(divide_round_up<std::size_t>(nodes, block_threads_k));
        dense_cuda_reachability_islands_kernel<<<blocks, block_threads_k, 0, stream>>>( //
            visited_words, static_cast<u32_t>(nodes), island_ids, list_cursor);
        u32_t islands_count = 0;
        if (cudaMemcpyAsync(&islands_count, list_cursor, sizeof(u32_t), cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return {status_t::device_error_k};

        relink_result_t result;
        result.islands_found = islands_count;
        if (!islands_count) return result;
        config.scope = refine_scope_t::global_k; // The healing walk is a walk by definition.
        refine_result_t const healed = refine_pass_(context, config, island_ids, islands_count);
        if (!healed) return {healed.status};
        result.visited_members = healed.visited_members;
        result.computed_distances = healed.computed_distances;

        // The diverse heal loses the stubborn tail - island nodes whose reverse candidacy
        // every target's judgment drops. Audit-and-force rounds close it: each round
        // wires every still-unreached node into its nearest reachable forward pick, and
        // intra-island edges then carry reachability to the rest of its island on the
        // next sweep - so the unreached set strictly shrinks while picks remain.
        constexpr u32_t force_rounds_cap_k = 8;
        for (u32_t round = 0; round != force_rounds_cap_k; ++round) {
            plan = {};
            std::size_t forced_sweeps = 0;
            if (status_t swept = sweep_reachability_(context, 1024, plan, words, forced_sweeps);
                swept != status_t::success_k)
                return {swept};
            u32_t* const healed_words = reinterpret_cast<u32_t*>(plan.get() + sizeof(unsigned long long));
            cudaMemsetAsync(list_cursor, 0, sizeof(u32_t), stream);
            dense_cuda_reachability_islands_kernel<<<blocks, block_threads_k, 0, stream>>>( //
                healed_words, static_cast<u32_t>(nodes), island_ids, list_cursor);
            u32_t residual_count = 0;
            if (cudaMemcpyAsync(&residual_count, list_cursor, sizeof(u32_t), cudaMemcpyDeviceToHost, stream) !=
                    cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess)
                return {status_t::device_error_k};
            if (!round) result.islands_forced = residual_count;
            if (!residual_count) break;
            u32_t const force_blocks = static_cast<u32_t>(
                divide_round_up<std::size_t>(residual_count, block_threads_k));
            dense_cuda_force_link_kernel<<<force_blocks, block_threads_k, 0, stream>>>(
                reinterpret_cast<u32_t*>(neighbors_plane_.base()), static_cast<u32_t>(connectivity()), island_ids,
                residual_count, healed_words);
            if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};
        }
        return result;
    }

  private:
    /** @brief Seeds the visited bitset from stride-spread landmarks and sweeps to
     *      convergence. On success @p plan carries @b [reachable u64 | visited words |
     *      expanded words | changed u32] on the context's stream. */
    status_t sweep_reachability_(dense_traversal_context_t& context, std::size_t seed_count,
                                 typename dense_traversal_context_t::bytes_ptr_t& plan, std::size_t& words,
                                 std::size_t& sweeps) noexcept {
        std::size_t const nodes = size();
        if (!vectors_plane_.reserved_bytes()) return status_t::uninitialized_k;
        if (context.device() != device_.ordinal) return status_t::config_mismatch_k;
        cudaStream_t const stream = context.stream();
        seed_count = min_of(max_of<std::size_t>(seed_count, 1), nodes);

        words = divide_round_up<std::size_t>(nodes, 32);
        // The widest member leads the carve: the 8-byte reachable counter, then the two
        // word-aligned bitsets, then the change counter.
        std::size_t const plan_bytes = sizeof(unsigned long long) + words * 2 * sizeof(u32_t) + sizeof(u32_t);
        plan = cuda_make_unique(context.allocator_, plan_bytes);
        if (!plan) return status_t::bad_alloc_k;
        u32_t* const visited_words = reinterpret_cast<u32_t*>(plan.get() + sizeof(unsigned long long));
        u32_t* const expanded_words = visited_words + words;
        u32_t* const changed = expanded_words + words;
        if (cudaMemsetAsync(plan.get(), 0, plan_bytes, stream) != cudaSuccess) return status_t::device_error_k;

        u32_t const seed_blocks = static_cast<u32_t>(divide_round_up<std::size_t>(seed_count, block_threads_k));
        dense_cuda_mark_seeds_kernel<<<seed_blocks, block_threads_k, 0, stream>>>(
            visited_words, static_cast<u32_t>(nodes), static_cast<u32_t>(seed_count));

        // Sweep to convergence: bits only turn on and rows expand once, so the sweep
        // count is bounded by the graph's landmark eccentricity, capped defensively.
        constexpr std::size_t sweeps_cap_k = 256;
        u32_t const grid = static_cast<u32_t>(device_.sm_count) * search_blocks_per_sm_k;
        sweeps = 0;
        for (; sweeps != sweeps_cap_k; ++sweeps) {
            if (cudaMemsetAsync(changed, 0, sizeof(u32_t), stream) != cudaSuccess) return status_t::device_error_k;
            dense_cuda_reachability_sweep_kernel<<<grid, block_threads_k, 0, stream>>>(
                reinterpret_cast<u32_t const*>(neighbors_plane_.base()), static_cast<u32_t>(nodes),
                static_cast<u32_t>(connectivity()), visited_words, expanded_words, changed);
            u32_t changed_rows = 0;
            if (cudaMemcpyAsync(&changed_rows, changed, sizeof(u32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess)
                return status_t::device_error_k;
            if (!changed_rows) break;
        }
        ++sweeps;
        return status_t::success_k;
    }

  public:
    /**
     *  @brief Finds the nearest published neighbors of every query row, writing
     *      @c config.count ids per query into @p result_ids - and their distances into
     *      @p result_distances when it is non-empty - ascending by distance, with sentinel
     *      ids marking absent matches. Safe to overlap with insertions on other streams.
     */
    search_result_t search(vectors_view_t queries, nk::matrix_span<dense_node_id_t> result_ids,
                           nk::matrix_span<dense_distance_t> result_distances, dense_traversal_context_t& context,
                           search_config_t config = {}) noexcept {
        std::size_t const queries_count = queries.extent(0);
        if (!queries_count) return {status_t::success_k};
        if (!vectors_plane_.reserved_bytes()) return {status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context.telemetry_) return {status_t::config_mismatch_k};
        cudaStream_t const stream = context.stream();
        bool const wants_distances = result_distances.extent(0) != 0;
        if (queries.extent(1) != dimensions() || result_ids.extent(0) != queries_count ||
            result_ids.extent(1) < config.count || config.count > 512 ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {status_t::config_mismatch_k};

        // The register ladder serves expansions up to its widest pool; anything past it
        // walks the arena-backed pool, whose results cover the head - the nearest 64.
        u32_t const expansion = config.expansion_factor > config.count ? config.expansion_factor : config.count * 2;
        bool const wide = expansion > dense_cuda_sorted_pool<16>::capacity_k;
        if (wide && config.count > dense_cuda_sorted_pool<2>::capacity_k) return {status_t::config_mismatch_k};

        // The pool must cover the requested count, whatever the expansion asked for.
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const pool_slots = pool_slots_for(expansion, count_slots > 2 ? count_slots : 2);
        dense_cuda_arena_shape_t const arena_shape = arena_shape_for(
            wide ? expansion : pool_slots * 32,
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        std::size_t const arena_warps = static_cast<std::size_t>(device_.sm_count) * search_blocks_per_sm_k *
                                        dense_cuda_warps_per_block_k;
        std::size_t const results_count = queries_count * config.count;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        std::byte* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes());
        if (!arena) return {status_t::bad_alloc_k};
        // The batch transport rides the context's grow-only staging: queries first - they
        // carry the widest alignment - then the result ids and distances; the verb drains
        // the stream before returning, so one buffer serves every call on this lane.
        std::size_t const queries_bytes = queries_count * row_bytes();
        std::size_t const ids_bytes = results_count * sizeof(dense_node_id_t);
        std::byte* const staging = context.ensure_staging( //
            queries_bytes + ids_bytes + (wants_distances ? results_count * sizeof(dense_distance_t) : 0));
        if (!staging) return {status_t::bad_alloc_k};
        void* const device_queries = staging;
        dense_node_id_t* const device_ids = reinterpret_cast<dense_node_id_t*>(staging + queries_bytes);
        dense_distance_t* const device_distances =
            wants_distances ? reinterpret_cast<dense_distance_t*>(staging + queries_bytes + ids_bytes) : nullptr;
        // Epoch discipline: zero marks never-used slots, and this launch's epochs ride
        // above the context's monotone floor - zeroing happens only on fresh allocations
        // and 32-bit epoch wrap, never on a warm arena.
        if (context.arena_bytes_ != arena_bytes_before || context.epoch_floor_ + queries_count > 0xFFFFFFFEull) {
            cudaMemsetAsync(arena, 0, context.arena_bytes_, stream);
            context.epoch_floor_ = 0;
        }
        u32_t const epoch_base = static_cast<u32_t>(context.epoch_floor_);
        context.epoch_floor_ += queries_count;
        // Pitched upload: host rows are packed scalars, device rows pad to whole fragments,
        // and the pad bytes must be zero for the zero-tail distance identity to hold.
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (scalar_row_bytes != row_bytes()) cudaMemsetAsync(device_queries, 0, queries_count * row_bytes(), stream);
        cudaMemcpy2DAsync(device_queries, row_bytes(), queries.byte_data(), scalar_row_bytes, scalar_row_bytes,
                          queries_count, cudaMemcpyHostToDevice, stream);
        dense_cuda_telemetry_block_t const before = context.telemetry_snapshot();

        dense_cuda_search_view<kit_t> view {planes(),
                                            reinterpret_cast<fragment_t const*>(device_queries),
                                            device_ids,
                                            device_distances,
                                            static_cast<u32_t>(queries_count),
                                            config.count,
                                            static_cast<u32_t>(size()),
                                            static_cast<u32_t>(row_bytes()),
                                            expansion,
                                            arena,
                                            arena_shape,
                                            epoch_base,
                                            context.telemetry_.get()};
        if (wide) launch_search_wide(view, stream);
        else launch_search(view, pool_slots, stream);
        cudaMemcpyAsync(result_ids.data(), device_ids, results_count * sizeof(dense_node_id_t), cudaMemcpyDeviceToHost,
                        stream);
        if (wants_distances)
            cudaMemcpyAsync(result_distances.data(), device_distances, results_count * sizeof(dense_distance_t),
                            cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) return {status_t::device_error_k};

        dense_cuda_telemetry_block_t const after = context.telemetry_snapshot();
        search_result_t result;
        result.count = config.count;
        result.visited_members = after.hops - before.hops;
        result.computed_distances = after.scored - before.scored;
        result.restarts = after.restarts - before.restarts;
        return result;
    }

    /**
     *  @brief One query in, the nearest matches ascending in @c context.results - the CPU
     *      verb's exact shape, so the shared scenarios & phases drive the GPU unmodified.
     *      A one-query grid is the correctness path, not the throughput path: batch
     *      queries through the overload above to fill the machine.
     */
    search_result_t search(vector_view_t query_vector, dense_traversal_context_t& context,
                           search_config_t config = {}) noexcept {
        context.results.count_ = 0;
        if (!size()) return {status_t::success_k}; // An empty index matches nothing, touches nothing.
        // Matches reported follow the CPU semantic: up to the expansion factor, defaulting
        // to twice the connectivity, clipped to the results landing and the graph.
        u32_t const expansion = config.expansion_factor ? config.expansion_factor
                                                        : static_cast<u32_t>(connectivity()) * 2;
        u32_t const count = min_of<u32_t>(min_of<u32_t>(dense_traversal_context_t::results_capacity_k, expansion),
                                          static_cast<u32_t>(size()));
        dense_node_id_t staged_ids[dense_traversal_context_t::results_capacity_k];
        dense_distance_t staged_distances[dense_traversal_context_t::results_capacity_k];
        vectors_view_t const one_query {query_vector.data(), {1, dimensions()}};
        config.count = count;
        search_result_t result = search(one_query, nk::matrix_span<dense_node_id_t> {staged_ids, {1, count}},
                                        nk::matrix_span<dense_distance_t> {staged_distances, {1, count}}, context,
                                        config);
        if (!result) return result;
        std::size_t live = 0;
        for (; live != count && staged_ids[live] != dense_node_id_unknown_k; ++live)
            context.results.entries_[live] = dense_scored_node_t {staged_ids[live], staged_distances[live]};
        context.results.count_ = live;
        result.count = live;
        return result;
    }

    /** @brief The serialized extents: published nodes x logical dimensions & connectivity -
     *      the fragment padding is a storage detail that never reaches a file. */
    struct saved_shape_t {
        std::size_t nodes = 0;
        std::size_t dimensions = 0;
        std::size_t connectivity = 0;
    };
    saved_shape_t saved_shape() const noexcept { return {size(), dimensions(), connectivity()}; }

    /**
     *  @brief Fills the caller's typed matrices with the two serialized planes: vectors as
     *      logical @b [nodes,dimensions] rows - the fragment padding stripped by a strided
     *      copy - and neighbors verbatim, already speaking the positional sentinel
     *      convention (an id at or past the node count names no edge, a slot-0 self-loop
     *      is published-empty). Norms are derived state and never serialized.
     */
    status_t save(nk::matrix_span<scalar_t> vectors, nk::matrix_span<node_id_t> neighbors,
                  synchronization_t synchronization) const noexcept {
        if (synchronization == synchronization_t::concurrent_k) return status_t::config_mismatch_k;
        saved_shape_t const shape = saved_shape();
        if (vectors.extent(0) != shape.nodes || vectors.extent(1) != shape.dimensions ||
            neighbors.extent(0) != shape.nodes || neighbors.extent(1) != shape.connectivity)
            return status_t::config_mismatch_k;
        if (!shape.nodes) return status_t::success_k;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (cudaMemcpy2D(vectors.data(), scalar_row_bytes, vectors_plane_.base(), row_bytes(), scalar_row_bytes,
                         shape.nodes, cudaMemcpyDeviceToHost) != cudaSuccess)
            return status_t::device_error_k;
        if (cudaMemcpy(neighbors.data(), neighbors_plane_.base(), shape.nodes * shape.connectivity * sizeof(node_id_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /**
     *  @brief Rebuilds the index from serialized planes: demands an exclusive caller and a
     *      virgin, geometry-frozen index; re-pads the vectors into fragment rows - freshly
     *      committed plane memory arrives zeroed, so the pad tails hold the identity -
     *      re-derives the norms on the device, and publishes the counters last.
     */
    status_t load(nk::matrix_view<scalar_t> vectors, nk::matrix_view<node_id_t> neighbors,
                  synchronization_t synchronization) noexcept {
        if (synchronization != synchronization_t::exclusive_k) return status_t::config_mismatch_k;
        if (!vectors_plane_.reserved_bytes()) return status_t::uninitialized_k;
        if (size()) return status_t::config_mismatch_k;
        std::size_t const nodes = vectors.extent(0);
        if (vectors.extent(1) != dimensions() || neighbors.extent(0) != nodes ||
            neighbors.extent(1) != connectivity() || nodes > max_node_count_k)
            return status_t::config_mismatch_k;
        if (!nodes) return status_t::success_k;
        if (status_t grown = grow_rows(nodes, nullptr); grown != status_t::success_k) return grown;
        std::size_t const scalar_row_bytes = packed_row_bytes();
        if (cudaMemcpy2D(vectors_plane_.base(), row_bytes(), vectors.data(), scalar_row_bytes, scalar_row_bytes, nodes,
                         cudaMemcpyHostToDevice) != cudaSuccess)
            return status_t::device_error_k;
        if (cudaMemcpy(neighbors_plane_.base(), neighbors.data(), nodes * connectivity() * sizeof(node_id_t),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return status_t::device_error_k;
        u32_t const blocks = static_cast<u32_t>(device_.sm_count) * search_blocks_per_sm_k;
        dense_cuda_derive_norms_kernel<kit_t><<<blocks, block_threads_k>>>(planes(), static_cast<u32_t>(nodes));
        // Publish the control block last, once every plane byte is in place.
        dense_cuda_control_t const restored {static_cast<u32_t>(nodes)};
        if (cudaMemcpy(control_.get(), &restored, sizeof(restored), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
            return status_t::device_error_k;
        admission_.claimed_rows.store(nodes, std::memory_order_release);
        return status_t::success_k;
    }

  private:
    using vectors_plane_allocator_t = typename vectors_allocator_t::template rebind<std::byte>::other;
    using neighbors_plane_allocator_t = typename neighbors_allocator_t::template rebind<std::byte>::other;
    using control_allocator_t = typename traversals_allocator_t::template rebind<dense_cuda_control_t>::other;

    /**
     *
     *  @section dense_vram_index_host_state Host state & its concurrency regimes
     *
     *  Four regimes cover everything below, mirroring the CPU index member-for-member:
     *  @c device_ and @c geometry_ are @b frozen - the specs at first reserve, the geometry
     *  by the type's only compare-and-swap - and read lock-free everywhere after. The three
     *  @b planes append-only grow under the admission latch; each reservable allocator IS
     *  its plane, owning the base pointer and the committed extent, so ceilings and mapped
     *  row counts are derived observers, never cached. The @b control block is device
     *  memory no host thread dereferences. The @b admission struct is the only mutable host
     *  state: every field is written solely between @c latch acquire and clear.
     *
     *  Counter correspondence to the CPU: its @c last_allocated_id_ is our derived
     *  @c mapped_rows(), its @c next_visible_id_ our @c admission_.claimed_rows, and its
     *  @c last_visible_id_ our device-side @c published_count - except GPU publication is
     *  unordered, so @c size() reports the dense claim frontier, and landmark sampling
     *  probes past unpublished ids by protocol instead of trusting a watermark.
     *
     *  That frontier is a safe bound because a batch is atomic against it: the range is
     *  stored only once its launch is accepted, and a launch that ran short has its still
     *  sentinel rows tombstoned before the verb returns. Every id below the frontier is
     *  therefore published, and the one gap no sweep reaches - a device fault poisoning
     *  the context - is what @c validate names.
     */

    /** @brief Hardware identity: the ordinal comes from the vectors allocator at
     *      construction; the remaining specs fill at the first @c reserve(). */
    cuda_device_specs_t device_;
    /** @brief The frozen geometry word - dims high, connectivity low, zero = unreserved. */
    std::atomic<packed_geometry_id_t> geometry_ {0};
    /** @brief The planes: norms share the vectors allocator's placement. */
    vectors_plane_allocator_t vectors_plane_, norms_plane_;
    neighbors_plane_allocator_t neighbors_plane_;
    /** @brief The device-resident control block, allocated by the first @c reserve(). */
    cuda_unique_ptr<dense_cuda_control_t, control_allocator_t> control_;
    /** @brief Grow-then-claim state: @c latch serializes siblings, @c claimed_rows
     *      release-publishes each batch's dense id range - what @c size() reports. */
    struct admission_t {
        std::atomic_flag latch = ATOMIC_FLAG_INIT;
        std::atomic<std::size_t> claimed_rows = 0;
    } admission_;

    /** @brief Rows physically mapped across all three planes, derived from the neighbors
     *      plane - it commits last in @c grow_rows, so its extent is the safe minimum even
     *      after a partially failed grow. */
    std::size_t mapped_rows() const noexcept {
        std::size_t const stride = connectivity() * sizeof(u32_t);
        return stride ? neighbors_plane_.committed_bytes() / stride : 0;
    }

    /** @brief Maps every plane up to @p rows and sentinel-fills the fresh neighbor rows -
     *      always ahead of any launch that could dispense new_node_ids into them. Callers
     *      hold the admission latch, or run exclusively like @c reserve and @c load. */
    status_t grow_rows(std::size_t rows, cudaStream_t stream) noexcept {
        if (rows <= mapped_rows()) return status_t::success_k;
        std::size_t const neighbors_before = neighbors_plane_.committed_bytes();
        if (status_t status = vectors_plane_.commit(rows * row_bytes()); status != status_t::success_k) return status;
        if (status_t status = norms_plane_.commit(rows * sizeof(norms_t)); status != status_t::success_k) return status;
        if (status_t status = neighbors_plane_.commit(rows * connectivity() * sizeof(u32_t));
            status != status_t::success_k)
            return status;
        std::size_t const fresh_bytes = neighbors_plane_.committed_bytes() - neighbors_before;
        if (fresh_bytes && cuMemsetD32Async(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>( //
                                                neighbors_plane_.base() + neighbors_before)),
                                            dense_node_id_unknown_k, fresh_bytes / 4, stream) != CUDA_SUCCESS)
            return status_t::device_error_k;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return status_t::device_error_k;
        return status_t::success_k;
    }

    /** @brief Rounds the requested expansion to the closed pool ladder: 64/128/256/512. */
    static u32_t pool_slots_for(u32_t expansion_factor, u32_t default_slots) noexcept {
        if (!expansion_factor) return default_slots;
        u32_t const slots = std::bit_ceil(divide_round_up<u32_t>(expansion_factor, 32));
        return min_of(max_of(slots, 2u), 16u);
    }

    /** @brief Blocks per grid: three resident blocks per SM saturate the insertion path
     *      at its register budget; oversubscribing eight per SM keeps enough independent
     *      queries in flight to reach the memory-level-parallelism knee while searching. */
    inline static constexpr u32_t insert_blocks_per_sm_k = dense_cuda_resident_blocks_per_sm_k;
    inline static constexpr u32_t search_blocks_per_sm_k = 8;
    inline static constexpr u32_t block_threads_k = dense_cuda_warps_per_block_k * 32;

    // Every kernel family rides the two shared helpers: `dense_cuda_launch` owns the
    // shared-memory opt-in and the launch shape, `dense_cuda_with_pool_slots` lowers the
    // runtime pool width onto the closed register ladder.
    void launch_insert(dense_cuda_build_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
            dense_cuda_launch<dense_cuda_insert_kernel<kit_t, slots.value>>( //
                view, device_.sm_count * insert_blocks_per_sm_k, stream);
        });
    }
    void launch_refine(dense_cuda_refine_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
            dense_cuda_launch<dense_cuda_refine_kernel<kit_t, slots.value>>( //
                view, device_.sm_count * insert_blocks_per_sm_k, stream);
        });
    }
    void launch_join(dense_cuda_refine_view<kit_t> view, cudaStream_t stream) noexcept {
        dense_cuda_launch<dense_cuda_join_kernel<kit_t>>(view, device_.sm_count * insert_blocks_per_sm_k, stream);
    }
    void launch_merge(dense_cuda_refine_view<kit_t> view, u32_t const* bucket_ids, u32_t const* bucket_counts,
                      u32_t bucket_width, cudaStream_t stream) noexcept {
        dense_cuda_launch<dense_cuda_merge_links_kernel<kit_t>>( //
            view, device_.sm_count * insert_blocks_per_sm_k, stream, bucket_ids, bucket_counts, bucket_width);
    }
    void launch_search(dense_cuda_search_view<kit_t> view, u32_t pool_slots, cudaStream_t stream) noexcept {
        dense_cuda_with_pool_slots(pool_slots, [&](auto slots) {
            dense_cuda_launch<dense_cuda_search_kernel<kit_t, slots.value>>( //
                view, device_.sm_count * search_blocks_per_sm_k, stream);
        });
    }
    void launch_search_wide(dense_cuda_search_view<kit_t> view, cudaStream_t stream) noexcept {
        dense_cuda_launch<dense_cuda_search_wide_kernel<kit_t>>( //
            view, device_.sm_count * search_blocks_per_sm_k, stream);
    }
};

#endif // defined(__CUDACC__)

} // namespace usearch
} // namespace unum
