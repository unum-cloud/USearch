/**
 *  @file include/usearch/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The device bodies every dense VRAM index compiled by NVCC or HIP-Clang shares: the views,
 *      the greedy walk, the diversity selection, the back-linker, insertion and refinement.
 *
 *  Every body here is templated on a quads kit, which names its vendor's warp policy as
 *  @c kit_::warp_t; see @c vram_queue.cuh for what a warp policy provides. The host classes, the
 *  kernels that launch these bodies, the launch bounds, the attention fold and the reachability
 *  audits stay in each vendor's own @c vram_<vendor>_index file, beside the kits in
 *  @c vram_<vendor>_metric - the split @c utilities/vram_device.hpp keeps beside its vendor twins.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the views carry the metric

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/types.hpp"

namespace unum {
namespace usearch {

/** Ceiling on concurrently in-flight candidate-vector gathers per lane - the memory-parallelism window
 *  over the @b candidate axis, never the dimension axis. */
inline constexpr u32_t dense_vram_max_gathers_in_flight_k = 8;

/** Floor of the warp policy's sleep while retrying a held slot-0 lock. */
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

/** What a vendor's quads kit binds: one storage dtype on one GPU generation, with the warp policy
 *  its walks shuffle through. */
template <typename kit_>
concept dense_vram_quads_kit = requires(
    typename kit_::fragment_t fragment, typename kit_::dot_t dot, typename kit_::weighted_t weighted,
    typename kit_::norms_accumulator_t norms_accumulator, typename kit_::norms_t norms, dense_distance_t ranked,
    nk_kernel_kind_t kernel) {
    typename kit_::scalar_t;
    typename kit_::warp_t;
    { kit_::capability_k } -> std::convertible_to<nk_capability_t>;
    { kit_::accumulate_dot(fragment, fragment, dot) } -> std::same_as<typename kit_::dot_t>;
    { kit_::accumulate_weighted(fragment, f32_t {}, weighted) } -> std::same_as<typename kit_::weighted_t>;
    { kit_::accumulate_norms(fragment, norms_accumulator) } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::partials_to_norms(norms_accumulator) } -> std::same_as<typename kit_::norms_t>;
    { kit_::distance(kernel, dot, norms, norms) } -> std::same_as<dense_distance_t>;
    { kit_::convert_to_distance(kernel, ranked) } -> std::same_as<dense_distance_t>;
};

/** Generic cross-lane reduction: folds sub-warp partials of width @p lanes_per_row, a warp-uniform
 *  power of two, down to the leader lane. One tree serves every kit. */
template <typename warp_type_, typename dot_type_>
__device__ __forceinline__ dot_type_ dense_vram_reduce(dot_type_ partial, u32_t lanes_per_row) noexcept {
    for (u32_t offset = lanes_per_row / 2; offset; offset >>= 1)
        partial += warp_type_::shuffle_down(partial, offset, static_cast<int>(lanes_per_row));
    return partial;
}

/** Full-warp reduction of a two-component norms accumulator - {sum,sumsq} for the
 *  arithmetic kits, {popcount,popcount} for the binary ones. */
template <typename warp_type_, typename pair_type_>
__device__ __forceinline__ pair_type_ dense_vram_reduce_norms(pair_type_ partial) noexcept {
#pragma unroll
    for (u32_t offset = warp_type_::width_k / 2; offset; offset >>= 1) {
        partial.x += warp_type_::shuffle_down(partial.x, offset);
        partial.y += warp_type_::shuffle_down(partial.y, offset);
    }
    return partial;
}

/** Rebroadcasts lane 0's reduced norms to every lane - values that feed
 *  lane-divergent control flow must be warp-uniform. */
template <typename warp_type_, typename pair_type_>
__device__ __forceinline__ pair_type_ dense_vram_broadcast_norms(pair_type_ reduced) noexcept {
    reduced.x = warp_type_::shuffle(reduced.x, 0);
    reduced.y = warp_type_::shuffle(reduced.y, 0);
    return reduced;
}

/** Typed views over the three device planes, @b [rows,*] each: packed vectors, per-row norms, and
 *  fixed-capacity neighbors. Under the flat virtual-memory arena the node ID @b is the location -
 *  one multiply, no chunk directory. */
template <dense_vram_quads_kit kit_>
struct dense_vram_planes_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** Vector plane, @b [rows,row_fragments] fragments, one aligned row per node. */
    fragment_t* vectors;

    /** Norms plane, @b [rows] records, written once pre-publication; null under dot, which reads
     *  none. */
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

    /** The metric the kit's epilogue finalizes dots into - runtime, so one kernel serves every
     *  metric. */
    nk_kernel_kind_t kernel;

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

    /** A row's norms, or zeros from the empty plane - a warp-uniform branch on a launch
     *  argument. */
    __device__ __forceinline__ norms_t norm_for(dense_node_id_t node_id) const noexcept {
        return norms ? norms[node_id] : norms_t {};
    }
    __device__ __forceinline__ norms_t& norm_ref_for(dense_node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ u32_t* neighbors_for(dense_node_id_t node_id) const noexcept {
        return neighbors + static_cast<std::size_t>(node_id) * connectivity;
    }
};

/**
 *  @brief Typed view over one node's neighbors - the only legal way to touch slots that
 *      concurrent warps may be reading, over the warp policy's device-scope atomics.
 *
 *  Slot 0 is the combined lock and first-neighbor word: the all-ones sentinel while unpublished
 *  or locked, the node's own ID when published empty, otherwise the first neighbor. Acquisition
 *  is bounded - spinning against a descheduled owner warp would livelock - and an unlinked
 *  neighbor is the telemetered cost.
 */
template <typename warp_type_>
struct dense_vram_neighbors_ref {
    using warp_t = warp_type_;
    u32_t* row_; // Non-const even for readers: device atomic references cannot bind const words pre-P3323.
    u32_t capacity_;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }

    /** Warp-ballot mask of the lanes whose index is a valid neighbor slot - a collective read
     *  reports the sentinel on the lanes outside it. */
    __device__ __forceinline__ typename warp_t::mask_t lane_mask() const noexcept {
        return capacity_ >= warp_t::width_k ? warp_t::all_k : warp_t::below(static_cast<int>(capacity_));
    }

    /** Acquire read of slot @p index, pairing with the lock holder's release fence so an ID read
     *  here comes with its node's payload; slot 0 belongs to @c peek_first. */
    __device__ __forceinline__ dense_node_id_t load_slot(u32_t index) const noexcept {
        return warp_t::load_acquire(row_[index]);
    }

    /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current lock
     *  holder. */
    __device__ __forceinline__ void store_slot(u32_t index, dense_node_id_t value) const noexcept {
        warp_t::store_relaxed(row_[index], value);
    }

    /** Acquire-load of the lock word: the sentinel means unpublished or locked, any other value is
     *  the first neighbor ID with the last unlocker's writes visible. */
    __device__ __forceinline__ dense_node_id_t peek_first() const noexcept { return warp_t::load_acquire(row_[0]); }

    /** Bounded lock acquisition: exchanges the sentinel in with acquire order, up to
     *  @p max_attempts times with jittered backoff; an exchange that receives the sentinel
     *  back did @b not acquire. @return The displaced first-neighbor value, or the sentinel. */
    __device__ __forceinline__ dense_node_id_t lock_if_free(u32_t max_attempts,
                                                            splitmix64_t& random_generator) const noexcept {
        for (u32_t attempt = 0; attempt != max_attempts; ++attempt) {
            dense_node_id_t const first_value = warp_t::exchange_acquire(row_[0], dense_node_id_unknown_k);
            if (first_value != dense_node_id_unknown_k) return first_value;
            warp_t::sleep(
                static_cast<u32_t>(lock_backoff_floor_ns_k + (random_generator() & lock_backoff_jitter_mask_k)));
        }
        return dense_node_id_unknown_k;
    }

    /** Release-stores @p first_value into slot 0 - unlocking and simultaneously (re)publishing the
     *  edited neighbors to every future acquiring reader. */
    __device__ __forceinline__ void unlock(dense_node_id_t first_value) const noexcept {
        warp_t::store_release(row_[0], first_value);
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

/** Which traversals a telemetry flush folds: inserting ones also own the lock and reselection
 *  counters. */
enum class dense_vram_traversal_t : bool { searching_k, inserting_k };

/** Per-warp register telemetry, folded into the device counters once per node; 32-bit in registers,
 *  since every counter is bounded by hops times connectivity, widened at the flush. */
struct dense_vram_telemetry_t {
    u32_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, reselections = 0, restarts = 0, locked_drops = 0;
    u32_t local_scored = 0, remote_scored = 0;
    u32_t tail_drops = 0, visited_saturations = 0;
    u32_t changed_nodes = 0;

    /** Folds the live counters into the device block; the search instantiation skips the lock and
     *  reselection atomics it never touches. */
    template <typename warp_type_, dense_vram_traversal_t traversal_>
    __device__ __forceinline__ void flush(dense_vram_telemetry_block_t* counters, int lane) noexcept {
        if (lane != 0) return;
        // Tallies alone: the drain after the launch publishes them, so every add is relaxed.
        auto const add = [](u64_t& counter, u32_t value) noexcept {
            warp_type_::fetch_add_relaxed(counter, static_cast<u64_t>(value));
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
        if constexpr (traversal_ == dense_vram_traversal_t::searching_k) return;
        add(counters->lock_attempts, lock_attempts);
        add(counters->lock_skips, lock_skips);
        add(counters->reselections, reselections);
    }
};

/** One walking warp's state: the candidate pool, the random stream, the running telemetry, and the shared scratch
 *  past its query span, where a rescoring hands its fold what the fold keeps. */
template <typename frontier_>
struct dense_vram_warp_context {
    frontier_ frontier;
    splitmix64_t random_generator;
    dense_vram_telemetry_t telemetry;
    byte_t* scratch = nullptr;
};

/** One warp's staged batch: candidate IDs, then the scored nodes and native dots of up to a
 *  lane-width of them. */
template <typename kit_>
struct dense_vram_gathered {

    /** Candidate node IDs, up to a full vector cache of them. */
    u32_t ids[dense_vram_cache_rows_k];

    /** The scored nodes of one batch, by batch row. */
    dense_vram_scored_node_t scored[kit_::warp_t::width_k];

    /** The native dots of one batch, parallel to @c scored. */
    typename kit_::dot_t dots[kit_::warp_t::width_k];
};

/** Graph products of one query row as a device fold: each scored batch appends its node IDs and
 *  native dots to the row until it fills, while @c scored keeps counting past it. */
template <typename kit_>
struct dense_vram_multiply_fold {
    using dot_type_ = typename kit_::dot_t;

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

    __device__ __forceinline__ walk_demand_t absorb(dense_vram_gathered<kit_> const& gathered, u32_t gathered_count,
                                                    int lane) noexcept {
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
template <dense_vram_quads_kit kit_, u32_t resident_blocks_ = dense_vram_resident_blocks_per_multiprocessor_k>
__device__ __forceinline__ dense_vram_scored_node_t dense_vram_score_batch( //
    dense_vram_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, u32_t const* batch_ids, u32_t gathered_count,
    dense_vram_gathered<kit_>& gathered, int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using norms_t = typename kit_::norms_t;
    // 128 concurrent random gathers saturate the memory path, so the window is 128 over the
    // resident warps.
    constexpr u32_t gathers_in_flight_k = min_of(dense_vram_max_gathers_in_flight_k,
                                                 128u / (dense_vram_warps_per_block_k * resident_blocks_));
    u32_t const lanes_log2 = 31 - __clz(planes.lanes_per_row);
    u32_t const rows_per_gather = warp_t::width_k >> lanes_log2;
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
            typename kit_::dot_t const dot = dense_vram_reduce<warp_t>(accumulators[slot], planes.lanes_per_row);
            u32_t const row = (group_base + slot) * rows_per_gather + subwarp;
            if (leader && group_base + slot < planes.lanes_per_row && row < gathered_count) {
                dense_distance_t const distance = kit_::distance(planes.kernel, dot, query_state, batch_norms[slot]);
                gathered.scored[row] = dense_vram_scored_node_t::make(distance, batch_ids[row]);
                gathered.dots[row] = dot;
            }
        }
    }
    warp_t::sync();
    return lane < static_cast<int>(gathered_count) ? gathered.scored[lane] : dense_vram_scored_node_t::sentinel();
}

/** The plain walk's rescoring: none, so every batch lands as the kit scored it. */
struct dense_vram_no_rescoring {

    template <typename frontier_>
    __device__ __forceinline__ dense_vram_no_rescoring
    for_query(u32_t, dense_vram_warp_context<frontier_> const&) const noexcept {
        return {};
    }
};

/** Indexer heads as a device rescoring: each scored dot becomes the score Σₕ wₕ · ReLU(qₕ · k) and its distance
 *  1 − score, the warp scoring the batch once per head exactly as it scored the navigating query. */
template <dense_vram_quads_kit kit_>
struct dense_vram_indexer_rescoring {
    using warp_t = typename kit_::warp_t;
    using fragment_t = typename kit_::fragment_t;

    /** Every query's heads as staged rows, @b [queries·heads,row_fragments], device-resident. */
    fragment_t const* heads;

    /** The signed head weights, @b [queries,heads], device-resident. */
    f32_t const* weights;

    /** Heads per query. */
    u32_t heads_count;

    /** Fragments per staged head row. */
    u32_t row_fragments;

    template <typename frontier_>
    __device__ __forceinline__ dense_vram_indexer_rescoring
    for_query(u32_t query_index, dense_vram_warp_context<frontier_> const&) const noexcept {
        dense_vram_indexer_rescoring narrowed = *this;
        narrowed.heads += static_cast<std::size_t>(query_index) * heads_count * row_fragments;
        narrowed.weights += static_cast<std::size_t>(query_index) * heads_count;
        return narrowed;
    }

    __device__ __forceinline__ dense_vram_scored_node_t rescore(dense_vram_planes_view<kit_> const& planes,
                                                                dense_vram_gathered<kit_>& gathered,
                                                                u32_t gathered_count, int lane,
                                                                dense_vram_scored_node_t scored) const noexcept {
        bool const live = static_cast<u32_t>(lane) < gathered_count;
        f32_t indexed = 0;
        for (u32_t head = 0; head != heads_count; ++head) {
            dense_vram_score_batch<kit_>(planes, heads + head * row_fragments, typename kit_::norms_t {}, gathered.ids,
                                         gathered_count, gathered, lane);
            if (live) indexed += weights[head] * fmaxf(static_cast<f32_t>(gathered.dots[lane]), 0.f);
            // The next head's scoring overwrites the dots, so every lane reads its own first.
            warp_t::sync();
        }
        if (!live) return scored;
        gathered.dots[lane] = indexed;
        scored = dense_vram_scored_node_t::make(1 - indexed, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/** Scores one gathered batch against the query, lets @p rescoring rewrite the scores, folds it into the pool, and
 *  hands it to @p fold - whose answer says whether the walk goes on. */
template <dense_vram_quads_kit kit_, u32_t resident_blocks_ = dense_vram_resident_blocks_per_multiprocessor_k,
          typename frontier_, typename fold_, typename rescoring_ = dense_vram_no_rescoring>
__device__ __forceinline__ walk_demand_t dense_vram_score_into_frontier( //
    dense_vram_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, dense_vram_gathered<kit_>& gathered, u32_t gathered_count,
    dense_vram_warp_context<frontier_>& context, int lane, fold_&& fold,
    rescoring_ const& rescoring = rescoring_ {}) noexcept {
    using warp_t = typename kit_::warp_t;
    context.telemetry.scored += gathered_count;
    if (planes.resident_end) {
        bool const counted = static_cast<u32_t>(lane) < gathered_count;
        u32_t const candidate = counted ? gathered.ids[lane] : 0;
        typename warp_t::mask_t const local_mask = warp_t::ballot(counted && candidate >= planes.resident_begin &&
                                                                  candidate < planes.resident_end);
        context.telemetry.local_scored += warp_t::count(local_mask);
        context.telemetry.remote_scored += gathered_count - warp_t::count(local_mask);
    }
    dense_vram_scored_node_t staged = dense_vram_score_batch<kit_, resident_blocks_>( //
        planes, query_span, query_state, gathered.ids, gathered_count, gathered, lane);
    // Even a call that changes nothing reorders the register-capped wide walk, so the plain walk makes none.
    if constexpr (!std::is_same_v<rescoring_, dense_vram_no_rescoring>)
        staged = rescoring.rescore(planes, gathered, gathered_count, lane, staged);
    // No dedup: the exact visited set upstream guarantees every ID stages once.
    context.frontier.push_batch(staged, lane);
    return fold.absorb(gathered, gathered_count, lane);
}

/**
 *  @brief Samples one lane-wide batch of distinct, published landmarks below @p sampling_bound
 *      into @p candidate_ids - uniform draws gated on the slot-0 publication word, probing
 *      forward on unpublished hits so out-of-order publication only costs retries.
 *  @return The number of staged landmarks; zero when the visited set saturated the bound.
 */
template <dense_vram_quads_kit kit_>
__device__ __forceinline__ u32_t dense_vram_sample_landmarks( //
    dense_vram_planes_view<kit_> const& planes, dense_node_id_t sampling_bound,
    dense_vram_visited_set<typename kit_::warp_t>& visited, splitmix64_t& random_generator,
    u32_t (&candidate_ids)[dense_vram_cache_rows_k], int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    dense_node_id_t draw = uniform_below<dense_node_id_t> {sampling_bound}(random_generator);
    bool published = false;
    for (int attempt = 0; attempt != 4 && !published; ++attempt) {
        dense_vram_neighbors_ref<warp_t> const row {planes.neighbors_for(draw), planes.connectivity};
        published = row.peek_first() != dense_node_id_unknown_k;
        if (!published) draw = draw + 1 >= sampling_bound ? 0 : draw + 1; // >=: a zero bound stays at row 0.
    }
    bool const fresh = visited.test_and_set(draw, lane, published);
    typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
    u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
    if (fresh) candidate_ids[ordinal] = draw;
    warp_t::sync();
    return warp_t::count(fresh_mask);
}

/** Continues a greedy best-first walk toward the query over whatever the frontier holds - one warp,
 *  one traversal; pre-staged entries compete with the landmark seeds drawn here and must already be
 *  visited. Read-only on the graph: no locks taken, transiently locked candidates dropped after a
 *  bounded retry, and no neighbor at or past @p first_invisible_id followed, since a batch's back-links
 *  splice its IDs into visible rows before the host turns its watermark. */
template <dense_vram_quads_kit kit_, u32_t resident_blocks_ = dense_vram_resident_blocks_per_multiprocessor_k,
          typename frontier_, typename fold_ = dummy_fold_t, typename rescoring_ = dense_vram_no_rescoring>
__device__ __forceinline__ void dense_vram_search_ahead( //
    dense_vram_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, dense_node_id_t sampling_bound, dense_node_id_t first_invisible_id,
    dense_vram_visited_set<typename kit_::warp_t>& visited, dense_vram_gathered<kit_>& gathered,
    dense_vram_warp_context<frontier_>& context, int lane, fold_&& fold = fold_ {},
    rescoring_ const& rescoring = rescoring_ {}) noexcept {

    using warp_t = typename kit_::warp_t;
    constexpr u32_t restarts_cap_k = 8;
    frontier_& frontier = context.frontier;
    dense_vram_telemetry_t& telemetry = context.telemetry;
    u32_t const hops_cap = frontier.hops_budget();

    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = dense_vram_sample_landmarks<kit_>( //
                planes, sampling_bound, visited, context.random_generator, gathered.ids, lane);
            if (gathered_count && dense_vram_score_into_frontier<kit_, resident_blocks_>( //
                                      planes, query_span, query_state, gathered, gathered_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                return;
            seeding = false;
        }
        else {
            dense_vram_scored_node_t const candidate = frontier.pop_nearest(lane);
            if (candidate == dense_vram_scored_node_t::sentinel()) {
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
            dense_vram_neighbors_ref<warp_t> const row {planes.neighbors_for(candidate.id()), planes.connectivity};
            dense_node_id_t first_value = dense_node_id_unknown_k;
            if (lane == 0) {
                first_value = row.peek_first();
                for (u32_t attempt = 0; attempt != traversal_peek_retries_k && first_value == dense_node_id_unknown_k;
                     ++attempt) {
                    warp_t::sleep(512);
                    first_value = row.peek_first();
                }
            }
            first_value = warp_t::shuffle(first_value, 0);
            // The shuffle moves the word, not the leader's acquire; the lane barrier orders the
            // other lanes' slot loads after it.
            warp_t::sync();
            if (first_value == dense_node_id_unknown_k) {
                ++telemetry.locked_drops;
                continue;
            }
            // The first warp-width of slots is loop-free, which keeps the hot path compact.
            dense_node_id_t const neighbor_id = lane == 0                                   ? first_value
                                                : static_cast<u32_t>(lane) < row.capacity() ? row.load_slot(lane)
                                                                                            : dense_node_id_unknown_k;
            // The bound never exceeds the ID space, so it rejects the sentinel too.
            bool const usable = neighbor_id < first_invisible_id && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(neighbor_id, lane, usable);
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
            u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
            if (fresh) gathered.ids[ordinal] = neighbor_id;
            warp_t::sync();
            u32_t const gathered_count = warp_t::count(fresh_mask);
            if (gathered_count && dense_vram_score_into_frontier<kit_, resident_blocks_>( //
                                      planes, query_span, query_state, gathered, gathered_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                return;
            // Wider connectivities continue a warp-width at a time; the bound is warp-uniform for the collectives.
            for (u32_t slot_base = warp_t::width_k; slot_base < planes.connectivity; slot_base += warp_t::width_k) {
                u32_t const slot = slot_base + lane;
                dense_node_id_t const wide_id = slot < planes.connectivity ? row.load_slot(slot)
                                                                           : dense_node_id_unknown_k;
                bool const wide_usable = wide_id < first_invisible_id && wide_id != candidate.id();
                bool const wide_fresh = visited.test_and_set(wide_id, lane, wide_usable);
                typename warp_t::mask_t const wide_mask = warp_t::ballot(wide_fresh);
                u32_t const wide_ordinal = warp_t::count(wide_mask & warp_t::below(lane));
                if (wide_fresh) gathered.ids[wide_ordinal] = wide_id;
                warp_t::sync();
                u32_t const wide_count = warp_t::count(wide_mask);
                if (wide_count && dense_vram_score_into_frontier<kit_, resident_blocks_>( //
                                      planes, query_span, query_state, gathered, wide_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                    return;
            }
        }
    }
}

/** Greedy best-first walk from random published landmarks toward the query: a fresh frontier
 *  handed to @ref dense_vram_search_ahead. */
template <dense_vram_quads_kit kit_, u32_t resident_blocks_ = dense_vram_resident_blocks_per_multiprocessor_k,
          typename frontier_, typename fold_ = dummy_fold_t, typename rescoring_ = dense_vram_no_rescoring>
__device__ __forceinline__ void dense_vram_search_around( //
    dense_vram_planes_view<kit_> const& planes, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, dense_node_id_t sampling_bound, dense_node_id_t first_invisible_id,
    dense_vram_visited_set<typename kit_::warp_t>& visited, dense_vram_gathered<kit_>& gathered,
    dense_vram_warp_context<frontier_>& context, int lane, fold_&& fold = fold_ {},
    rescoring_ const& rescoring = rescoring_ {}) noexcept {
    context.frontier.reset(lane);
    dense_vram_search_ahead<kit_, resident_blocks_>( //
        planes, query_span, query_state, sampling_bound, first_invisible_id, visited, gathered, context, lane, fold,
        rescoring);
}

/** Runtime-strided view over the warp's shared vector cache - rows of @c row_fragments_ fragments
 *  each, aliasing the scratch slice. */
template <typename fragment_type_>
struct dense_vram_vector_cache {
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
 *  pre-sorted from the lane-wide bitonic network and merge in place, entries past @c capacity_
 *  drop. */
template <typename warp_type_>
struct dense_vram_sorted_buffer {
    using warp_t = warp_type_;
    dense_vram_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_;

    __device__ __forceinline__ dense_vram_scored_node_t at(u32_t rank) const noexcept { return entries_[rank]; }

    __device__ __forceinline__ void insert_batch(dense_vram_scored_node_t value, u32_t count, int lane) noexcept {
        constexpr u32_t span_batches_k = divide_round_up(dense_vram_cache_rows_k, warp_t::width_k);
        bool const incoming = static_cast<u32_t>(lane) < count;
        u32_t incoming_rank = incoming ? static_cast<u32_t>(lane) : 0;
        for (u32_t i = 0; i != size_; ++i) // Warp-uniform bound: every lane walks the span.
            incoming_rank += incoming && entries_[i].distance_bits() <= value.distance_bits();
        // Existing entries shift by the number of incoming values strictly ahead of them;
        // ties resolve the other way than above, so no two final positions collide.
        dense_vram_scored_node_t moved[span_batches_k];
        u32_t moved_to[span_batches_k];
        u32_t batch = 0;
        for (u32_t base = 0; base < size_; base += warp_t::width_k, ++batch) {
            u32_t const index = base + lane;
            bool const active = index < size_;
            moved[batch] = active ? entries_[index] : dense_vram_scored_node_t::sentinel();
            u32_t shift = 0;
            for (u32_t source = 0; source != count; ++source) // Warp-uniform: shuffles inside.
                shift += active && dense_shuffle_from<warp_t>(value, static_cast<int>(source)).distance_bits() <
                                       moved[batch].distance_bits();
            moved_to[batch] = index + shift;
        }
        warp_t::sync(); // Every displaced entry is staged in registers before any write lands.
        batch = 0;
        for (u32_t base = 0; base < size_; base += warp_t::width_k, ++batch)
            if (base + lane < size_ && moved_to[batch] < capacity_) entries_[moved_to[batch]] = moved[batch];
        if (incoming && incoming_rank < capacity_) entries_[incoming_rank] = value;
        warp_t::sync();
        size_ = min_of(size_ + count, capacity_);
    }
};

/** Shared-memory spans backing one warp's neighbor re-selection, carved after the vector cache:
 *  the selected target IDs, the ascending per-candidate scores, and the kept cache positions. */
struct dense_vram_selection_scratch_t {
    u32_t* selected_ids;
    dense_vram_scored_node_t* scored_entries;
    u32_t* kept_positions;

    /** Carves the three spans past the vector cache's @p cache_bytes: the ascending candidate
     *  scores first, since they carry the widest alignment, then the two ID spans. */
    static __device__ __forceinline__ dense_vram_selection_scratch_t carve(byte_t* scratch, u32_t connectivity,
                                                                           std::size_t cache_bytes) noexcept {
        u32_t const slots = min_of(connectivity, static_cast<u32_t>(dense_vram_cache_rows_k));
        byte_t* const base = scratch + cache_bytes;
        return {reinterpret_cast<u32_t*>(base + sizeof(dense_vram_scored_node_t) * slots),
                reinterpret_cast<dense_vram_scored_node_t*>(base),
                reinterpret_cast<u32_t*>(base + (sizeof(dense_vram_scored_node_t) + sizeof(u32_t)) * slots)};
    }
};

/**
 *  @brief Greedy diversity selection over the ascending pool prefix: a candidate is kept only
 *      when it is closer to the query than to every kept neighbor, so neighbors spread across
 *      directions; the kept neighbors are ascending by construction.
 *  @return The kept count; @p selected_ids holds that many node IDs on return.
 */
template <dense_vram_quads_kit kit_, typename frontier_>
__device__ __forceinline__ u32_t dense_vram_select_diverse( //
    dense_vram_planes_view<kit_> const& planes, dense_vram_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[dense_vram_cache_rows_k], u32_t (&candidate_ids)[dense_vram_cache_rows_k],
    u32_t* selected_ids, dense_vram_warp_context<frontier_>& context, int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    frontier_& frontier = context.frontier;

    // Stage the ascending prefix IDs & cache their vectors and norms.
    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < dense_vram_cache_rows_k; base += warp_t::width_k) {
        dense_vram_scored_node_t const entry = base + lane < dense_vram_cache_rows_k
                                                   ? frontier.rank_entry(base + lane)
                                                   : dense_vram_scored_node_t::sentinel();
        typename warp_t::mask_t const live = warp_t::ballot(!(entry == dense_vram_scored_node_t::sentinel()));
        if (!(entry == dense_vram_scored_node_t::sentinel())) candidate_ids[base + lane] = entry.id();
        candidates_count += warp_t::count(live);
        if (live != warp_t::all_k) break;
    }
    warp_t::sync();
    for (u32_t word = lane; word < candidates_count * planes.row_fragments; word += warp_t::width_k) {
        u32_t const row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, row);
        warp_t::copy_async(&cache.at(row, fragment), planes.vector_span_for(candidate_ids[row]) + fragment);
    }
    warp_t::copy_wait();
    for (u32_t row = lane; row < candidates_count; row += warp_t::width_k)
        cache_norms[row] = planes.norm_for(candidate_ids[row]);
    warp_t::sync();

    u32_t selected_count = 0;
    for (u32_t candidate_index = 0; candidate_index < candidates_count && selected_count < planes.connectivity;
         ++candidate_index) {
        f32_t const to_query = frontier.rank_entry(candidate_index).distance();
        bool diverse = true;
        for (u32_t base = 0; base < selected_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < selected_count) {
                u32_t const kept_position = selected_ids[base + lane];
                typename kit_::dot_t accumulator {};
                for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                    accumulator = kit_::accumulate_dot(cache.at(candidate_index, fragment),
                                                       cache.at(kept_position, fragment), accumulator);
                f32_t const between = kit_::distance(planes.kernel, accumulator, cache_norms[candidate_index],
                                                     cache_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++context.telemetry.scored;
            }
            diverse = warp_t::all(lane_diverse);
        }
        if (diverse) {
            if (lane == 0) selected_ids[selected_count] = candidate_index;
            ++selected_count;
            warp_t::sync();
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
        warp_t::sync();
    }
    // Rewrite the kept positions as node IDs in place - the caller links to nodes, not cache rows.
    warp_t::sync();
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        selected_ids[slot] = candidate_ids[selected_ids[slot]];
    warp_t::sync();
    return selected_count;
}

/** Back-links the node: locks each selected neighbor's slots through the bounded slot-0 exchange,
 *  appends when there is room, and otherwise re-runs the diversity heuristic over every neighbor
 *  plus the newcomer under the lock. At most one lock is held at a time, only after this node's own
 *  publication, and a lock busy past the budget skips that neighbor. */
template <dense_vram_quads_kit kit_>
__device__ __forceinline__ void dense_vram_back_link_neighbors( //
    dense_vram_planes_view<kit_> const& planes, dense_node_id_t new_node_id, typename kit_::norms_t new_norms,
    dense_vram_selection_scratch_t selection, u32_t selected_count,
    dense_vram_vector_cache<typename kit_::fragment_t> cache,
    typename kit_::norms_t (&cache_norms)[dense_vram_cache_rows_k], dense_vram_gathered<kit_>& gathered,
    splitmix64_t& random_generator, dense_vram_telemetry_t& telemetry, int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    // The last two cache rows host the newcomer and the target, whose vector the scored batch reads as its query.
    constexpr u32_t newcomer_row_k = dense_vram_cache_rows_k - 1;
    constexpr u32_t target_row_k = dense_vram_cache_rows_k - 2;
    // Those two rows also bound how many neighbors one warp can re-select; wider connectivities keep sentinel tails.
    u32_t const connectivity = min_of(planes.connectivity, dense_vram_cache_rows_k - 2u);

    for (u32_t selected_index = 0; selected_index < selected_count; ++selected_index) {
        dense_node_id_t const target = selection.selected_ids[selected_index];
        dense_vram_neighbors_ref<warp_t> const row {planes.neighbors_for(target), connectivity};
        dense_node_id_t first_value = dense_node_id_unknown_k;
        if (lane == 0) first_value = row.lock_if_free(back_link_lock_attempts_k, random_generator);
        first_value = warp_t::shuffle(first_value, 0);
        warp_t::sync(); // The sweep below must follow the leader's acquiring exchange on every lane.
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
        for (u32_t base = 0; base < connectivity; base += warp_t::width_k) {
            u32_t const slot = base + lane;
            dense_node_id_t const slot_value = slot == 0             ? first_value
                                               : slot < connectivity ? row.load_slot(slot)
                                                                     : dense_node_id_unknown_k;
            if (slot < connectivity) gathered.ids[slot] = slot_value;
            typename warp_t::mask_t const empty_mask = warp_t::ballot(slot < connectivity &&
                                                                      slot_value == dense_node_id_unknown_k);
            if (empty_mask && empty_slot == connectivity) empty_slot = base + warp_t::first(empty_mask);
        }
        warp_t::sync();
        // Every slot store below is relaxed; this fence heads each lane's stores, so an acquiring
        // reader of any slot sees the newcomer's payload.
        warp_t::fence_release();
        if (empty_slot != connectivity) { // Fast path: append into that slot, unlock unchanged.
            if (lane == 0) row.store_slot(empty_slot, new_node_id);
            warp_t::sync();
            if (lane == 0) row.unlock(first_value);
            continue;
        }

        // Every slot taken: re-select diversely over the neighbors plus the newcomer, ascending by
        // distance to the target itself.
        ++telemetry.reselections;
        for (u32_t word = lane; word < planes.row_fragments; word += warp_t::width_k)
            cache.at(target_row_k, word) = planes.vector_for(target, word);
        warp_t::sync();
        typename kit_::norms_t const target_state = planes.norm_for(target);

        // The newcomer's distance to the target, from its freshly-owned vector row.
        typename kit_::dot_t accumulator {};
        for (u32_t word = lane; word < planes.row_fragments; word += warp_t::width_k)
            accumulator = kit_::accumulate_dot(planes.vector_for(new_node_id, word), cache.at(target_row_k, word),
                                               accumulator);
        typename kit_::dot_t const self_dot = dense_vram_reduce<warp_t>(accumulator, warp_t::width_k);
        f32_t const newcomer_distance = kit_::distance(planes.kernel, warp_t::shuffle(self_dot, 0), target_state,
                                                       new_norms);
        telemetry.scored += connectivity + 1;

        // Score the neighbors a warp-width at a time, pre-sort each batch bitonically, and merge
        // them into the ascending sorted buffer.
        dense_vram_sorted_buffer<warp_t> sorted_candidates {selection.scored_entries, 0, connectivity};
        for (u32_t base = 0; base < connectivity; base += warp_t::width_k) {
            u32_t const batch_count = min_of(connectivity - base, warp_t::width_k);
            dense_vram_scored_node_t const scored = dense_vram_score_batch<kit_>( //
                planes, cache.row(target_row_k), target_state, gathered.ids + base, batch_count, gathered, lane);
            dense_vram_scored_node_t const sorted_batch = dense_sort_lanes<warp_t>(
                static_cast<u32_t>(lane) < batch_count
                    ? dense_vram_scored_node_t::make(scored.distance(), base + lane) // ID = slot index.
                    : dense_vram_scored_node_t::sentinel(),
                lane);
            sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
        }

        // Cache the neighbors' vectors plus the newcomer at the tail row through staging copies,
        // which hold no registers where the device has them, then merge-walk them ascending by
        // distance to the target.
        u32_t const link_words = connectivity * planes.row_fragments;
        for (u32_t word = lane; word < link_words; word += warp_t::width_k) {
            u32_t const cache_row = planes.row_of_word(word), fragment = planes.fragment_of_word(word, cache_row);
            warp_t::copy_async(&cache.at(cache_row, fragment),
                               planes.vector_span_for(gathered.ids[cache_row]) + fragment);
        }
        warp_t::copy_wait();
        for (u32_t word = lane; word < planes.row_fragments; word += warp_t::width_k)
            cache.at(newcomer_row_k, word) = planes.vector_for(new_node_id, word);
        for (u32_t cache_row = lane; cache_row < connectivity; cache_row += warp_t::width_k)
            cache_norms[cache_row] = planes.norm_for(gathered.ids[cache_row]);
        if (lane == 0) cache_norms[newcomer_row_k] = new_norms;
        warp_t::sync();

        // The merge walk: the sorted neighbors and the newcomer, ascending by distance to the target, as cache rows.
        constexpr u32_t walked_out_k = ~0u;
        u32_t existing_index = 0;
        bool newcomer_pending = true;
        f32_t candidate_distance = 0;
        auto next_candidate = [&]() noexcept {
            dense_vram_scored_node_t const next_link = existing_index < sorted_candidates.size_
                                                           ? sorted_candidates.at(existing_index)
                                                           : dense_vram_scored_node_t::sentinel();
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
            for (u32_t base = 0; base < refined_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
                bool lane_diverse = true;
                if (base + lane < refined_count) {
                    u32_t const kept_position = selection.kept_positions[base + lane];
                    typename kit_::dot_t pair_accumulator {};
                    for (u32_t fragment = 0; fragment != planes.row_fragments; ++fragment)
                        pair_accumulator = kit_::accumulate_dot(cache.at(candidate_position, fragment),
                                                                cache.at(kept_position, fragment), pair_accumulator);
                    f32_t const between = kit_::distance(planes.kernel, pair_accumulator,
                                                         cache_norms[candidate_position], cache_norms[kept_position]);
                    lane_diverse = between >= candidate_distance;
                    ++telemetry.scored;
                }
                diverse = warp_t::all(lane_diverse);
            }
            if (diverse) {
                if (lane == 0) selection.kept_positions[refined_count] = candidate_position;
                ++refined_count;
                warp_t::sync();
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
            warp_t::sync();
        }

        // Rewrite the neighbors: tail slots relaxed, shrink sentinel-terminated, head unlocks.
        warp_t::sync();
        for (u32_t slot = lane; slot < connectivity; slot += warp_t::width_k) {
            if (slot == 0) continue;
            dense_node_id_t linked = dense_node_id_unknown_k;
            if (slot < refined_count) {
                u32_t const kept_position = selection.kept_positions[slot];
                linked = kept_position == newcomer_row_k ? new_node_id : gathered.ids[kept_position];
            }
            row.store_slot(slot, linked);
        }
        warp_t::sync();
        if (lane == 0) {
            u32_t const head_position = selection.kept_positions[0];
            row.unlock(head_position == newcomer_row_k ? new_node_id : gathered.ids[head_position]);
        }
    }
}

/** Layout of one warp's slice of the traversal arena, computed on the host and carried by the
 *  views: the exact visited set, then the wide frontier's spill tail when it has one. */
struct dense_vram_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    constexpr std::size_t slice_bytes() const noexcept {
        return static_cast<std::size_t>(visited_slots) * sizeof(u64_t) +
               static_cast<std::size_t>(tail_entries) * sizeof(dense_vram_scored_node_t);
    }
    __device__ __forceinline__ u64_t* visited_span(byte_t* base, std::size_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(base + warp_index * slice_bytes());
    }
    __device__ __forceinline__ dense_vram_scored_node_t* tail_span(byte_t* base,
                                                                   std::size_t warp_index) const noexcept {
        return reinterpret_cast<dense_vram_scored_node_t*>( //
            base + warp_index * slice_bytes() + static_cast<std::size_t>(visited_slots) * sizeof(u64_t));
    }
};

/** One warp's slice of a launch: its coordinates, its dynamic-shared scratch, and its arena
 *  visited span - the shared prologue of every traversal kernel. */
struct dense_vram_warp_slice_t {
    int lane, warp;
    std::size_t warp_index;
    byte_t* scratch;
    u64_t* visited_slots;
};

/** Carves the calling warp's slice out of @p view and the block's @p dynamic_scratch. */
template <typename warp_type_, typename view_>
__device__ __forceinline__ dense_vram_warp_slice_t dense_vram_carve_slice(view_ const& view,
                                                                          byte_t* dynamic_scratch) noexcept {
    int const warp = threadIdx.x / warp_type_::width_k, lane = threadIdx.x % warp_type_::width_k;
    std::size_t const warp_index = blockIdx.x * static_cast<std::size_t>(dense_vram_warps_per_block_k) + warp;
    byte_t* const scratch = dynamic_scratch + static_cast<std::size_t>(warp) * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch, view.arena_shape.visited_span(view.arena, warp_index)};
}

/** Geometry & plane pointers one insertion launch operates on. Every plane row an assignable
 *  new_node_id can touch is mapped and sentinel-filled ahead of the launch. */
template <dense_vram_quads_kit kit_>
struct dense_vram_build_view {
    dense_vram_planes_view<kit_> planes;

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
    dense_vram_arena_shape_t arena_shape;

    /** The context's epoch floor at launch: batch-local epochs ride above it, so stale arena slots
     *  compare older and a warm arena is never re-zeroed. */
    u32_t epoch_base;

    /** The index's published-row count, advanced at each slot-0 release and polled by the admission
     *  ramp; it implies nothing about @b which rows. */
    u32_t* published_rows;
    dense_vram_telemetry_block_t* telemetry;

    /** Rows of the batch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;
};

/** Inputs & outputs of one search launch: a batch of queries in, the nearest IDs and - when
 *  requested - their distances out, both ascending per query. */
template <dense_vram_quads_kit kit_>
struct dense_vram_search_view {
    dense_vram_planes_view<kit_> planes;

    /** The query rows, @b [queries,row_fragments], device-resident. */
    typename kit_::fragment_t const* queries;

    u32_t queries_count;

    /** The watermark snapshotted at launch: landmarks are drawn below it and no neighbor at or past it is
     *  followed, while the walk still gates on slot-0 words, which concurrent back-linkers lock. */
    u32_t sampling_bound;

    /** Bytes of the per-warp dynamic-shared slice - the query span. */
    u32_t scratch_bytes_per_warp;

    /** Requested expansion; past the register ladder it is the wide walk's budget and sizes the
     *  arena's spill tail. */
    u32_t expansion;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice: the exact visited set, then the wide frontier's spill tail. */
    dense_vram_arena_shape_t arena_shape;

    /** The context's epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    dense_vram_telemetry_block_t* telemetry;
};

/** Inserts the node identified by @p new_node_id: lands the payload and norms, walks the published
 *  graph, selects diverse neighbors, publishes with one release store, then back-links them - one
 *  warp per node. */
template <dense_vram_quads_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_vram_insert_one( //
    dense_vram_build_view<kit_> const& view, dense_node_id_t new_node_id,
    dense_vram_visited_set<typename kit_::warp_t>& visited, typename kit_::fragment_t* query_span,
    dense_vram_vector_cache<typename kit_::fragment_t> cache, dense_vram_selection_scratch_t selection,
    typename kit_::norms_t (&cache_norms)[dense_vram_cache_rows_k], dense_vram_gathered<kit_>& gathered,
    dense_vram_warp_context<frontier_>& context, int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    dense_vram_planes_view<kit_> const& planes = view.planes;

    // One pass lands the payload, derives the norms, and parks the query in its shared span.
    typename kit_::fragment_t const* staging_row =
        view.staging + static_cast<std::size_t>(new_node_id - view.first_id) * planes.row_fragments;
    typename kit_::fragment_t* owned_row = planes.vector_span_for(new_node_id);
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += warp_t::width_k) {
        typename kit_::fragment_t const value = staging_row[fragment];
        owned_row[fragment] = value;
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = dense_vram_reduce_norms<warp_t>(norms_accumulator);
    norms_accumulator = dense_vram_broadcast_norms<warp_t>(norms_accumulator);
    typename kit_::norms_t const new_norms = kit_::partials_to_norms(norms_accumulator);
    if (lane == 0 && planes.norms) planes.norm_ref_for(new_node_id) = new_norms;
    typename kit_::norms_t const query_state = new_norms;
    warp_t::sync();

    // Walk the watermark - a virgin index publishes an empty node instead.
    if (new_node_id != 0)
        dense_vram_search_around<kit_>( //
            planes, query_span, query_state, new_node_id, view.end_id, visited, gathered, context, lane);
    else context.frontier.reset(lane);

    // The own row needs no lock: no reader arrives before the slot-0 release below flips the sentinel.
    u32_t const selected_count = dense_vram_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered.ids, selection.selected_ids, context, lane);
    dense_vram_neighbors_ref<warp_t> const own_neighbors {planes.neighbors_for(new_node_id), planes.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    warp_t::sync();
    dense_node_id_t const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) {
        own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
        // A tally for the admission ramp; the slot-0 release above is what publishes the row.
        warp_t::fetch_add_relaxed(*view.published_rows, 1u);
    }

    // Only now - published, no locks held - back-link the neighbors.
    dense_vram_back_link_neighbors<kit_>( //
        planes, new_node_id, new_norms, selection, selected_count, cache, cache_norms, gathered,
        context.random_generator, context.telemetry, lane);
}

/** Geometry & plane pointers one refinement launch operates on: the live planes it reads, and the
 *  shadow neighbor plane it writes - same @b [rows,connectivity] shape, observed by no reader. */
template <dense_vram_quads_kit kit_>
struct dense_vram_refine_view {
    dense_vram_planes_view<kit_> planes;

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
    dense_vram_arena_shape_t arena_shape;

    /** The context's monotone epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    dense_vram_telemetry_block_t* telemetry;

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
template <dense_vram_quads_kit kit_, typename frontier_>
__device__ __forceinline__ void dense_vram_refine_one( //
    dense_vram_refine_view<kit_> const& view, dense_node_id_t node_id,
    dense_vram_visited_set<typename kit_::warp_t>& visited, typename kit_::fragment_t* query_span,
    dense_vram_vector_cache<typename kit_::fragment_t> cache, dense_vram_selection_scratch_t selection,
    typename kit_::norms_t (&cache_norms)[dense_vram_cache_rows_k], dense_vram_gathered<kit_>& gathered,
    dense_vram_warp_context<frontier_>& context, int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    dense_vram_planes_view<kit_> const& planes = view.planes;

    // The query is the node's own stored vector; its norms are already derived state.
    for (u32_t fragment = lane; fragment < planes.row_fragments; fragment += warp_t::width_k)
        query_span[fragment] = planes.vector_for(node_id, fragment);
    typename kit_::norms_t const query_state = planes.norm_for(node_id);
    warp_t::sync();

    // The node itself never joins its own candidate pool - visited before anything stages.
    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane, true);

    // Seed the frontier with the current neighbors, staged like a traversal expansion minus the hop bookkeeping.
    dense_vram_neighbors_ref<warp_t> const own_neighbors {planes.neighbors_for(node_id), planes.connectivity};
    dense_node_id_t first_value = dense_node_id_unknown_k;
    if (lane == 0) first_value = own_neighbors.peek_first();
    first_value = warp_t::shuffle(first_value, 0);
    if (first_value != dense_node_id_unknown_k && first_value != node_id)
        for (u32_t slot_base = 0; slot_base < planes.connectivity; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            dense_node_id_t const seed_id = slot == 0                    ? first_value
                                            : slot < planes.connectivity ? own_neighbors.load_slot(slot)
                                                                         : dense_node_id_unknown_k;
            bool const usable = seed_id != dense_node_id_unknown_k && seed_id != node_id;
            bool const fresh = visited.test_and_set(seed_id, lane, usable);
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
            u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
            if (fresh) gathered.ids[ordinal] = seed_id;
            warp_t::sync();
            u32_t const gathered_count = warp_t::count(fresh_mask);
            if (gathered_count)
                dense_vram_score_into_frontier<kit_>( //
                    planes, query_span, query_state, gathered, gathered_count, context, lane, dummy_fold_t {});
        }

    // Walk the graph past the seeds, then re-select the neighbors from the whole pool.
    dense_vram_search_ahead<kit_>( //
        planes, query_span, query_state, view.sampling_bound, view.sampling_bound, visited, gathered, context, lane);
    u32_t const selected_count = dense_vram_select_diverse<kit_>( //
        planes, cache, cache_norms, gathered.ids, selection.selected_ids, context, lane);

    // The insertion's slot convention: neighbors ascending, sentinel tail, a slot-0 self-loop meaning published-empty.
    u32_t* const shadow_slots = view.shadow_neighbors + static_cast<std::size_t>(node_id) * planes.connectivity;
    for (u32_t slot = lane; slot < planes.connectivity; slot += warp_t::width_k)
        shadow_slots[slot] = slot < selected_count ? selection.selected_ids[slot]
                             : slot == 0           ? node_id
                                                   : dense_node_id_unknown_k;
    warp_t::sync();
}

/** Writes each query's nearest @c count IDs and distances out of a search walk, ascending, with
 *  sentinel IDs past the live prefix. */
template <dense_vram_quads_kit kit_>
struct dense_vram_search_sink {
    using fold_t = dummy_fold_t;

    /** The landing for the matched IDs, @b [queries,count]. */
    dense_node_id_t* result_ids;

    /** The landing for the matched distances, @b [queries,count]; null skips the writes. */
    dense_distance_t* result_distances;

    /** Matches kept per query. */
    u32_t count;

    /** The metric the ranked distances convert out of. */
    nk_kernel_kind_t kernel;

    template <typename frontier_>
    __device__ __forceinline__ fold_t open(u32_t, dense_vram_warp_context<frontier_> const&, int) const noexcept {
        return {};
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const&, dense_vram_warp_context<frontier_>& context, u32_t query_index,
                                          int lane) const noexcept {
        // The rank extraction shuffles - every lane participates, only in-range ones store.
        context.frontier.finalize_ranks(lane);
        for (u32_t rank_base = 0; rank_base < count; rank_base += kit_::warp_t::width_k) {
            u32_t const rank = rank_base + lane;
            dense_vram_scored_node_t const ranked_entry = context.frontier.rank_entry(rank);
            if (rank < count) {
                std::size_t const slot = static_cast<std::size_t>(query_index) * count + rank;
                result_ids[slot] = ranked_entry.id();
                if (result_distances)
                    result_distances[slot] = kit_::convert_to_distance(kernel, ranked_entry.distance());
            }
        }
    }
};

/** Writes each query's kept node IDs and native dots out of a multiply walk, and the pairs it scored. */
template <dense_vram_quads_kit kit_>
struct dense_vram_multiply_sink {
    using fold_t = dense_vram_multiply_fold<kit_>;

    /** Node IDs of the kept pairs, @b [queries,wanted], sentinel past them. */
    dense_node_id_t* ids;

    /** Native dots of the kept pairs, parallel to @c ids. */
    typename kit_::dot_t* products;

    /** Pairs scored per query, above @c wanted when a row was truncated. */
    u32_t* counts;

    /** Pairs kept per query. */
    u32_t wanted;

    template <typename frontier_>
    __device__ __forceinline__ fold_t open(u32_t query_index, dense_vram_warp_context<frontier_> const&,
                                           int) const noexcept {
        std::size_t const first = static_cast<std::size_t>(query_index) * wanted;
        return {ids + first, products + first, wanted};
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, dense_vram_warp_context<frontier_>&, u32_t query_index,
                                          int lane) const noexcept {
        for (u32_t slot = fold.written + lane; slot < wanted; slot += kit_::warp_t::width_k)
            fold.ids[slot] = dense_node_id_unknown_k;
        if (lane == 0) counts[query_index] = fold.scored;
    }
};

/** One query's full walk: stages the query into its shared span, walks the graph under the fold the
 *  sink opens and the query's slice of @p rescorings, and lets the sink close the query. Safe against
 *  concurrent insertions: the walk gates on every slot-0 word and never writes the graph. */
template <dense_vram_quads_kit kit_, u32_t resident_blocks_, typename sink_, typename rescoring_, typename frontier_>
__device__ __forceinline__ void dense_vram_walk_one( //
    dense_vram_search_view<kit_> const& view, sink_ const& sink, rescoring_ const& rescorings, u32_t query_index,
    typename kit_::fragment_t* query_span, dense_vram_visited_set<typename kit_::warp_t>& visited, frontier_ frontier,
    dense_vram_gathered<kit_>& gathered, int lane) noexcept {

    using warp_t = typename kit_::warp_t;
    typename kit_::fragment_t const* query_row = view.queries +
                                                 static_cast<std::size_t>(query_index) * view.planes.row_fragments;
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < view.planes.row_fragments; fragment += warp_t::width_k) {
        typename kit_::fragment_t const value = query_row[fragment];
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = dense_vram_reduce_norms<warp_t>(norms_accumulator);
    norms_accumulator = dense_vram_broadcast_norms<warp_t>(norms_accumulator);
    typename kit_::norms_t const query_state = kit_::partials_to_norms(norms_accumulator);
    warp_t::sync();

    dense_vram_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = splitmix64_t {0xA5A5A5A5ull ^ (static_cast<u64_t>(query_index) * 33 + lane)};
    context.scratch = reinterpret_cast<byte_t*>(query_span + view.planes.row_fragments);
    typename sink_::fold_t fold = sink.open(query_index, context, lane);
    rescoring_ const rescoring = rescorings.for_query(query_index, context);
    dense_vram_search_around<kit_, resident_blocks_>( //
        view.planes, query_span, query_state, view.sampling_bound, view.sampling_bound, visited, gathered, context,
        lane, fold, rescoring);
    sink.close(fold, context, query_index, lane);
    context.telemetry.tail_drops = context.frontier.dropped();
    context.telemetry.visited_saturations = visited.saturations_;
    context.telemetry.template flush<warp_t, dense_vram_traversal_t::searching_k>(view.telemetry, lane);
}

} // namespace usearch
} // namespace unum
