/**
 *  @file include/usearch/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The device bodies every dense VRAM index compiled by NVCC or HIP-Clang shares: the views,
 *      the greedy walk, the diversity selection, the back-linker, insertion, refinement, and the
 *      sharded search's rankings.
 *
 *  Every body here is templated on a quads kit, which names its vendor's warp policy as
 *  @c kit_::warp_t; see @c vram_queue.cuh for what a warp policy provides. Walks and builders also
 *  take a schedule, which scores a walk's batch and moves candidate rows into the warp's cache. The
 *  host classes, the kernels that launch these bodies, the launch bounds, the attention fold and
 *  the reachability audits stay in each vendor's own @c vram_<vendor>_index file, beside the kits
 *  in @c vram_<vendor>_metric - the split @c utilities/vram_device.hpp keeps beside its vendor
 *  twins.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the views carry the metric

#include "usearch/utilities/types.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_queue.cuh"

namespace unum::usearch::dense {

/** Rounds a back-linker sweeps its held neighbors before abandoning them - each round tries every
 *  neighbor still held once, after the others' links; abandonments are telemetered. */
inline constexpr u32_t back_link_lock_rounds_k = 256;

/** What a vendor's quads kit binds: one storage dtype on one GPU generation, with the warp policy
 *  its walks shuffle through. */
template <typename kit_>
concept vram_quads_kit = requires(typename kit_::fragment_t fragment, typename kit_::dot_t dot,
                                  typename kit_::weighted_t weighted,
                                  typename kit_::norms_accumulator_t norms_accumulator, typename kit_::norms_t norms,
                                  distance_t ranked, nk_kernel_kind_t kernel) {
    typename kit_::scalar_t;
    typename kit_::warp_t;
    requires std::is_convertible_v<decltype(kit_::capability_k), nk_capability_t>;
    { kit_::accumulate_dot(fragment, fragment, dot) } -> std::same_as<typename kit_::dot_t>;
    { kit_::accumulate_weighted(fragment, f32_t {}, weighted) } -> std::same_as<typename kit_::weighted_t>;
    { kit_::accumulate_norms(fragment, norms_accumulator) } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::partials_to_norms(norms_accumulator) } -> std::same_as<typename kit_::norms_t>;
    { kit_::distance(kernel, dot, norms, norms) } -> std::same_as<distance_t>;
    { kit_::convert_to_distance(kernel, ranked) } -> std::same_as<distance_t>;
};

/** Row @p row of a @b [rows,stride] matrix at @p base - the one place a row offset widens past 32
 *  bits, since node IDs times row strides outgrow them. */
template <typename value_type_>
__device__ __forceinline__ value_type_* vram_row(value_type_* base, u32_t row, u32_t stride) noexcept {
    return base + static_cast<std::size_t>(row) * stride;
}

/** The random stream of @p lane walking for @p stream_key - the node it inserts or refines, or the
 *  query it answers - seeded through @ref mix_bits as the host seeds its streams, so no two lanes
 *  or keys share one. */
template <typename warp_type_>
__device__ __forceinline__ splitmix64_t vram_lane_generator(u32_t stream_key, u32_t lane) noexcept {
    return splitmix64_t {mix_bits(static_cast<u64_t>(stream_key) * warp_type_::width_k + lane + 1)};
}

/** The calling warp's place in a launch: its grid-wide index, its lane, and the grid's warps. */
struct vram_grid_warp_t {
    u32_t index, lane, count;
};

template <typename warp_type_>
__device__ __forceinline__ vram_grid_warp_t vram_grid_warp() noexcept {
    return {(blockIdx.x * blockDim.x + threadIdx.x) / warp_type_::width_k, threadIdx.x % warp_type_::width_k,
            gridDim.x * blockDim.x / warp_type_::width_k};
}

/** Generic cross-lane reduction: folds sub-warp partials of width @p lanes_per_row, a warp-uniform
 *  power of two, down to the leader lane. One tree serves every kit. */
template <typename warp_type_, typename dot_type_>
__device__ __forceinline__ dot_type_ vram_reduce(dot_type_ partial, u32_t lanes_per_row) noexcept {
    for (u32_t offset = lanes_per_row / 2; offset; offset >>= 1)
        partial += warp_type_::shuffle_down(partial, offset, lanes_per_row);
    return partial;
}

/** Full-warp reduction of a two-component norms accumulator - {sum,sumsq} for the
 *  arithmetic kits, {popcount,popcount} for the binary ones. */
template <typename warp_type_, typename pair_type_>
__device__ __forceinline__ pair_type_ vram_reduce_norms(pair_type_ partial) noexcept {
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
__device__ __forceinline__ pair_type_ vram_broadcast_norms(pair_type_ reduced) noexcept {
    reduced.x = warp_type_::shuffle(reduced.x, 0);
    reduced.y = warp_type_::shuffle(reduced.y, 0);
    return reduced;
}

/** Typed views over the device's flat storage, @b [rows,*] each: the packed vectors matrix, the
 *  per-row norms column, and the fixed-capacity neighbors matrix. Under the flat virtual-memory
 *  arena the node ID @b is the location - one multiply, no chunk directory. */
template <vram_quads_kit kit_>
struct vram_flat_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** Vectors matrix, @b [rows,row_fragments] fragments, one aligned row per node. */
    fragment_t* vectors;

    /** Norms column, @b [rows] records, written once pre-publication; null under dot, which reads
     *  none. */
    norms_t* norms;

    /** Neighbors matrix, @b [rows,connectivity] IDs; slot 0 doubles as the lock word. */
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
     *  indices split without division; never below the quotient, and at most one above it for every
     *  word under 2^31, which @c row_of_word corrects. */
    u32_t row_fragments_magic;

    /** The metric the kit's epilogue finalizes dots into - runtime, so one kernel serves every
     *  metric. */
    nk_kernel_kind_t kernel;

    /** Rows physically resident on the executing device, feeding the locality telemetry; both zero
     *  when the view spans one device. */
    u32_t resident_begin = 0, resident_end = 0;

    /** The row holding flattened @p word, exact for every word under 2^31, so for any cache a slice
     *  under 4 GB seats. */
    __device__ __forceinline__ u32_t row_of_word(u32_t word) const noexcept {
        u32_t const row = static_cast<u32_t>((static_cast<u64_t>(word) * row_fragments_magic) >> 31);
        return row - (row * row_fragments > word);
    }
    __device__ __forceinline__ u32_t fragment_of_word(u32_t word, u32_t row) const noexcept {
        return word - row * row_fragments;
    }

    __device__ __forceinline__ fragment_t vector_for(node_id_t node_id, u32_t fragment_index) const noexcept {
        fragment_t const* const row = vector_span_for(node_id);
        return row[fragment_index];
    }
    __device__ __forceinline__ fragment_t* vector_span_for(node_id_t node_id) const noexcept {
        return vram_row(vectors, node_id, row_fragments);
    }

    /** A row's norms, or zeros from the empty column - a warp-uniform branch on a launch
     *  argument. */
    __device__ __forceinline__ norms_t norm_for(node_id_t node_id) const noexcept {
        return norms ? norms[node_id] : norms_t {};
    }
    __device__ __forceinline__ norms_t& norm_ref_for(node_id_t node_id) const noexcept { return norms[node_id]; }
    __device__ __forceinline__ u32_t* neighbors_for(node_id_t node_id) const noexcept {
        return vram_row(neighbors, node_id, connectivity);
    }
};

/**
 *  @brief Typed view over one node's neighbors - the only legal way to touch slots that
 *      concurrent warps may be reading, over the warp policy's device-scope atomics.
 *
 *  Slot 0 is the combined lock and first-neighbor word: the all-ones sentinel while unpublished
 *  or locked, the node's own ID when published empty, otherwise the first neighbor. Acquisition
 *  is one exchange that never waits - spinning against a descheduled owner warp would livelock -
 *  so a held row is revisited after other work, and an unlinked neighbor is the telemetered cost.
 */
template <typename warp_type_>
struct vram_neighbors_ref {
    using warp_t = warp_type_;
    u32_t* row_; // Non-const even for readers: device atomic references cannot bind const words pre-P3323.
    u32_t capacity_;

    __device__ __forceinline__ u32_t capacity() const noexcept { return capacity_; }

    /** Acquire read of slot @p index, pairing with the lock holder's release fence so an ID read
     *  here comes with its node's payload; slot 0 belongs to @c peek_first. */
    __device__ __forceinline__ node_id_t load_slot(u32_t index) const noexcept {
        return warp_t::load_acquire(row_[index]);
    }

    /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current lock
     *  holder. */
    __device__ __forceinline__ void store_slot(u32_t index, node_id_t value) const noexcept {
        warp_t::store_relaxed(row_[index], value);
    }

    /** Acquire-load of the lock word: the sentinel means unpublished or locked, any other value is
     *  the first neighbor ID with the last unlocker's writes visible. */
    __device__ __forceinline__ node_id_t peek_first() const noexcept { return warp_t::load_acquire(row_[0]); }

    /** One acquisition attempt: exchanges the sentinel in with acquire order; an exchange that
     *  receives the sentinel back did @b not acquire. @return The displaced first-neighbor value,
     *  or the sentinel. */
    __device__ __forceinline__ node_id_t lock_if_free() const noexcept {
        return warp_t::exchange_acquire(row_[0], node_id_unknown_k);
    }

    /** Release-stores @p first_value into slot 0 - unlocking and simultaneously (re)publishing the
     *  edited neighbors to every future acquiring reader. */
    __device__ __forceinline__ void unlock(node_id_t first_value) const noexcept {
        warp_t::store_release(row_[0], first_value);
    }
};

/** Which traversals a telemetry flush folds: inserting ones also own the lock and reselection
 *  counters. */
enum class vram_traversal_t : bool { searching_k, inserting_k };

/** Per-warp register telemetry, folded into the device counters once per node; 32-bit in registers,
 *  since every counter is bounded by hops times connectivity, widened at the flush. */
struct vram_telemetry_t {
    u32_t hops = 0, scored = 0, lock_attempts = 0, lock_skips = 0, reselections = 0, restarts = 0, locked_drops = 0;
    u32_t local_scored = 0, remote_scored = 0;
    u32_t tail_drops = 0, visited_saturations = 0;
    u32_t changed_nodes = 0;

    /** Folds the live counters into the device block; the search instantiation skips the lock and
     *  reselection atomics it never touches. */
    template <typename warp_type_, vram_traversal_t traversal_>
    __device__ __forceinline__ void flush(vram_telemetry_block_t* counters, u32_t lane) noexcept {
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
        if constexpr (traversal_ == vram_traversal_t::searching_k) return;
        add(counters->lock_attempts, lock_attempts);
        add(counters->lock_skips, lock_skips);
        add(counters->reselections, reselections);
    }
};

/** One walking warp's state: the candidate pool, the random stream, the running telemetry, and the shared scratch
 *  past its query span, where a rescoring hands its fold what the fold keeps. */
template <typename frontier_>
struct vram_warp_context {
    frontier_ frontier;
    splitmix64_t random_generator;
    vram_telemetry_t telemetry;
    byte_t* scratch = nullptr;
};

/** One warp's staged batch: candidate IDs, then the scored nodes and native dots of up to a
 *  lane-width of them. */
template <vram_quads_kit kit_>
struct vram_gathered {

    /** Candidate node IDs, up to a full vector cache of them. */
    u32_t ids[vram_cache_rows_k];

    /** The scored nodes of one batch, by batch row. */
    vram_scored_node_t scored[kit_::warp_t::width_k];

    /** The native dots of one batch, parallel to @c scored. */
    typename kit_::dot_t dots[kit_::warp_t::width_k];
};

/** Graph products of one query row as a device fold: each scored batch appends its node IDs and
 *  native dots to the row until it fills, while @c scored keeps counting past it. */
template <vram_quads_kit kit_>
struct vram_multiply_fold {
    using dot_t = typename kit_::dot_t;

    /** The row's node ID outputs. */
    node_id_t* ids;

    /** The row's product outputs, positionally paired with @c ids. */
    dot_t* products;

    /** The row's width. */
    u32_t wanted;

    /** Pairs written so far, identical on every lane. */
    u32_t written = 0;

    /** Pairs scored, written or not, identical on every lane. */
    u32_t scored = 0;

    __device__ __forceinline__ walk_demand_t absorb(vram_gathered<kit_> const& gathered, u32_t gathered_count,
                                                    u32_t lane) noexcept {
        u32_t const writable = min_of<u32_t>(gathered_count, wanted - written);
        if (lane < writable) {
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
 *  A sub-warp of @c lanes_per_row lanes owns one row, re-reading the query from its span each step,
 *  so no register budget caps the dimensions. A lane whose row lies past the batch re-reads the
 *  batch's first row instead of taking a guard, since guards around each gather slot would split
 *  the slots' dot chains, which the float kits need interleaved. Each lane finalizes its own row,
 *  so the metric's epilogue runs once per lane. Plain loads are safe: every staged ID came through
 *  an acquiring slot-0 read.
 *
 *  Each lane loads @p fragments_per_load_ adjacent fragments a step, the schedule's load width: the
 *  lanes per row shrink while one step still spans the row, and the steps shrink past it. A wider
 *  load needs every row aligned to it, so rows of another stride score one fragment at a time.
 *
 *  @return This lane's scored entry, or the sentinel on lanes past @p gathered_count.
 */
template <vram_quads_kit kit_, u32_t resident_blocks_ = vram_resident_blocks_per_multiprocessor_k,
          u32_t fragments_per_load_ = 1>
__device__ __forceinline__ vram_scored_node_t vram_score_batch( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, u32_t const* batch_ids, u32_t gathered_count, vram_gathered<kit_>& gathered,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;
    if constexpr (fragments_per_load_ != 1)
        if (storage.row_fragments % fragments_per_load_ != 0)
            return vram_score_batch<kit_, resident_blocks_>(storage, query_span, query_state, batch_ids, gathered_count,
                                                            gathered, lane);

    // Aligned to its own size, so the compiler issues each lane's load as one request.
    struct alignas(sizeof(fragment_t) * fragments_per_load_) load_t {
        fragment_t fragments[fragments_per_load_];
    };
    // 128 concurrent random gathers of a fragment saturate the memory path, so the window is 128
    // fragments over the resident warps, which a wider load fills with fewer gathers.
    constexpr u32_t gathers_in_flight_k =
        min_of(vram_max_gathers_in_flight_k, 128u / (vram_warps_per_block_k * resident_blocks_)) / fragments_per_load_;
    u32_t const lanes_per_row = storage.lanes_per_row >= storage.row_fragments
                                    ? storage.lanes_per_row / fragments_per_load_
                                    : storage.lanes_per_row;
    u32_t const steps_per_row = divide_round_up(storage.steps_per_row, fragments_per_load_);
    u32_t const row_loads = storage.row_fragments / fragments_per_load_;
    u32_t const lanes_log2 = 31 - __clz(lanes_per_row);
    u32_t const rows_per_gather = warp_t::width_k >> lanes_log2;
    u32_t const lane_in_row = lane & (lanes_per_row - 1);
    u32_t const subwarp = lane >> lanes_log2;
    u32_t const live_slots = divide_round_up(gathered_count * lanes_per_row, warp_t::width_k);
    bool const live = lane < gathered_count;
    // Requested before the rows, so the two round trips overlap.
    typename kit_::norms_t const target_state = live ? storage.norm_for(batch_ids[lane]) : typename kit_::norms_t {};
    fragment_t const* const query = query_span + lane_in_row * fragments_per_load_;

    // Slot groups are independent; tiling keeps at most `gathers_in_flight_k` loads live per lane.
#pragma unroll 1
    for (u32_t first_slot = 0; first_slot < live_slots; first_slot += gathers_in_flight_k) {
        fragment_t const* rows[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            u32_t const row = (first_slot + slot) * rows_per_gather + subwarp;
            rows[slot] = storage.vector_span_for(batch_ids[row < gathered_count ? row : 0]) +
                         lane_in_row * fragments_per_load_;
        }
        dot_t accumulators[gathers_in_flight_k];
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) accumulators[slot] = {};
#pragma unroll 1
        for (u32_t step = 0; step != steps_per_row; ++step) {
            if (lane_in_row + (step << lanes_log2) < row_loads) {
                u32_t const offset = (step << lanes_log2) * fragments_per_load_;
                load_t loaded[gathers_in_flight_k];
#pragma unroll
                for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot)
                    loaded[slot] = *reinterpret_cast<load_t const*>(rows[slot] + offset);
#pragma unroll
                for (u32_t fragment = 0; fragment != fragments_per_load_; ++fragment) {
                    fragment_t const query_value = query[offset + fragment];
#pragma unroll
                    for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot)
                        accumulators[slot] = kit_::accumulate_dot(loaded[slot].fragments[fragment], query_value,
                                                                  accumulators[slot]);
                }
            }
        }
#pragma unroll
        for (u32_t slot = 0; slot != gathers_in_flight_k; ++slot) {
            // Warp-uniform, so the reduction's full-mask shuffles stay legal.
            if (first_slot + slot >= live_slots) break;
            dot_t const dot = vram_reduce<warp_t>(accumulators[slot], lanes_per_row);
            u32_t const row = (first_slot + slot) * rows_per_gather + subwarp;
            if (lane_in_row == 0 && row < gathered_count) gathered.dots[row] = dot;
        }
    }
    warp_t::sync();
    if (!live) return vram_scored_node_t::sentinel();
    vram_scored_node_t const scored = vram_scored_node_t::make(
        kit_::distance(storage.kernel, gathered.dots[lane], query_state, target_state), batch_ids[lane]);
    gathered.scored[lane] = scored;
    return scored;
}

/** What a vendor's schedule binds for one kit: how a walk scores its gathered batch, and how one
 *  warp's candidate rows reach its vector cache, in shared memory or its arena slice, apart from
 *  the kit's arithmetic. @c score_batch lands what @ref vram_score_batch lands for the batch's
 *  IDs, its load window sized for @c resident_blocks_ blocks per multiprocessor. @c stage_rows
 *  queues the rows an ID span names into the cache's leading rows over @c stages_k stages, and
 *  @c await_rows lands at least that many leading rows for every lane; every staged row lands
 *  before the next staging. A schedule is one warp's state over its @c shared_t in static shared
 *  memory, picked by the class like the kit. */
template <typename schedule_, typename kit_>
concept vram_schedule = requires(schedule_& schedule, u32_t rows, vram_flat_view<kit_> const& storage,
                                 typename kit_::fragment_t const* query_span, typename kit_::norms_t query_state,
                                 vram_gathered<kit_>& gathered, u32_t lane) {
    typename schedule_::shared_t;
    requires std::is_convertible_v<decltype(schedule_::stages_k), u32_t>;
    schedule.await_rows(rows);
    {
        schedule_::template score_batch<kit_, vram_resident_blocks_per_multiprocessor_k>(
            storage, query_span, query_state, gathered, rows, lane)
    } -> std::same_as<vram_scored_node_t>;
};

/** The plain walk's rescoring: none, so every batch lands as the kit scored it. */
struct vram_no_rescoring {

    template <typename frontier_>
    __device__ __forceinline__ vram_no_rescoring for_query(u32_t, vram_warp_context<frontier_> const&) const noexcept {
        return {};
    }
};

/** Indexer heads as a device rescoring: each scored dot becomes the score Σₕ wₕ · ReLU(qₕ · k) and its distance
 *  1 − score, the warp scoring the batch once per head exactly as it scored the navigating query. */
template <vram_quads_kit kit_>
struct vram_indexer_rescoring {
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
    __device__ __forceinline__ vram_indexer_rescoring for_query(u32_t query_index,
                                                                vram_warp_context<frontier_> const&) const noexcept {
        vram_indexer_rescoring narrowed = *this;
        narrowed.heads = vram_row(heads, query_index, heads_count * row_fragments);
        narrowed.weights = vram_row(weights, query_index, heads_count);
        return narrowed;
    }

    __device__ __forceinline__ vram_scored_node_t rescore(vram_flat_view<kit_> const& storage,
                                                          vram_gathered<kit_>& gathered, u32_t gathered_count,
                                                          u32_t lane, vram_scored_node_t scored) const noexcept {
        bool const live = lane < gathered_count;
        f32_t indexed = 0;
        for (u32_t head = 0; head != heads_count; ++head) {
            vram_score_batch<kit_>(storage, heads + head * row_fragments, typename kit_::norms_t {}, gathered.ids,
                                   gathered_count, gathered, lane);
            if (live) indexed += weights[head] * fmaxf(static_cast<f32_t>(gathered.dots[lane]), 0.f);
            // The next head's scoring overwrites the dots, so every lane reads its own first.
            warp_t::sync();
        }
        if (!live) return scored;
        gathered.dots[lane] = indexed;
        scored = vram_scored_node_t::make(1 - indexed, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/** Scores one gathered batch against the query through @p schedule_, lets @p rescoring rewrite
 *  the scores, folds it into the pool, and hands it to @p fold - whose answer says whether the
 *  walk goes on. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_,
          u32_t resident_blocks_ = vram_resident_blocks_per_multiprocessor_k, typename frontier_, typename fold_,
          typename rescoring_ = vram_no_rescoring>
__device__ __forceinline__ walk_demand_t vram_score_into_frontier( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, vram_gathered<kit_>& gathered, u32_t gathered_count,
    vram_warp_context<frontier_>& context, u32_t lane, fold_&& fold,
    rescoring_ const& rescoring = rescoring_ {}) noexcept {
    using warp_t = typename kit_::warp_t;
    context.telemetry.scored += gathered_count;
    if (storage.resident_end) {
        bool const counted = lane < gathered_count;
        u32_t const candidate = counted ? gathered.ids[lane] : 0;
        typename warp_t::mask_t const local_mask = warp_t::ballot(counted && candidate >= storage.resident_begin &&
                                                                  candidate < storage.resident_end);
        context.telemetry.local_scored += warp_t::count(local_mask);
        context.telemetry.remote_scored += gathered_count - warp_t::count(local_mask);
    }
    vram_scored_node_t staged = schedule_::template score_batch<kit_, resident_blocks_>( //
        storage, query_span, query_state, gathered, gathered_count, lane);
    // Even a call that changes nothing reorders the register-capped wide walk, so the plain walk makes none.
    if constexpr (!std::is_same_v<rescoring_, vram_no_rescoring>)
        staged = rescoring.rescore(storage, gathered, gathered_count, lane, staged);
    // No dedup: the exact visited set upstream guarantees every ID stages once.
    context.frontier.push_batch(staged, lane);
    return fold.absorb(gathered, gathered_count, lane);
}

/**
 *  @brief Samples one lane-wide batch of distinct, published landmarks in
 *      @b [sampling_first,sampling_bound) into @p candidate_ids.
 *
 *  The draws are uniform and gated on the slot-0 publication word: over the first few probes a lane
 *  meeting an unpublished row draws again, then each lane probes forward past unpublished and
 *  visited rows for one lap of the range.
 *
 *  @return The number of staged landmarks; zero once every published row of the range is
 *      visited, or the visited set saturated.
 */
template <vram_quads_kit kit_>
__device__ __forceinline__ u32_t vram_sample_landmarks( //
    vram_flat_view<kit_> const& storage, node_id_t sampling_first, node_id_t sampling_bound,
    vram_visited_set<typename kit_::warp_t>& visited, splitmix64_t& random_generator,
    u32_t (&candidate_ids)[vram_cache_rows_k], u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    // Rows still inserting gather at the range's top, which a redraw leaves far sooner than a lap.
    constexpr u64_t redraws_k = 16;
    node_id_t const span = sampling_bound - sampling_first;
    uniform_below<node_id_t> const uniform {span};
    node_id_t draw = sampling_first + uniform(random_generator);
    bool fresh = false;
    u32_t const saturations = visited.saturations_;
    // The trip count is warp-uniform, since every lane takes part in each collective claim, and a
    // saturated set reports every claim visited, so the lap ends at its first saturation instead of
    // probing the whole range.
    u64_t const redraws = min_of<u64_t>(span, redraws_k);
    for (u64_t probes = 0; probes != redraws + span && !warp_t::all(fresh) && visited.saturations_ == saturations;
         ++probes) {
        bool published = false;
        if (!fresh) {
            vram_neighbors_ref<warp_t> const row {storage.neighbors_for(draw), storage.connectivity};
            published = row.peek_first() != node_id_unknown_k;
        }
        bool const claimed = visited.test_and_set(published ? draw : node_id_unknown_k, lane);
        if (!fresh && !claimed)
            draw = !published && probes < redraws ? sampling_first + uniform(random_generator)
                   : draw + 1 == sampling_bound   ? sampling_first
                                                  : draw + 1;
        fresh = fresh || claimed;
    }
    typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
    u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
    if (fresh) candidate_ids[ordinal] = draw;
    warp_t::sync();
    return warp_t::count(fresh_mask);
}

/** Continues a greedy best-first walk toward the query over whatever the frontier holds - one warp,
 *  one traversal, its landmark seeds drawn from @b [sampling_first,sampling_bound); pre-staged
 *  entries compete with them and must already be visited, and each call walks its own hop budget.
 *  Read-only on the graph: no locks taken, a transiently locked candidate held through one other
 *  expansion and dropped if still locked then, and no neighbor at or past @p first_invisible_id
 *  followed, since a batch's back-links splice its IDs into visible rows before the host turns its
 *  watermark. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_,
          u32_t resident_blocks_ = vram_resident_blocks_per_multiprocessor_k, typename frontier_,
          typename fold_ = dummy_fold_t, typename rescoring_ = vram_no_rescoring>
__device__ __forceinline__ void vram_search_ahead( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, node_id_t sampling_first, node_id_t sampling_bound,
    node_id_t first_invisible_id, vram_visited_set<typename kit_::warp_t>& visited, vram_gathered<kit_>& gathered,
    vram_warp_context<frontier_>& context, u32_t lane, fold_&& fold = fold_ {},
    rescoring_ const& rescoring = rescoring_ {}) noexcept {

    using warp_t = typename kit_::warp_t;
    frontier_& frontier = context.frontier;
    vram_telemetry_t& telemetry = context.telemetry;
    u32_t hops_cap = telemetry.hops + frontier.hops_budget();

    // The one candidate held for its second look, and the hop it was held at; a drained frontier
    // moves that hop out of reach, so the next iteration looks at once.
    vram_scored_node_t held = vram_scored_node_t::sentinel();
    u32_t held_at_hop = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = vram_sample_landmarks<kit_>( //
                storage, sampling_first, sampling_bound, visited, context.random_generator, gathered.ids, lane);
            if (!gathered_count) break; // Every published row of the range is already visited.
            if (vram_score_into_frontier<kit_, schedule_, resident_blocks_>( //
                    storage, query_span, query_state, gathered, gathered_count, context, lane, fold, rescoring) ==
                walk_demand_t::sated_k)
                return;
            seeding = false;
        }
        else {
            bool const revisiting = !(held == vram_scored_node_t::sentinel()) && telemetry.hops != held_at_hop;
            vram_scored_node_t const candidate = revisiting ? held : frontier.pop_nearest(lane);
            if (revisiting) held = vram_scored_node_t::sentinel();
            if (candidate == vram_scored_node_t::sentinel()) {
                if (!(held == vram_scored_node_t::sentinel())) {
                    held_at_hop = ~0u;
                    continue;
                }
                // A drained frontier before an exhausted walk re-seeds at one hop of the budget,
                // so a walk whose full tail drops every landmark still ends on the cap.
                if (frontier.exhausted() || telemetry.hops >= hops_cap) break;
                seeding = true;
                ++telemetry.restarts, --hops_cap;
                continue;
            }
            if (frontier.beyond_radius(candidate)) break;
            if (telemetry.hops >= hops_cap) break;
            ++telemetry.hops;

            // Only the leader reads the racing lock word: per-lane reads could disagree on one
            // unlock, and a value-dependent `continue` would fork the warp across iterations.
            vram_neighbors_ref<warp_t> const row {storage.neighbors_for(candidate.id()), storage.connectivity};
            node_id_t first_value = lane == 0 ? row.peek_first() : node_id_unknown_k;
            first_value = warp_t::shuffle(first_value, 0);
            // The shuffle moves the word, not the leader's acquire; the lane barrier orders the
            // other lanes' slot loads after it.
            warp_t::sync();
            if (first_value == node_id_unknown_k) {
                // Held through one other expansion instead of waited on; a second refusal drops it.
                if (revisiting || !(held == vram_scored_node_t::sentinel())) ++telemetry.locked_drops;
                else held = candidate, held_at_hop = telemetry.hops;
                continue;
            }
            // The first warp-width of slots is loop-free, which keeps the hot path compact.
            node_id_t const neighbor_id = lane == 0               ? first_value
                                          : lane < row.capacity() ? row.load_slot(lane)
                                                                  : node_id_unknown_k;
            // The bound never exceeds the ID space, so it rejects the sentinel too.
            bool const usable = neighbor_id < first_invisible_id && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(usable ? neighbor_id : node_id_unknown_k, lane);
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
            u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
            if (fresh) gathered.ids[ordinal] = neighbor_id;
            warp_t::sync();
            u32_t const gathered_count = warp_t::count(fresh_mask);
            if (gathered_count && vram_score_into_frontier<kit_, schedule_, resident_blocks_>( //
                                      storage, query_span, query_state, gathered, gathered_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                return;
            // Wider connectivities continue a warp-width at a time; the bound is warp-uniform for the collectives.
            for (u32_t slot_base = warp_t::width_k; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
                u32_t const slot = slot_base + lane;
                node_id_t const wide_id = slot < storage.connectivity ? row.load_slot(slot) : node_id_unknown_k;
                bool const wide_usable = wide_id < first_invisible_id && wide_id != candidate.id();
                bool const wide_fresh = visited.test_and_set(wide_usable ? wide_id : node_id_unknown_k, lane);
                typename warp_t::mask_t const wide_mask = warp_t::ballot(wide_fresh);
                u32_t const wide_ordinal = warp_t::count(wide_mask & warp_t::below(lane));
                if (wide_fresh) gathered.ids[wide_ordinal] = wide_id;
                warp_t::sync();
                u32_t const wide_count = warp_t::count(wide_mask);
                if (wide_count && vram_score_into_frontier<kit_, schedule_, resident_blocks_>( //
                                      storage, query_span, query_state, gathered, wide_count, context, lane, fold,
                                      rescoring) == walk_demand_t::sated_k)
                    return;
            }
        }
    }
    telemetry.locked_drops += !(held == vram_scored_node_t::sentinel());
}

/** Greedy best-first walk from random published landmarks toward the query: a fresh frontier
 *  handed to @ref vram_search_ahead. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_,
          u32_t resident_blocks_ = vram_resident_blocks_per_multiprocessor_k, typename frontier_,
          typename fold_ = dummy_fold_t, typename rescoring_ = vram_no_rescoring>
__device__ __forceinline__ void vram_search_around( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, node_id_t sampling_bound, node_id_t first_invisible_id,
    vram_visited_set<typename kit_::warp_t>& visited, vram_gathered<kit_>& gathered,
    vram_warp_context<frontier_>& context, u32_t lane, fold_&& fold = fold_ {},
    rescoring_ const& rescoring = rescoring_ {}) noexcept {
    context.frontier.reset(lane);
    vram_search_ahead<kit_, schedule_, resident_blocks_>( //
        storage, query_span, query_state, 0, sampling_bound, first_invisible_id, visited, gathered, context, lane, fold,
        rescoring);
}

/** Runtime-strided view over the warp's vector cache - rows of @c row_fragments_ fragments each,
 *  aliasing the scratch slice, in shared memory or the warp's arena slice. */
template <typename fragment_type_>
struct vram_vector_cache {
    fragment_type_* data_;
    u32_t row_fragments_;

    __device__ __forceinline__ fragment_type_* row(u32_t row_index) const noexcept {
        return data_ + row_index * row_fragments_;
    }
    __device__ __forceinline__ fragment_type_& at(u32_t row_index, u32_t fragment_index) const noexcept {
        return row(row_index)[fragment_index];
    }
};

/** Warp-cooperative ascending buffer of scored nodes over a shared-memory span: chunks arrive
 *  pre-sorted from the lane-wide bitonic network and merge in place, entries past @c capacity_
 *  drop. */
template <typename warp_type_>
struct vram_sorted_buffer {
    using warp_t = warp_type_;
    vram_scored_node_t* entries_;
    u32_t size_ = 0;
    u32_t capacity_;

    __device__ __forceinline__ vram_scored_node_t at(u32_t rank) const noexcept { return entries_[rank]; }

    __device__ __forceinline__ void insert_batch(vram_scored_node_t value, u32_t count, u32_t lane) noexcept {
        constexpr u32_t span_batches_k = divide_round_up(vram_cache_rows_k, warp_t::width_k);
        bool const incoming = lane < count;
        u32_t incoming_rank = incoming ? lane : 0;
        for (u32_t i = 0; i != size_; ++i) // Warp-uniform bound: every lane walks the span.
            incoming_rank += incoming && entries_[i].distance_bits() <= value.distance_bits();
        // Existing entries shift by the number of incoming values strictly ahead of them;
        // ties resolve the other way than above, so no two final positions collide.
        vram_scored_node_t moved[span_batches_k];
        u32_t moved_to[span_batches_k];
        u32_t batch = 0;
        for (u32_t base = 0; base < size_; base += warp_t::width_k, ++batch) {
            u32_t const index = base + lane;
            bool const active = index < size_;
            moved[batch] = active ? entries_[index] : vram_scored_node_t::sentinel();
            u32_t shift = 0;
            for (u32_t source = 0; source != count; ++source) // Warp-uniform: shuffles inside.
                shift += (shuffle_from<warp_t>(value, source).distance_bits() < moved[batch].distance_bits()) && active;
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
struct vram_selection_scratch_t {
    u32_t* selected_ids;
    vram_scored_node_t* scored_entries;
    u32_t* kept_positions;

    /** Each cached candidate's distance to the query, by cache row, staged once so the selection
     *  loop reads memory, not a register pool it would unroll to index. */
    f32_t* candidate_distances;

    /** Carves the four spans past the last row of @p cache: the ascending candidate scores first,
     *  since they carry the widest alignment, then the two ID spans and the staged distances. */
    template <typename fragment_type_>
    static __device__ __forceinline__ vram_selection_scratch_t carve(vram_vector_cache<fragment_type_> cache,
                                                                     u32_t connectivity) noexcept {
        u32_t const slots = min_of(connectivity, vram_cache_rows_k);
        vram_scored_node_t* const scored_entries = reinterpret_cast<vram_scored_node_t*>(cache.row(vram_cache_rows_k));
        u32_t* const selected_ids = reinterpret_cast<u32_t*>(scored_entries + slots);
        u32_t* const kept_positions = selected_ids + slots;
        return {selected_ids, scored_entries, kept_positions, reinterpret_cast<f32_t*>(kept_positions + slots)};
    }
};

/**
 *  @brief Greedy diversity selection over the ascending pool prefix: a candidate is kept only
 *      when it is closer to the query than to every kept neighbor, so neighbors spread across
 *      directions; the kept neighbors are ascending by construction.
 *  @return The kept count; @p selection.selected_ids holds that many node IDs on return.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, typename frontier_>
__device__ __forceinline__ u32_t vram_select_diverse( //
    vram_flat_view<kit_> const& storage, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    typename kit_::norms_t (&cache_norms)[vram_cache_rows_k], u32_t (&candidate_ids)[vram_cache_rows_k],
    vram_selection_scratch_t selection, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    u32_t* const selected_ids = selection.selected_ids;
    frontier_& frontier = context.frontier;
    frontier.settle(lane);

    // Stage the ascending prefix IDs and distances & cache their vectors and norms.
    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < vram_cache_rows_k; base += warp_t::width_k) {
        vram_scored_node_t const entry = base + lane < vram_cache_rows_k ? frontier.rank_entry(base + lane)
                                                                         : vram_scored_node_t::sentinel();
        typename warp_t::mask_t const live = warp_t::ballot(!(entry == vram_scored_node_t::sentinel()));
        if (!(entry == vram_scored_node_t::sentinel())) {
            candidate_ids[base + lane] = entry.id();
            selection.candidate_distances[base + lane] = entry.distance();
        }
        candidates_count += warp_t::count(live);
        if (live != warp_t::all_k) break;
    }
    warp_t::sync();
    schedule.stage_rows(storage, cache, candidate_ids, candidates_count, lane);
    for (u32_t row = lane; row < candidates_count; row += warp_t::width_k)
        cache_norms[row] = storage.norm_for(candidate_ids[row]);
    warp_t::sync();

    u32_t selected_count = 0;
    // Rolled: unrolled, every candidate would carry its own pairwise dot and metric dispatch.
#pragma unroll 1
    for (u32_t candidate_index = 0; candidate_index < candidates_count && selected_count < storage.connectivity;
         ++candidate_index) {
        // Kept rows precede the candidate, so later stages land behind the earlier checks.
        schedule.await_rows(candidate_index + 1);
        f32_t const to_query = selection.candidate_distances[candidate_index];
        bool diverse = true;
        for (u32_t base = 0; base < selected_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < selected_count) {
                u32_t const kept_position = selected_ids[base + lane];
                typename kit_::dot_t accumulator {};
                // Unrolled deeper, the pairwise loops grow the builders by thousands of
                // `accumulate_dot` copies.
#pragma unroll 4
                for (u32_t fragment = 0; fragment != storage.row_fragments; ++fragment)
                    accumulator = kit_::accumulate_dot(cache.at(candidate_index, fragment),
                                                       cache.at(kept_position, fragment), accumulator);
                f32_t const between = kit_::distance(storage.kernel, accumulator, cache_norms[candidate_index],
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
    // The cache outlives this selection, so no staged copy may still be landing in it.
    schedule.await_rows(candidates_count);
    // The slots diversity leaves empty take the nearest rejected candidates, as the CPU's `select_diverse` does:
    // on a dot graph, whose hubs reject one another, a row of diverse picks alone is a handful of links.
    u32_t const diverse_count = selected_count;
    for (u32_t candidate_index = 0, matched = 0;
         candidate_index < candidates_count && selected_count < storage.connectivity; ++candidate_index) {
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

/** Back-links the node: locks each selected neighbor's slots through one slot-0 exchange, appends
 *  when there is room, and otherwise re-runs the diversity heuristic over every neighbor plus the
 *  newcomer under the lock. At most one lock is held at a time, only after this node's own
 *  publication. A held neighbor is never waited on: it is deferred past the others, every round
 *  retries the deferred ones once, and the ones still held after the last round are skipped. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_>
__device__ __forceinline__ void vram_back_link_neighbors( //
    vram_flat_view<kit_> const& storage, node_id_t new_node_id, typename kit_::norms_t new_norms,
    vram_selection_scratch_t selection, u32_t selected_count, vram_vector_cache<typename kit_::fragment_t> cache,
    schedule_& schedule, typename kit_::norms_t (&cache_norms)[vram_cache_rows_k], vram_gathered<kit_>& gathered,
    vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    // The last two cache rows host the newcomer and the target, whose vector the scored batch reads as its query.
    constexpr u32_t newcomer_row_k = vram_cache_rows_k - 1;
    constexpr u32_t target_row_k = vram_cache_rows_k - 2;
    // Those two rows also bound how many neighbors one warp can re-select; wider connectivities keep sentinel tails.
    u32_t const connectivity = min_of(storage.connectivity, vram_cache_rows_k - 2u);

    u32_t pending_count = selected_count;
    for (u32_t round = 0; pending_count && round != back_link_lock_rounds_k; ++round) {
        u32_t deferred_count = 0;
        for (u32_t pending_index = 0; pending_index != pending_count; ++pending_index) {
            node_id_t const target = selection.selected_ids[pending_index];
            vram_neighbors_ref<warp_t> const row {storage.neighbors_for(target), connectivity};
            // A retry reads before it writes, so a held row's line is not hammered while its holder
            // rewrites it.
            node_id_t first_value = lane == 0 && (round == 0 || row.peek_first() != node_id_unknown_k)
                                        ? row.lock_if_free()
                                        : node_id_unknown_k;
            first_value = warp_t::shuffle(first_value, 0);
            warp_t::sync(); // The sweep below must follow the leader's acquiring exchange on every lane.
            ++telemetry.lock_attempts;
            if (first_value == node_id_unknown_k) {
                // The deferred prefix never passes the read cursor, so it compacts in place.
                if (lane == 0) selection.selected_ids[deferred_count] = target;
                ++deferred_count;
                continue;
            }
            if (first_value == target) { // Published-empty self-loop: the newcomer is the first neighbor.
                if (lane == 0) row.unlock(new_node_id);
                continue;
            }

            // Read every locked neighbor a warp-width at a time, remembering the first empty
            // slot and staging every live neighbor ID for the potential re-selection.
            u32_t empty_slot = connectivity;
            for (u32_t base = 0; base < connectivity; base += warp_t::width_k) {
                u32_t const slot = base + lane;
                node_id_t const slot_value = slot == 0             ? first_value
                                             : slot < connectivity ? row.load_slot(slot)
                                                                   : node_id_unknown_k;
                if (slot < connectivity) gathered.ids[slot] = slot_value;
                typename warp_t::mask_t const empty_mask = warp_t::ballot(slot < connectivity &&
                                                                          slot_value == node_id_unknown_k);
                if (empty_mask && empty_slot == connectivity) empty_slot = base + warp_t::first(empty_mask);
            }
            warp_t::sync();
            // Every slot store below is relaxed; this fence heads each lane's stores, so an
            // acquiring reader of any slot sees the newcomer's payload.
            warp_t::fence_release();
            if (empty_slot != connectivity) { // Fast path: append into that slot, unlock unchanged.
                if (lane == 0) row.store_slot(empty_slot, new_node_id);
                warp_t::sync();
                if (lane == 0) row.unlock(first_value);
                continue;
            }

            // Every slot taken: re-select diversely over the neighbors plus the newcomer,
            // ascending by distance to the target itself.
            ++telemetry.reselections;
            for (u32_t word = lane; word < storage.row_fragments; word += warp_t::width_k)
                cache.at(target_row_k, word) = storage.vector_for(target, word);
            warp_t::sync();
            typename kit_::norms_t const target_state = storage.norm_for(target);

            // The newcomer's distance to the target, from its freshly-owned vector row.
            typename kit_::dot_t accumulator {};
            for (u32_t word = lane; word < storage.row_fragments; word += warp_t::width_k)
                accumulator = kit_::accumulate_dot(storage.vector_for(new_node_id, word), cache.at(target_row_k, word),
                                                   accumulator);
            typename kit_::dot_t const self_dot = vram_reduce<warp_t>(accumulator, warp_t::width_k);
            f32_t const newcomer_distance = kit_::distance(storage.kernel, warp_t::shuffle(self_dot, 0), target_state,
                                                           new_norms);
            telemetry.scored += connectivity + 1;

            // Score the neighbors a warp-width at a time, pre-sort each batch bitonically, and
            // merge them into the ascending sorted buffer.
            vram_sorted_buffer<warp_t> sorted_candidates {selection.scored_entries, 0, connectivity};
            for (u32_t base = 0; base < connectivity; base += warp_t::width_k) {
                u32_t const batch_count = min_of(connectivity - base, warp_t::width_k);
                vram_scored_node_t const scored = vram_score_batch<kit_>( //
                    storage, cache.row(target_row_k), target_state, gathered.ids + base, batch_count, gathered, lane);
                vram_scored_node_t const sorted_batch = sort_lanes<warp_t>(
                    lane < batch_count ? vram_scored_node_t::make(scored.distance(), base + lane) // ID = slot index.
                                       : vram_scored_node_t::sentinel(),
                    lane);
                sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
            }

            // Stage the neighbors' vectors, and the newcomer at the tail row, then merge-walk
            // them ascending by distance to the target - an order of rows, not of positions, so
            // every row lands first.
            schedule.stage_rows(storage, cache, gathered.ids, connectivity, lane);
            for (u32_t word = lane; word < storage.row_fragments; word += warp_t::width_k)
                cache.at(newcomer_row_k, word) = storage.vector_for(new_node_id, word);
            for (u32_t cache_row = lane; cache_row < connectivity; cache_row += warp_t::width_k)
                cache_norms[cache_row] = storage.norm_for(gathered.ids[cache_row]);
            if (lane == 0) cache_norms[newcomer_row_k] = new_norms;
            schedule.await_rows(connectivity);
            warp_t::sync();

            // The merge walk: the sorted neighbors and the newcomer, ascending by distance to the
            // target, as cache rows.
            constexpr u32_t walked_out_k = ~0u;
            u32_t existing_index = 0;
            bool newcomer_pending = true;
            f32_t candidate_distance = 0;
            auto next_candidate = [&]() noexcept {
                vram_scored_node_t const next_link = existing_index < sorted_candidates.size_
                                                         ? sorted_candidates.at(existing_index)
                                                         : vram_scored_node_t::sentinel();
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
#pragma unroll 4
                        for (u32_t fragment = 0; fragment != storage.row_fragments; ++fragment)
                            pair_accumulator = kit_::accumulate_dot(cache.at(candidate_position, fragment),
                                                                    cache.at(kept_position, fragment),
                                                                    pair_accumulator);
                        f32_t const between = kit_::distance(storage.kernel, pair_accumulator,
                                                             cache_norms[candidate_position],
                                                             cache_norms[kept_position]);
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

            // The slots diversity leaves empty take the nearest rejected candidates, as the CPU's
            // and the refine merge's do: a row that only shrinks drops links its neighbors may be
            // reached through, and on a dot graph, whose hubs reject one another, it strands half
            // the nodes. The kept prefix is a subsequence of the same walk.
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
                node_id_t linked = node_id_unknown_k;
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
        pending_count = deferred_count;
        warp_t::sync(); // The next round reads the IDs the leader compacted.
    }
    telemetry.lock_skips += pending_count;
}

/** Layout of one warp's slice of the traversal arena, computed on the host and carried by the
 *  views: the exact visited set, the wide frontier's spill tail when it has one, then a builder's
 *  scratch when shared memory cannot seat its rows. */
struct vram_arena_shape_t {
    u32_t visited_slots;
    u32_t visited_hash_shift;
    u32_t tail_entries;

    /** Bytes of a builder's scratch - query span, candidate cache and re-selection spans - kept
     *  here instead of dynamic shared memory, or zero when shared memory seats it. */
    u32_t cache_bytes;

    /** Bytes of one warp's slice, a multiple of 16 so every visited span starts on the boundary its
     *  paired probes load from; the slices together may pass 4 GB, one alone never does. */
    __device__ __forceinline__ u32_t slice_bytes() const noexcept {
        u32_t const bytes = visited_slots * sizeof(u64_t) + tail_entries * sizeof(vram_scored_node_t) + cache_bytes;
        return (bytes + 15u) & ~15u;
    }
    __device__ __forceinline__ u64_t* visited_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<u64_t*>(vram_row(base, warp_index, slice_bytes()));
    }
    __device__ __forceinline__ vram_scored_node_t* tail_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<vram_scored_node_t*>(visited_span(base, warp_index) + visited_slots);
    }
    __device__ __forceinline__ byte_t* cache_span(byte_t* base, u32_t warp_index) const noexcept {
        return reinterpret_cast<byte_t*>(tail_span(base, warp_index) + tail_entries);
    }
};

/** One warp's slice of a launch: its coordinates, its scratch - dynamic shared memory, or its arena
 *  slice's cache span when shared memory cannot seat it - and its arena visited span, the shared
 *  prologue of every traversal kernel. */
struct vram_warp_slice_t {
    u32_t lane, warp, warp_index;
    byte_t* scratch;
    u64_t* visited_slots;
};

/** Carves the calling warp's slice out of @p view and the block's @p dynamic_scratch; the arena
 *  base is allocation-aligned and every slice a multiple of 16 bytes, so each visited span takes
 *  paired 16-byte probes. */
template <typename warp_type_, typename view_>
__device__ __forceinline__ vram_warp_slice_t vram_carve_slice(view_ const& view, byte_t* dynamic_scratch) noexcept {
    u32_t const warp = threadIdx.x / warp_type_::width_k, lane = threadIdx.x % warp_type_::width_k;
    u32_t const warp_index = blockIdx.x * vram_warps_per_block_k + warp;
    byte_t* const scratch = view.arena_shape.cache_bytes ? view.arena_shape.cache_span(view.arena, warp_index)
                                                         : dynamic_scratch + warp * view.scratch_bytes_per_warp;
    return {lane, warp, warp_index, scratch, view.arena_shape.visited_span(view.arena, warp_index)};
}

/** Geometry & storage pointers one insertion launch operates on. Every storage row an assignable
 *  new_node_id can touch is mapped and sentinel-filled ahead of the launch. */
template <vram_quads_kit kit_>
struct vram_build_view {
    vram_flat_view<kit_> storage;

    /** Incoming vectors, @b [rows,row_fragments] in assigned-ID order, device-resident. */
    typename kit_::fragment_t const* staging;

    /** IDs strictly below this bound may be claimed & inserted by this launch. */
    u32_t end_id;

    /** Bytes of the per-warp dynamic-shared slice: the query span, grown to fit the candidate
     *  cache and the re-selection spans; zero when the arena shape carries them instead. */
    u32_t scratch_bytes_per_warp;

    /** The batch's base ID: staging row 0 lands there, and the warps assign themselves offsets
     *  within @b [first_id,end_id). */
    node_id_t first_id;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice: the exact visited set, the arena frontier's tail on the rung past
     *  the register ladder, and the builders' scratch when shared memory cannot seat it. */
    vram_arena_shape_t arena_shape;

    /** The arena frontier's expansion budget on the rung past the register ladder; the ladder's
     *  rungs ignore it. */
    u32_t expansion;

    /** The context's epoch floor at launch: batch-local epochs ride above it, so stale arena slots
     *  compare older and a warm arena is never re-zeroed. */
    u32_t epoch_base;

    /** The index's published-row count, advanced at each slot-0 release and read by the host; it
     *  implies nothing about @b which rows. */
    u32_t* published_rows;
    vram_telemetry_block_t* telemetry;

    /** Rows of the batch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;
};

/** Inputs & outputs of one search launch: a batch of queries in, the nearest IDs and - when
 *  requested - their distances out, both ascending per query. */
template <vram_quads_kit kit_>
struct vram_search_view {
    vram_flat_view<kit_> storage;

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
    vram_arena_shape_t arena_shape;

    /** The context's epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    vram_telemetry_block_t* telemetry;

    /** Queries of the batch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_queries;
};

/** Inserts the node identified by @p new_node_id: lands the payload and norms, walks the published
 *  graph, selects diverse neighbors, publishes with one release store, then back-links them - one
 *  warp per node. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, typename frontier_>
__device__ __forceinline__ void vram_insert_one( //
    vram_build_view<kit_> const& view, node_id_t new_node_id, vram_visited_set<typename kit_::warp_t>& visited,
    typename kit_::fragment_t* query_span, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_selection_scratch_t selection, typename kit_::norms_t (&cache_norms)[vram_cache_rows_k],
    vram_gathered<kit_>& gathered, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_flat_view<kit_> const& storage = view.storage;

    // One pass lands the payload, derives the norms, and parks the query in its span.
    typename kit_::fragment_t const* staging_row = vram_row(view.staging, new_node_id - view.first_id,
                                                            storage.row_fragments);
    typename kit_::fragment_t* owned_row = storage.vector_span_for(new_node_id);
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < storage.row_fragments; fragment += warp_t::width_k) {
        typename kit_::fragment_t const value = staging_row[fragment];
        owned_row[fragment] = value;
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = vram_reduce_norms<warp_t>(norms_accumulator);
    norms_accumulator = vram_broadcast_norms<warp_t>(norms_accumulator);
    typename kit_::norms_t const new_norms = kit_::partials_to_norms(norms_accumulator);
    if (lane == 0 && storage.norms) storage.norm_ref_for(new_node_id) = new_norms;
    typename kit_::norms_t const query_state = new_norms;
    warp_t::sync();

    // Walk the watermark - a virgin index publishes an empty node instead.
    if (new_node_id != 0)
        vram_search_around<kit_, schedule_>( //
            storage, query_span, query_state, new_node_id, view.end_id, visited, gathered, context, lane);
    else context.frontier.reset(lane);

    // The own row needs no lock: no reader arrives before the slot-0 release below flips the sentinel.
    u32_t const selected_count = vram_select_diverse<kit_>( //
        storage, cache, schedule, cache_norms, gathered.ids, selection, context, lane);
    vram_neighbors_ref<warp_t> const own_neighbors {storage.neighbors_for(new_node_id), storage.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    warp_t::sync();
    node_id_t const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) {
        own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
        // A tally for the host; the slot-0 release above is what publishes the row.
        warp_t::fetch_add_relaxed(*view.published_rows, 1u);
    }

    // Only now - published, no locks held - back-link the neighbors.
    vram_back_link_neighbors<kit_>( //
        storage, new_node_id, new_norms, selection, selected_count, cache, schedule, cache_norms, gathered,
        context.telemetry, lane);
}

/** Geometry & storage pointers one refinement launch operates on: the live storage it reads, and
 *  the shadow neighbors matrix it writes - same @b [rows,connectivity] shape, observed by no
 *  reader. */
template <vram_quads_kit kit_>
struct vram_refine_view {
    vram_flat_view<kit_> storage;

    /** Refined neighbors land here, @b [rows,connectivity] IDs - plain stores, published wholesale
     *  by the host after the launch drains. */
    u32_t* shadow_neighbors;

    /** IDs strictly below this bound are refined by this launch. */
    u32_t end_id;

    /** Bytes of the per-warp dynamic-shared slice - the insertion path's shape. */
    u32_t scratch_bytes_per_warp;

    /** The pass's base ID: the warps assign themselves offsets in @b [first_id,end_id). */
    node_id_t first_id;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice - see the build view's twin. */
    vram_arena_shape_t arena_shape;

    /** The arena frontier's expansion budget - see the build view's twin. */
    u32_t expansion;

    /** The context's monotone epoch floor at launch - see the build view's twin. */
    u32_t epoch_base;
    vram_telemetry_block_t* telemetry;

    /** Rows of the pass warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;

    /** Optional dense ID array to refine instead of the @b [first_id,end_id) range; null refines
     *  the range, and under an array @c end_id bounds its length and offsets index it. */
    u32_t const* node_ids = nullptr;

    /** Published-node ceiling for the walk's landmark sampling: always the graph size, never
     *  @c end_id, which under an ID array is its length. */
    u32_t sampling_bound = 0;

    /** The first row of the appended part, where the rows a shard absorbed begin: each node first
     *  walks across into the other part alone. Zero for a plain refine, which has no parts. */
    u32_t appended_base = 0;
};

/** Re-selects the neighbors of @p node_id: the node's own vector is the query, its current
 *  neighbors seed the walk so every one already wired competes in the re-selection, and the diverse
 *  selection lands in the shadow slots. Under an appended part the node first walks across into the
 *  other part from an empty pool, so that part's nearest compete too. Read-only on the live
 *  storage. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, typename frontier_>
__device__ __forceinline__ void vram_refine_one( //
    vram_refine_view<kit_> const& view, node_id_t node_id, vram_visited_set<typename kit_::warp_t>& visited,
    typename kit_::fragment_t* query_span, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_selection_scratch_t selection, typename kit_::norms_t (&cache_norms)[vram_cache_rows_k],
    vram_gathered<kit_>& gathered, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_flat_view<kit_> const& storage = view.storage;

    // The query is the node's own stored vector; its norms are already derived state.
    for (u32_t fragment = lane; fragment < storage.row_fragments; fragment += warp_t::width_k)
        query_span[fragment] = storage.vector_for(node_id, fragment);
    typename kit_::norms_t const query_state = storage.norm_for(node_id);
    warp_t::sync();

    // The node itself never joins its own candidate pool - visited before anything stages.
    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane);

    // The walk across runs first, with landmarks from the other part alone: the live graph links no
    // part to the other, so the walk stays there, and its nearest would lose to the own neighbors.
    // Both walks share one call site, so the walk inlines once and keeps a plain refine's
    // registers.
    bool const appended = node_id >= view.appended_base;
    node_id_t sampling_first = appended ? 0 : view.appended_base;
    node_id_t sampling_bound = appended ? view.appended_base : view.sampling_bound;
#pragma unroll 1
    for (u32_t walk_number = view.appended_base ? 0 : 1; walk_number != 2; ++walk_number) {
        if (walk_number) {
            context.frontier.renew();
            sampling_first = 0, sampling_bound = view.sampling_bound;
            // Seed the frontier with the current neighbors, staged like a traversal expansion minus
            // the hop bookkeeping.
            vram_neighbors_ref<warp_t> const own_neighbors {storage.neighbors_for(node_id), storage.connectivity};
            node_id_t first_value = node_id_unknown_k;
            if (lane == 0) first_value = own_neighbors.peek_first();
            first_value = warp_t::shuffle(first_value, 0);
            if (first_value != node_id_unknown_k && first_value != node_id)
                for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
                    u32_t const slot = slot_base + lane;
                    node_id_t const seed_id = slot == 0                     ? first_value
                                              : slot < storage.connectivity ? own_neighbors.load_slot(slot)
                                                                            : node_id_unknown_k;
                    bool const fresh = visited.test_and_set(seed_id != node_id ? seed_id : node_id_unknown_k, lane);
                    typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
                    u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
                    if (fresh) gathered.ids[ordinal] = seed_id;
                    warp_t::sync();
                    u32_t const gathered_count = warp_t::count(fresh_mask);
                    if (gathered_count)
                        vram_score_into_frontier<kit_, schedule_>( //
                            storage, query_span, query_state, gathered, gathered_count, context, lane, dummy_fold_t {});
                }
        }
        vram_search_ahead<kit_, schedule_>( //
            storage, query_span, query_state, sampling_first, sampling_bound, view.sampling_bound, visited, gathered,
            context, lane);
    }

    // Re-select the neighbors from the whole pool.
    u32_t const selected_count = vram_select_diverse<kit_>( //
        storage, cache, schedule, cache_norms, gathered.ids, selection, context, lane);

    // The insertion's slot convention: neighbors ascending, sentinel tail, a slot-0 self-loop meaning published-empty.
    u32_t* const shadow_slots = vram_row(view.shadow_neighbors, node_id, storage.connectivity);
    for (u32_t slot = lane; slot < storage.connectivity; slot += warp_t::width_k)
        shadow_slots[slot] = slot < selected_count ? selection.selected_ids[slot]
                             : slot == 0           ? node_id
                                                   : node_id_unknown_k;
    warp_t::sync();
}

/** Hands @p callback every neighbor of @p row the calling lane owns - its lane-strided slots, past
 *  sentinels, self-loops and IDs at or past @p rows_count. */
template <typename warp_type_, typename callback_type_>
__device__ __forceinline__ void vram_for_each_link(u32_t const* neighbors, u32_t connectivity, u32_t rows_count,
                                                   u32_t row, u32_t lane, callback_type_&& callback) noexcept {
    u32_t const* const slots = vram_row(neighbors, row, connectivity);
    for (u32_t slot = lane; slot < connectivity; slot += warp_type_::width_k)
        if (u32_t const target = slots[slot]; target != node_id_unknown_k && target != row && target < rows_count)
            callback(target);
}

/** Selects one node's final neighbors from its forward picks and its reverse bucket, the diversity
 *  heuristic over the union ascending by distance to the node, backfilled with the nearest pruned
 *  ones. The neighbors land in the live matrix with plain stores, since the pass owns the graph
 *  exclusively. The ascending scores and the kept cache positions lie past the cache's last row. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_>
__device__ __forceinline__ void vram_merge_back_links_one( //
    vram_refine_view<kit_> const& view, u32_t const* bucket_ids, u32_t const* bucket_counts, u32_t bucket_width,
    node_id_t node_id, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    typename kit_::norms_t (&cache_norms)[vram_cache_rows_k], vram_gathered<kit_>& gathered,
    vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_flat_view<kit_> const& storage = view.storage;
    vram_scored_node_t* const sorted_span = reinterpret_cast<vram_scored_node_t*>(cache.row(vram_cache_rows_k));
    u32_t* const kept_positions = reinterpret_cast<u32_t*>(sorted_span + vram_cache_rows_k);

    // The last cache row hosts the node's own vector, so only the rows before it host candidates.
    constexpr u32_t query_row_k = vram_cache_rows_k - 1;
    for (u32_t fragment = lane; fragment < storage.row_fragments; fragment += warp_t::width_k)
        cache.at(query_row_k, fragment) = storage.vector_for(node_id, fragment);
    typename kit_::norms_t const query_state = storage.norm_for(node_id);
    warp_t::sync();

    // Stage the forward picks; the shadow neighbors are this pass's own output, already distinct.
    u32_t candidates_count = 0;
    u32_t const* const shadow_slots = vram_row(view.shadow_neighbors, node_id, storage.connectivity);
    for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        u32_t const forward_id = slot < storage.connectivity ? shadow_slots[slot] : node_id_unknown_k;
        bool const usable = forward_id != node_id_unknown_k && forward_id != node_id;
        typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
        u32_t const ordinal = warp_t::count(usable_mask & warp_t::below(lane));
        if (usable) gathered.ids[candidates_count + ordinal] = forward_id;
        candidates_count += warp_t::count(usable_mask);
        warp_t::sync();
    }
    u32_t const forward_count = candidates_count;

    // Stage the reverse bucket, dropping IDs the forward prefix already holds - mutual picks land
    // in both - and clamping the union to the candidate cache.
    u32_t const landed = min_of(bucket_counts[node_id], bucket_width);
    u32_t const* const reverse_slots = vram_row(bucket_ids, node_id, bucket_width);
    for (u32_t slot_base = 0; slot_base < landed && candidates_count < query_row_k; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        u32_t const reverse_id = slot < landed ? reverse_slots[slot] : node_id_unknown_k;
        bool usable = reverse_id != node_id_unknown_k && reverse_id != node_id;
        for (u32_t forward_index = 0; usable && forward_index != forward_count; ++forward_index)
            usable = gathered.ids[forward_index] != reverse_id;
        typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
        u32_t const ordinal = warp_t::count(usable_mask & warp_t::below(lane));
        if (usable && candidates_count + ordinal < query_row_k) gathered.ids[candidates_count + ordinal] = reverse_id;
        candidates_count = min_of(candidates_count + warp_t::count(usable_mask), query_row_k);
        warp_t::sync();
    }

    // Score every candidate against the node & sort ascending; IDs are cache-row indices.
    telemetry.scored += candidates_count;
    vram_sorted_buffer<warp_t> sorted_candidates {sorted_span, 0, query_row_k};
    for (u32_t base = 0; base < candidates_count; base += warp_t::width_k) {
        u32_t const batch_count = min_of(candidates_count - base, warp_t::width_k);
        vram_scored_node_t const scored = vram_score_batch<kit_>( //
            storage, cache.row(query_row_k), query_state, gathered.ids + base, batch_count, gathered, lane);
        vram_scored_node_t const sorted_batch = sort_lanes<warp_t>(
            lane < batch_count ? vram_scored_node_t::make(scored.distance(), base + lane) // ID = candidate index.
                               : vram_scored_node_t::sentinel(),
            lane);
        sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
    }

    // Cache the candidate vectors & norms for the pairwise diversity checks, which visit the rows
    // by distance, so every row lands first.
    schedule.stage_rows(storage, cache, gathered.ids, candidates_count, lane);
    for (u32_t cache_row = lane; cache_row < candidates_count; cache_row += warp_t::width_k)
        cache_norms[cache_row] = storage.norm_for(gathered.ids[cache_row]);
    schedule.await_rows(candidates_count);
    warp_t::sync();

    // The greedy diversity walk, ascending by distance to the node.
    u32_t refined_count = 0;
    for (u32_t merge_index = 0; merge_index != sorted_candidates.size_ && refined_count != storage.connectivity;
         ++merge_index) {
        vram_scored_node_t const entry = sorted_candidates.at(merge_index);
        f32_t const to_query = entry.distance();
        u32_t const candidate_position = entry.id();
        bool diverse = true;
        for (u32_t base = 0; base < refined_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < refined_count) {
                u32_t const kept_position = kept_positions[base + lane];
                typename kit_::dot_t pair_accumulator {};
#pragma unroll 4
                for (u32_t fragment = 0; fragment != storage.row_fragments; ++fragment)
                    pair_accumulator = kit_::accumulate_dot(cache.at(candidate_position, fragment),
                                                            cache.at(kept_position, fragment), pair_accumulator);
                f32_t const between = kit_::distance(storage.kernel, pair_accumulator, cache_norms[candidate_position],
                                                     cache_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++telemetry.scored;
            }
            diverse = warp_t::all(lane_diverse);
        }
        if (diverse) {
            if (lane == 0) kept_positions[refined_count] = candidate_position;
            ++refined_count;
            warp_t::sync();
        }
    }

    // Fill the remaining slots with the nearest pruned candidates. The diverse prefix is a
    // subsequence of the sorted candidates, so one cursor tells a kept candidate from a rejected
    // one.
    u32_t const diverse_count = refined_count;
    for (u32_t merge_index = 0, matched = 0;
         merge_index != sorted_candidates.size_ && refined_count != storage.connectivity; ++merge_index) {
        u32_t const candidate_position = sorted_candidates.at(merge_index).id();
        if (matched != diverse_count && kept_positions[matched] == candidate_position) {
            ++matched;
            continue;
        }
        if (lane == 0) kept_positions[refined_count] = candidate_position;
        ++refined_count;
        warp_t::sync();
    }

    // Land the final neighbors in the live matrix - the insertion's slot convention. The pass owns
    // the graph, so a slot-wise comparison is the settling signal.
    warp_t::sync();
    u32_t* const live_slots = storage.neighbors_for(node_id);
    typename warp_t::mask_t changed_mask = 0;
    for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) { // Warp-uniform.
        u32_t const slot = slot_base + lane;
        u32_t const value = slot < refined_count ? gathered.ids[kept_positions[slot]]
                            : slot == 0          ? node_id
                                                 : node_id_unknown_k;
        changed_mask |= warp_t::ballot(slot < storage.connectivity && live_slots[slot] != value);
        if (slot < storage.connectivity) live_slots[slot] = value;
    }
    telemetry.changed_nodes += changed_mask != 0;
    warp_t::sync();
}

/** Writes each query's nearest @c count IDs and distances out of a search walk, ascending, with
 *  sentinel IDs past the live prefix. */
template <vram_quads_kit kit_>
struct vram_search_sink {
    using fold_t = dummy_fold_t;

    /** The landing for the matched IDs, @b [queries,count]. */
    node_id_t* result_ids;

    /** The landing for the matched distances, @b [queries,count]; null skips the writes. */
    distance_t* result_distances;

    /** Matches kept per query. */
    u32_t count;

    /** The metric the ranked distances convert out of. */
    nk_kernel_kind_t kernel;

    template <typename frontier_>
    __device__ __forceinline__ fold_t open(u32_t, vram_warp_context<frontier_> const&, u32_t) const noexcept {
        return {};
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const&, vram_warp_context<frontier_>& context, u32_t query_index,
                                          u32_t lane) const noexcept {
        // The rank extraction shuffles - every lane participates, only in-range ones store.
        node_id_t* const ids_row = vram_row(result_ids, query_index, count);
        distance_t* const distances_row = result_distances ? vram_row(result_distances, query_index, count) : nullptr;
        context.frontier.for_each_rank(count, lane, [&](u32_t rank, vram_scored_node_t entry) noexcept {
            if (rank >= count) return;
            ids_row[rank] = entry.id();
            if (distances_row) distances_row[rank] = kit_::convert_to_distance(kernel, entry.distance());
        });
    }
};

/** Writes each query's kept node IDs and native dots out of a multiply walk, and the pairs it scored. */
template <vram_quads_kit kit_>
struct vram_multiply_sink {
    using fold_t = vram_multiply_fold<kit_>;

    /** Node IDs of the kept pairs, @b [queries,wanted], sentinel past them. */
    node_id_t* ids;

    /** Native dots of the kept pairs, parallel to @c ids. */
    typename kit_::dot_t* products;

    /** Pairs scored per query, above @c wanted when a row was truncated. */
    u32_t* counts;

    /** Pairs kept per query. */
    u32_t wanted;

    template <typename frontier_>
    __device__ __forceinline__ fold_t open(u32_t query_index, vram_warp_context<frontier_> const&,
                                           u32_t) const noexcept {
        return {vram_row(ids, query_index, wanted), vram_row(products, query_index, wanted), wanted};
    }

    template <typename frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, vram_warp_context<frontier_>&, u32_t query_index,
                                          u32_t lane) const noexcept {
        for (u32_t slot = fold.written + lane; slot < wanted; slot += kit_::warp_t::width_k)
            fold.ids[slot] = node_id_unknown_k;
        if (lane == 0) counts[query_index] = fold.scored;
    }
};

/** One query's full walk: stages the query into its shared span, walks the graph under the fold the
 *  sink opens and the query's slice of @p rescorings, and lets the sink close the query. Safe against
 *  concurrent insertions: the walk gates on every slot-0 word and never writes the graph. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, typename sink_,
          typename rescoring_, typename frontier_>
__device__ __forceinline__ void vram_walk_one( //
    vram_search_view<kit_> const& view, sink_ const& sink, rescoring_ const& rescorings, u32_t query_index,
    typename kit_::fragment_t* query_span, vram_visited_set<typename kit_::warp_t>& visited, frontier_ frontier,
    vram_gathered<kit_>& gathered, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    typename kit_::fragment_t const* query_row = vram_row(view.queries, query_index, view.storage.row_fragments);
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < view.storage.row_fragments; fragment += warp_t::width_k) {
        typename kit_::fragment_t const value = query_row[fragment];
        query_span[fragment] = value;
        norms_accumulator = kit_::accumulate_norms(value, norms_accumulator);
    }
    norms_accumulator = vram_reduce_norms<warp_t>(norms_accumulator);
    norms_accumulator = vram_broadcast_norms<warp_t>(norms_accumulator);
    typename kit_::norms_t const query_state = kit_::partials_to_norms(norms_accumulator);
    warp_t::sync();

    vram_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = vram_lane_generator<warp_t>(query_index, lane);
    context.scratch = reinterpret_cast<byte_t*>(query_span + view.storage.row_fragments);
    typename sink_::fold_t fold = sink.open(query_index, context, lane);
    rescoring_ const rescoring = rescorings.for_query(query_index, context);
    vram_search_around<kit_, schedule_, resident_blocks_>( //
        view.storage, query_span, query_state, view.sampling_bound, view.sampling_bound, visited, gathered, context,
        lane, fold, rescoring);
    sink.close(fold, context, query_index, lane);
    context.telemetry.tail_drops = context.frontier.dropped();
    context.telemetry.visited_saturations = visited.saturations_;
    context.telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

/** Moves rows @b [first_row,first_row+rows_count) of @p neighbors between ID spaces as the host's
 *  @c rebase_neighbors and @c compact_neighbors do, one warp per row: an ID inside
 *  @b [source_first,source_bound) lands at @p shift past its offset, the rest fold away, the
 *  survivors pack to the front, and a row nothing survived in points at itself. */
template <typename warp_type_>
__device__ __forceinline__ void vram_rebase_links(u32_t* neighbors, u32_t first_row, u32_t rows_count,
                                                  u32_t connectivity, u32_t source_first, u32_t source_bound,
                                                  u32_t shift) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = first_row + warp.index; row < first_row + rows_count; row += warp.count) {
        u32_t* const slots = vram_row(neighbors, row, connectivity);
        u32_t kept = 0;
        for (u32_t base = 0; base < connectivity; base += warp_type_::width_k) { // Warp-uniform trip count.
            u32_t const slot = base + warp.lane;
            u32_t const id = slot < connectivity ? slots[slot] : node_id_unknown_k;
            bool const inside = source_first <= id && id < source_bound;
            // Every lane read its slot before the vote, and a survivor only moves down.
            typename warp_type_::mask_t const survivors = warp_type_::ballot(inside);
            if (inside)
                slots[kept + warp_type_::count(survivors & warp_type_::below(warp.lane))] = id - source_first + shift;
            kept += warp_type_::count(survivors);
        }
        for (u32_t slot = kept + warp.lane; slot < connectivity; slot += warp_type_::width_k)
            slots[slot] = node_id_unknown_k;
        if (!kept && warp.lane == 0) slots[0] = row;
    }
}

/** A ranking of opaque keys as a sharded search lands it on a device: @c width keys of @c key_bytes
 *  bytes and as many scores, f64 or f32 by @c score_bytes, per query, rows @c keys_stride and
 *  @c scores_stride bytes apart, and each query's count of ranks held. */
struct vram_ranking_view {
    byte_t* keys;
    byte_t* scores;
    std::size_t* counts;
    std::size_t keys_stride, scores_stride;
    u32_t width, key_bytes, score_bytes;

    __device__ __forceinline__ byte_t* key(u32_t query, u32_t rank) const noexcept {
        return keys + query * keys_stride + std::size_t {rank} * key_bytes;
    }

    __device__ __forceinline__ f64_t score(u32_t query, u32_t rank) const noexcept {
        byte_t const* const row = scores + query * scores_stride;
        f64_t const* const wide_row = reinterpret_cast<f64_t const*>(row);
        f32_t const* const narrow_row = reinterpret_cast<f32_t const*>(row);
        return score_bytes == sizeof(f64_t) ? wide_row[rank] : narrow_row[rank];
    }

    /** Lands @p key and @p score at @p rank of @p query; an f32 row narrows exactly, since every
     *  score began as an f32 distance. */
    __device__ __forceinline__ void store(u32_t query, u32_t rank, byte_t const* key_source,
                                          f64_t score_value) const noexcept {
        // A loop, since `std::memcpy` names the host function under HIP.
        byte_t* const key_target = key(query, rank);
        for (u32_t offset = 0; offset != key_bytes; ++offset) key_target[offset] = key_source[offset];
        byte_t* const row = scores + query * scores_stride;
        f64_t* const wide_row = reinterpret_cast<f64_t*>(row);
        f32_t* const narrow_row = reinterpret_cast<f32_t*>(row);
        if (score_bytes == sizeof(f64_t)) wide_row[rank] = score_value;
        else narrow_row[rank] = static_cast<f32_t>(score_value);
    }

    /** The same ranking @p bytes further on, where the next shard's gathered block starts. */
    __device__ __forceinline__ vram_ranking_view shifted(std::size_t bytes) const noexcept {
        vram_ranking_view next = *this;
        next.keys += bytes, next.scores += bytes;
        next.counts = reinterpret_cast<std::size_t*>(reinterpret_cast<byte_t*>(counts) + bytes);
        return next;
    }
};

/** Whether a rank counts the values equal to the probed one, or only the ones below it. */
enum class vram_ties_t : bool { excluded_k, included_k };

/** How many of the ascending @p count values @p value_of yields lie below @p value, or at most at
 *  it when @p ties are included: one binary search. */
template <typename value_type_, typename value_of_type_>
__device__ __forceinline__ u32_t vram_rank_below(u32_t count, value_type_ value, vram_ties_t ties,
                                                 value_of_type_&& value_of) noexcept {
    u32_t first = 0;
    while (count) {
        u32_t const half = count / 2;
        value_type_ const probe = value_of(first + half);
        if (ties == vram_ties_t::included_k ? probe <= value : probe < value) first += half + 1, count -= half + 1;
        else count = half;
    }
    return first;
}

/**
 *  @brief Resolves one shard's walked matches into @p ranking, one warp per query: each live
 *      match's key through the shard's device @p keys column, its distance as its score, and the
 *      query's count.
 *
 *  A query's matches, @p walked per row at @p ids and @p distances, end at its first sentinel ID.
 *  Ranks follow @c merge_search_results filling an empty ranking, which places each match before
 *  the equal ones already held, so a run of equal distances lands reversed; ranks past the width
 *  drop.
 */
template <typename warp_type_>
__device__ __forceinline__ void vram_resolve_matches(node_id_t const* ids, distance_t const* distances, u32_t walked,
                                                     byte_t const* keys, u32_t queries_count,
                                                     vram_ranking_view const& ranking) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t query = warp.index; query < queries_count; query += warp.count) {
        node_id_t const* const query_ids = vram_row(ids, query, walked);
        distance_t const* const query_distances = vram_row(distances, query, walked);
        u32_t live = walked;
        for (u32_t base = 0; base < walked; base += warp_type_::width_k) { // Warp-uniform bounds for the vote.
            u32_t const slot = base + warp.lane;
            typename warp_type_::mask_t const ended = warp_type_::ballot(slot >= walked ||
                                                                         query_ids[slot] == node_id_unknown_k);
            if (ended) {
                live = base + warp_type_::first(ended);
                break;
            }
        }
        auto const distance_of = [&](u32_t position) noexcept { return f64_t {query_distances[position]}; };
        for (u32_t position = warp.lane; position < live; position += warp_type_::width_k) {
            f64_t const distance = query_distances[position];
            u32_t const rank = vram_rank_below(live, distance, vram_ties_t::excluded_k, distance_of) +
                               vram_rank_below(live, distance, vram_ties_t::included_k, distance_of) - 1 - position;
            if (rank < ranking.width)
                ranking.store(query, rank, keys + std::size_t {query_ids[position]} * ranking.key_bytes, distance);
        }
        if (warp.lane == 0) ranking.counts[query] = min_of(live, ranking.width);
    }
}

/**
 *  @brief Merges the @p shards rankings gathered @p block_bytes apart from @p gathered into
 *      @p merged, one warp per query, exactly as @c merge_search_results folds them in shard
 *      order, without sorting.
 *
 *  An entry's merged rank is its own plus the entries every other shard ranks ahead of it: an
 *  earlier shard's below its score, and a later shard's at most at it, since a later fold places
 *  its entries before the equal ones already held. Ranks past the width drop. The host gather folds
 *  shards as they finish rather than in shard order, so where distances tie across shards the two
 *  paths may order, and at the width keep, different keys over the same distances.
 */
template <typename warp_type_>
__device__ __forceinline__ void vram_merge_matches(vram_ranking_view const& gathered, std::size_t block_bytes,
                                                   u32_t shards, u32_t queries_count,
                                                   vram_ranking_view const& merged) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t query = warp.index; query < queries_count; query += warp.count) {
        std::size_t held = 0;
        for (u32_t shard = 0; shard != shards; ++shard) {
            vram_ranking_view const own = gathered.shifted(shard * block_bytes);
            u32_t const own_count = static_cast<u32_t>(own.counts[query]);
            held += own_count;
            for (u32_t position = warp.lane; position < own_count; position += warp_type_::width_k) {
                f64_t const score = own.score(query, position);
                u32_t rank = position;
                for (u32_t other = 0; other != shards && rank < merged.width; ++other) {
                    if (other == shard) continue;
                    vram_ranking_view const theirs = gathered.shifted(other * block_bytes);
                    vram_ties_t const ties = other > shard ? vram_ties_t::included_k : vram_ties_t::excluded_k;
                    rank += vram_rank_below(static_cast<u32_t>(theirs.counts[query]), score, ties,
                                            [&](u32_t rank_there) noexcept { return theirs.score(query, rank_there); });
                }
                if (rank < merged.width) merged.store(query, rank, own.key(query, position), score);
            }
        }
        if (warp.lane == 0) merged.counts[query] = min_of<std::size_t>(held, merged.width);
    }
}

} // namespace unum::usearch::dense
