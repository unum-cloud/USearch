/**
 *  @file include/usearch/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The device bodies every dense VRAM index compiled by NVCC or HIP-Clang shares: the views,
 *      the greedy walk, the diversity selection, the back-linker, insertion, refinement, the fold
 *      of a compaction's shards, the attention fold, the reachability audits, the sharded search's
 *      rankings, and a split's partition and scatter.
 *
 *  Every body here is templated on a quads kit and its vendor's warp policy, or on that warp policy
 *  alone; see @c vram_queue.cuh for what a warp policy provides. Walks and builders also take a
 *  schedule, which names the warp policy as @c schedule_::warp_t, scores a walk's batch and moves
 *  candidate rows into the warp's cache; every other body takes the policy beside the kit. The
 *  views are read off the plain records of @c vram_index.hpp. The kernels that launch these bodies,
 *  their launch bounds and the exact scan stay in each vendor's own @c vram_<vendor>_index file,
 *  beside the kits in @c vram_<vendor>_metric - the split @c utilities/vram_device.hpp keeps beside
 *  its vendor twins.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - a lane's row start crosses the warp as an integer

#include <limits>      // `std::numeric_limits` - the running maximum's identity
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::conditional_t` - a builder's frontier by its rung

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the views carry the metric
#include <numkong/types.h>        // `NUMKONG_ARCH_CUDA_` & `NUMKONG_ARCH_ROCM_`, which the cast below tests
#include <numkong/cast/simt.cuh>  // `nk_cast_unit_simt_` - a gathered row's cast

#include "usearch/utilities/types.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/utilities/vram_queue.cuh"
#include "usearch/dense/vram_index.hpp" // the records the views are read off

namespace unum::usearch::dense {

/** What a vendor's quads kit binds: one storage dtype on one GPU generation. */
template <typename kit_>
concept vram_quads_kit = requires(typename kit_::fragment_t fragment, typename kit_::dot_t dot,
                                  typename kit_::weighted_t weighted,
                                  typename kit_::norms_accumulator_t norms_accumulator, typename kit_::norms_t norms,
                                  distance_t ranked, nk_kernel_kind_t kernel) {
    typename kit_::scalar_t;
    requires std::is_convertible_v<decltype(kit_::capability_k), nk_capability_t>;
    { kit_::accumulate_dot(fragment, fragment, dot) } -> std::same_as<typename kit_::dot_t>;
    { kit_::accumulate_weighted(fragment, f32_t {}, weighted) } -> std::same_as<typename kit_::weighted_t>;
    { kit_::accumulate_norms(fragment, norms_accumulator) } -> std::same_as<typename kit_::norms_accumulator_t>;
    { kit_::partials_to_norms(norms_accumulator) } -> std::same_as<typename kit_::norms_t>;
    { kit_::distance(kernel, dot, norms, norms) } -> std::same_as<distance_t>;
    { kit_::convert_to_distance(kernel, ranked) } -> std::same_as<distance_t>;
};

/** Runtime-strided view over the warp's vector cache - rows of @c row_fragments_ fragments each,
 *  aliasing the scratch slice, in shared memory or the warp's arena slice. In the arena the cache
 *  is by reference: a staged row is the storage row its ID names, read in place, not a copy. */
template <typename fragment_type_>
struct vram_vector_cache {
    fragment_type_* data_;
    u32_t row_fragments_;

    /** The staged IDs a by-reference cache resolves its rows through, or null when it copies
     *  them. */
    u32_t* ids_ = nullptr;

    __device__ __forceinline__ fragment_type_* row(u32_t row_index) const noexcept {
        return data_ + row_index * row_fragments_;
    }
    __device__ __forceinline__ fragment_type_& at(u32_t row_index, u32_t fragment_index) const noexcept {
        return data_[row_index * row_fragments_ + fragment_index];
    }

    /** Row @p row_index as a staging or @ref place_row left it. */
    template <typename storage_>
    __device__ __forceinline__ fragment_type_ const* staged_row(storage_ const& storage,
                                                                u32_t row_index) const noexcept {
        return ids_ ? storage.vector_span_for(ids_[row_index]) : row(row_index);
    }

    /** Puts the storage row of @p node_id at @p row_index for one warp, which syncs before reading
     *  it. */
    template <typename warp_, typename storage_>
    __device__ __forceinline__ void place_row(storage_ const& storage, u32_t row_index, u32_t node_id,
                                              u32_t lane) const noexcept {
        if (ids_) {
            if (lane == 0) ids_[row_index] = node_id;
            return;
        }
        for (u32_t fragment = lane; fragment < row_fragments_; fragment += warp_::width_k)
            at(row_index, fragment) = storage.vector_for(node_id, fragment);
    }
};

/** Carves the re-selection spans of @p rows entries each past the last of @p cache's @p rows
 *  rows. */
template <typename node_id_type_, typename fragment_type_>
__device__ __forceinline__ vram_selection_scratch<node_id_type_> carve_selection_scratch(
    vram_vector_cache<fragment_type_> cache, u32_t rows) noexcept {
    return vram_selection_scratch<node_id_type_>::carve(reinterpret_cast<byte_t*>(cache.row(rows)), rows);
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

/** How every vector row a view reads is laid out and scored: the fragment stride, the lanes and
 *  steps covering one row, and the metric. */
struct vram_row_geometry_t {

    /** Fragments per vector row - the stride of the vectors; the row's byte length is the dimension
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

    /** The staged row holding flattened @p word, exact for every word under 2^31, so for any cache
     *  a slice under 4 GB fits. */
    __device__ __forceinline__ u32_t row_of_word(u32_t word) const noexcept {
        u32_t const row = static_cast<u32_t>((static_cast<u64_t>(word) * row_fragments_magic) >> 31);
        return row - (row * row_fragments > word);
    }
    __device__ __forceinline__ u32_t fragment_of_word(u32_t word, u32_t row) const noexcept {
        return word - row * row_fragments;
    }
};

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

    /** Neighbors matrix, @b [rows,connectivity] IDs; slot 0 holds the sentinel until the row
     *  publishes. */
    u32_t* neighbors;

    /** Neighbor slots per node: the stride of @c neighbors and the connectivity. */
    u32_t connectivity;

    vram_row_geometry_t geometry;

    /** Rows physically resident on the executing device, feeding the locality telemetry; both zero
     *  when the view spans one device. */
    u32_t resident_begin = 0, resident_end = 0;

    __device__ __forceinline__ fragment_t vector_for(node_id_t node_id, u32_t fragment_index) const noexcept {
        fragment_t const* const row = vector_span_for(node_id);
        return row[fragment_index];
    }
    __device__ __forceinline__ fragment_t* vector_span_for(node_id_t node_id) const noexcept {
        return vram_row(vectors, node_id, geometry.row_fragments);
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

/** The shards of one fold bin as one storage: a bin ID keeps its shard's place in the bin from bit
 *  @c row_bits up and its row below, so one shift finds the shard, whose slabs every device of the
 *  bin maps. The neighbors are the fold's own rows, never a shard's live ones. */
template <vram_quads_kit kit_>
struct vram_bin_view {
    using fragment_t = typename kit_::fragment_t;
    using norms_t = typename kit_::norms_t;

    /** Every shard of the bin by its place, resident on the executing device. */
    vram_bin_shard_t const* shards;

    /** Neighbor slots per node, as in @ref vram_flat_view. */
    u32_t connectivity;

    /** The geometry every shard of the bin shares. */
    vram_row_geometry_t geometry;

    /** The low bits of a bin ID, its row in its shard, and their mask; a bin holds two shards at
     *  least, so never all 32. */
    u32_t row_bits;
    u32_t row_mask;

    __device__ __forceinline__ u32_t shard_of(node_id_t node_id) const noexcept { return node_id >> row_bits; }
    __device__ __forceinline__ u32_t row_of(node_id_t node_id) const noexcept { return node_id & row_mask; }

    __device__ __forceinline__ fragment_t vector_for(node_id_t node_id, u32_t fragment_index) const noexcept {
        fragment_t const* const row = vector_span_for(node_id);
        return row[fragment_index];
    }
    __device__ __forceinline__ fragment_t* vector_span_for(node_id_t node_id) const noexcept {
        return vram_row(reinterpret_cast<fragment_t*>(shards[shard_of(node_id)].vectors), row_of(node_id),
                        geometry.row_fragments);
    }

    /** A row's norms, or zeros where the shards keep none - every shard alike. */
    __device__ __forceinline__ norms_t norm_for(node_id_t node_id) const noexcept {
        norms_t const* const norms = reinterpret_cast<norms_t const*>(shards[shard_of(node_id)].norms);
        return norms ? norms[row_of(node_id)] : norms_t {};
    }
    __device__ __forceinline__ u32_t* neighbors_for(node_id_t node_id) const noexcept {
        return vram_row(shards[shard_of(node_id)].neighbors, row_of(node_id), connectivity);
    }
};

/** One warp's staged batch: candidate IDs, then the scored nodes and native dots of up to a
 *  lane-width of them. */
template <vram_quads_kit kit_, typename warp_type_>
struct vram_gathered {

    /** Candidate node IDs of a round's staging, two warp-widths: the scored batch, then the carry
     *  toward the next one; a builder caches its selection's in its own scratch. */
    u32_t ids[2 * warp_type_::width_k];

    /** The scored nodes of one batch, by batch row. */
    vram_scored_node<node_id_t> scored[warp_type_::width_k];

    /** The native dots of one batch, parallel to @c scored. */
    typename kit_::dot_t dots[warp_type_::width_k];
};

/** Graph products of one query row as a device fold: each scored batch appends its node IDs and
 *  native dots to the row until it fills, while @c scored keeps counting past it. */
template <vram_quads_kit kit_, typename warp_type_>
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

    __device__ __forceinline__ walk_demand_t absorb(vram_gathered<kit_, warp_type_> const& gathered,
                                                    u32_t gathered_count, u32_t lane) noexcept {
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
 *  @brief Scores the batch @p batch_ids points at, writing each row's scored node and native dot to
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
template <vram_quads_kit kit_, typename warp_type_, u32_t resident_blocks_, u32_t fragments_per_load_ = 1,
          typename storage_>
__device__ __forceinline__ vram_scored_node<node_id_t> vram_score_batch( //
    storage_ const& storage, typename kit_::fragment_t const* query_span, typename kit_::norms_t query_state,
    u32_t const* batch_ids, u32_t gathered_count, vram_gathered<kit_, warp_type_>& gathered, u32_t lane) noexcept {

    using warp_t = warp_type_;
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;
    if constexpr (fragments_per_load_ != 1)
        if (storage.geometry.row_fragments % fragments_per_load_ != 0)
            return vram_score_batch<kit_, warp_t, resident_blocks_>(storage, query_span, query_state, batch_ids,
                                                                    gathered_count, gathered, lane);

    // Aligned to its own size, so the compiler issues each lane's load as one request.
    struct alignas(sizeof(fragment_t) * fragments_per_load_) load_t {
        fragment_t fragments[fragments_per_load_];
    };
    // 128 concurrent random gathers of a fragment saturate the memory path, so the window is 128
    // fragments over the resident warps, which a wider load fills with fewer gathers.
    constexpr u32_t gathers_in_flight_k =
        min_of(vram_max_gathers_in_flight_k, 128u / (vram_warps_per_block_k * resident_blocks_)) / fragments_per_load_;
    u32_t const lanes_per_row = storage.geometry.lanes_per_row >= storage.geometry.row_fragments
                                    ? storage.geometry.lanes_per_row / fragments_per_load_
                                    : storage.geometry.lanes_per_row;
    u32_t const steps_per_row = divide_round_up(storage.geometry.steps_per_row, fragments_per_load_);
    u32_t const row_loads = storage.geometry.row_fragments / fragments_per_load_;
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
    if (!live) return vram_scored_node<node_id_t>::sentinel();
    vram_scored_node<node_id_t> const scored = vram_scored_node<node_id_t>::make(
        kit_::distance(storage.geometry.kernel, gathered.dots[lane], query_state, target_state), batch_ids[lane]);
    gathered.scored[lane] = scored;
    return scored;
}

/** What a vendor's schedule binds for one kit: how a walk scores its gathered batch, and how one
 *  warp's candidate rows reach its vector cache in shared memory, a by-reference cache staging
 *  none, apart from the kit's arithmetic. @c score_batch writes what
 *  @ref unum::usearch::dense::vram_score_batch writes for the batch's IDs, its load window sized
 *  for @c resident_blocks_ blocks per multiprocessor. @c stage_rows queues the rows of an ID span
 *  into the cache's leading rows over @c stages_k stages, and @c await_rows waits for at least that
 *  many leading rows for every lane; every staged row is in place before the next staging.
 *  @c keep_diverse runs the greedy diversity prefix @ref unum::usearch::dense::vram_keep_diverse
 *  runs, on the kit's tensor-core tiles where the vendor has them. A schedule is one warp's state
 *  over its @c shared_t in static shared memory, picked by the class like the kit. */
template <typename schedule_, typename kit_>
concept vram_schedule = requires(
    schedule_& schedule, u32_t rows, vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, vram_gathered<kit_, typename schedule_::warp_t>& gathered, u32_t lane,
    vram_vector_cache<typename kit_::fragment_t> cache,
    vram_sorted_buffer<typename schedule_::warp_t, node_id_t> const& sorted,
    vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms, bool& kept_zero,
    vram_telemetry_t& telemetry) {
    typename schedule_::shared_t;
    typename schedule_::warp_t;
    requires std::is_convertible_v<decltype(schedule_::stages_k), u32_t>;
    schedule.await_rows(rows);
    {
        schedule_::template score_batch<kit_, 1>(storage, query_span, query_state, gathered, rows, lane)
    } -> std::same_as<vram_scored_node<node_id_t>>;
    {
        schedule.template keep_diverse<kit_>(storage, cache, sorted, rows, rows, selection, candidate_norms, kept_zero,
                                             telemetry, lane)
    } -> std::same_as<u32_t>;
};

/** The schedule cascade, the twin of @c quads_ladder: each vendor specializes it under a constraint
 *  on @p caps_, naming the picked @c schedule_t and its @c picked_k. Schedules state their
 *  @c capability_k and @c compiled_k as kits do, and the first both admit wins. */
template <nk_capability_t caps_>
struct schedule_ladder;

/** The plain walk's rescoring: none, so every batch stays as the kit scored it. */
struct vram_no_rescoring {

    template <vram_frontier frontier_>
    __device__ __forceinline__ vram_no_rescoring for_query(u32_t, vram_warp_context<frontier_> const&) const noexcept {
        return {};
    }
};

/** The insertion and refine walks' rescoring: scores stay as the kit scored them, but a
 *  zero-distance lane enters only while @c frontier 's nearest holds no zero, and only the lowest
 *  such lane of its batch, so a duplicate class spends one rank of the pool instead of all. */
template <vram_frontier frontier_>
struct vram_single_zero_rescoring {
    frontier_ const& frontier;

    template <u32_t resident_blocks_, vram_quads_kit kit_, typename warp_type_>
    __device__ __forceinline__ vram_scored_node<node_id_t> rescore(vram_flat_view<kit_> const&,
                                                                   vram_gathered<kit_, warp_type_>&, u32_t, u32_t lane,
                                                                   vram_scored_node<node_id_t> scored) const noexcept {
        bool const zero = scored.distance() == 0;
        typename warp_type_::mask_t const zeros = warp_type_::ballot(zero);
        bool const held = frontier.nearest().distance() == 0;
        return zero && (held || lane != warp_type_::first(zeros)) ? vram_scored_node<node_id_t>::sentinel() : scored;
    }
};

/** Indexer heads as a device rescoring: each scored dot becomes the score Σₕ wₕ · ReLU(qₕ · k) and its distance
 *  1 − score, the warp scoring the batch once per head exactly as it scored the navigating query. */
template <vram_quads_kit kit_, typename warp_type_>
struct vram_indexer_rescoring {
    using warp_t = warp_type_;
    using fragment_t = typename kit_::fragment_t;

    /** Every query's heads as staged rows, @b [queries·heads,row_fragments], device-resident. */
    fragment_t const* heads;

    /** The signed head weights, @b [queries,heads], device-resident. */
    f32_t const* weights;

    /** Heads per query. */
    u32_t heads_count;

    /** Fragments per staged head row. */
    u32_t row_fragments;

    template <vram_frontier frontier_>
    __device__ __forceinline__ vram_indexer_rescoring for_query(u32_t query_index,
                                                                vram_warp_context<frontier_> const&) const noexcept {
        vram_indexer_rescoring narrowed = *this;
        narrowed.heads = vram_row(heads, query_index, heads_count * row_fragments);
        narrowed.weights = vram_row(weights, query_index, heads_count);
        return narrowed;
    }

    template <u32_t resident_blocks_>
    __device__ __forceinline__ vram_scored_node<node_id_t> rescore(vram_flat_view<kit_> const& storage,
                                                                   vram_gathered<kit_, warp_t>& gathered,
                                                                   u32_t gathered_count, u32_t lane,
                                                                   vram_scored_node<node_id_t> scored) const noexcept {
        bool const live = lane < gathered_count;
        f32_t indexed = 0;
        for (u32_t head = 0; head != heads_count; ++head) {
            vram_score_batch<kit_, warp_t, resident_blocks_>(storage, heads + head * row_fragments,
                                                             typename kit_::norms_t {}, gathered.ids, gathered_count,
                                                             gathered, lane);
            if (live) indexed += weights[head] * fmaxf(static_cast<f32_t>(gathered.dots[lane]), 0.f);
            // The next head's scoring overwrites the dots, so every lane reads its own first.
            warp_t::sync();
        }
        if (!live) return scored;
        gathered.dots[lane] = indexed;
        scored = vram_scored_node<node_id_t>::make(1 - indexed, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/** Scores one gathered batch against the query through @p schedule_ under @p resident_blocks_
 *  blocks per multiprocessor, lets @p rescoring rewrite the scores under the same count, folds it
 *  into the candidates, and hands it to @p fold - whose answer says whether the walk goes on. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, vram_frontier frontier_,
          typename fold_, typename rescoring_ = vram_no_rescoring>
__device__ __forceinline__ walk_demand_t vram_score_into_frontier( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, vram_gathered<kit_, typename schedule_::warp_t>& gathered, u32_t gathered_count,
    vram_warp_context<frontier_>& context, u32_t lane, fold_&& fold,
    rescoring_ const& rescoring = rescoring_ {}) noexcept {
    using warp_t = typename schedule_::warp_t;
    context.telemetry.scored += gathered_count;
    if (storage.resident_end) {
        bool const counted = lane < gathered_count;
        u32_t const candidate = counted ? gathered.ids[lane] : 0;
        typename warp_t::mask_t const local_mask = warp_t::ballot(counted && candidate >= storage.resident_begin &&
                                                                  candidate < storage.resident_end);
        context.telemetry.local_scored += warp_t::count(local_mask);
        context.telemetry.remote_scored += gathered_count - warp_t::count(local_mask);
    }
    vram_scored_node<node_id_t> staged = schedule_::template score_batch<kit_, resident_blocks_>( //
        storage, query_span, query_state, gathered, gathered_count, lane);
    // Even a call that changes nothing reorders the register-capped wide walk, so the plain walk makes none.
    if constexpr (!std::is_same_v<rescoring_, vram_no_rescoring>)
        staged = rescoring.template rescore<resident_blocks_>(storage, gathered, gathered_count, lane, staged);
    // No dedup: the exact visited set upstream guarantees every ID stages once.
    context.frontier.insert_batch(staged, lane);
    return fold.absorb(gathered, gathered_count, lane);
}

/**
 *  @brief Samples one lane-wide batch of distinct, published landmarks in
 *      @b [sampling_first,sampling_bound) into @p candidate_ids.
 *
 *  The draws are uniform and gated on the slot-0 publication word: a lane meeting an unpublished
 *  row draws again, which ends since a wave admits at most a share of the rows below it, so a draw
 *  is published with probability at least a half, and a lane meeting a visited row steps forward,
 *  for one lap of the range at most.
 *
 *  @param[in] storage The graph's rows.
 *  @param[in] sampling_first The first row of the range sampled.
 *  @param[in] sampling_bound One past the last row of the range sampled.
 *  @param[in,out] visited The rows this walk has scored already.
 *  @param[in] frontier The walk's frontier, asked whether it holds a draw once a saturated visited
 *      set re-scores it.
 *  @param[in,out] random_generator The draw state of this lane.
 *  @param[out] candidate_ids Receives the staged landmarks.
 *  @param[in] lane This lane's index in its warp.
 *  @return The number of staged landmarks; zero once every published row of the range is
 *      visited, or the visited set saturated.
 */
template <vram_quads_kit kit_, typename warp_type_, vram_frontier frontier_>
__device__ __forceinline__ u32_t vram_sample_landmarks( //
    vram_flat_view<kit_> const& storage, node_id_t sampling_first, node_id_t sampling_bound,
    vram_visited_set<warp_type_, node_id_t>& visited, frontier_ const& frontier, splitmix64_t& random_generator,
    u32_t (&candidate_ids)[2 * warp_type_::width_k], u32_t lane) noexcept {

    using warp_t = warp_type_;
    node_id_t const span = sampling_bound - sampling_first;
    uniform_below<node_id_t> const uniform {span};
    node_id_t draw = sampling_first + uniform(random_generator);
    node_id_t steps = 0;
    bool fresh = false;
    u32_t const saturations = visited.saturations_;
    // The trip count is warp-uniform, since every lane takes part in each collective claim, and the
    // lap ends at the set's first saturation, whose claims re-score.
    while (!warp_t::all(fresh || steps == span) && visited.saturations_ == saturations) {
        bool const searching = !fresh && steps != span;
        bool published = false;
        if (searching) {
            vram_neighbors_ref<warp_t, node_id_t> const row {storage.neighbors_for(draw), storage.connectivity};
            published = row.peek_first() != node_id_unknown_k;
        }
        // A saturating claim re-scores, so a draw the frontier holds stages no second copy; every
        // lane takes part in that collective scan.
        bool const drawn = visited.test_and_set(searching && published ? draw : node_id_unknown_k, lane);
        bool const claimed = !(visited.saturations_ != saturations && frontier.holds(draw, lane)) && drawn;
        // An unpublished row draws again; a visited one steps forward, one lap at most.
        bool const stepping = searching && !claimed && published;
        if (searching && !claimed && !published) draw = sampling_first + uniform(random_generator);
        if (stepping) draw = draw + 1 == sampling_bound ? sampling_first : draw + 1;
        steps += stepping;
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
 *  entries compete with them and must already be visited, and each call walks to convergence.
 *  Read-only on the graph: no locks taken, a transiently locked candidate held through one other
 *  expansion and dropped if still locked then, and no neighbor at or past @p first_invisible_id
 *  followed, since a batch's back-links splice its IDs into visible rows before the host turns its
 *  watermark. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, vram_frontier frontier_,
          typename fold_ = dummy_fold_t, typename rescoring_ = vram_no_rescoring>
__device__ __forceinline__ void vram_search_ahead( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, node_id_t sampling_first, node_id_t sampling_bound,
    node_id_t first_invisible_id, vram_visited_set<typename schedule_::warp_t, node_id_t>& visited,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_warp_context<frontier_>& context, u32_t lane,
    fold_&& fold = fold_ {}, rescoring_ const& rescoring = rescoring_ {}) noexcept {

    using warp_t = typename schedule_::warp_t;
    frontier_& frontier = context.frontier;
    vram_telemetry_t& telemetry = context.telemetry;

    // The one candidate held for its second look, and the hop it was held at; a drained frontier
    // moves that hop out of reach, so the next iteration looks at once.
    vram_scored_node<node_id_t> held = vram_scored_node<node_id_t>::sentinel();
    u32_t held_at_hop = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = vram_sample_landmarks<kit_, warp_t>( //
                storage, sampling_first, sampling_bound, visited, frontier, context.random_generator, gathered.ids,
                lane);
            if (!gathered_count) break; // Every published row of the range is already visited.
            if (vram_score_into_frontier<kit_, schedule_, resident_blocks_>( //
                    storage, query_span, query_state, gathered, gathered_count, context, lane, fold, rescoring) ==
                walk_demand_t::sated_k)
                return;
            seeding = false;
        }
        else {
            bool const revisiting = !(held == vram_scored_node<node_id_t>::sentinel()) && telemetry.hops != held_at_hop;
            vram_scored_node<node_id_t> const candidate = revisiting ? held : frontier.pop_nearest(lane);
            if (revisiting) held = vram_scored_node<node_id_t>::sentinel();
            if (candidate == vram_scored_node<node_id_t>::sentinel()) {
                if (!(held == vram_scored_node<node_id_t>::sentinel())) {
                    held_at_hop = ~0u;
                    continue;
                }
                // A drained frontier before an exhausted walk re-seeds, until no published row of
                // the range is left unvisited.
                if (frontier.exhausted()) break;
                seeding = true;
                ++telemetry.restarts;
                continue;
            }
            if (frontier.beyond_radius(candidate)) break;
            ++telemetry.hops;

            // Only the leader reads the racing lock word: per-lane reads could disagree on one
            // unlock, and a value-dependent `continue` would fork the warp across iterations.
            vram_neighbors_ref<warp_t, node_id_t> const row {storage.neighbors_for(candidate.id()),
                                                             storage.connectivity};
            node_id_t first_value = lane == 0 ? row.peek_first() : node_id_unknown_k;
            first_value = warp_t::shuffle(first_value, 0);
            // The shuffle moves the word, not the leader's acquire; the lane barrier orders the
            // other lanes' slot loads after it.
            warp_t::sync();
            if (first_value == node_id_unknown_k) {
                // Held through one other expansion instead of waited on; a second refusal drops it.
                if (revisiting || !(held == vram_scored_node<node_id_t>::sentinel())) ++telemetry.locked_drops;
                else held = candidate, held_at_hop = telemetry.hops;
                continue;
            }
            auto const score = [&](u32_t count) noexcept {
                return vram_score_into_frontier<kit_, schedule_, resident_blocks_>( //
                    storage, query_span, query_state, gathered, count, context, lane, fold, rescoring);
            };
            u32_t pending = 0;
            if (vram_expand_round<warp_t>(frontier, visited, telemetry, storage, candidate, first_value,
                                          context.walk_width, first_invisible_id, held, held_at_hop, gathered.ids,
                                          pending, warp_t::width_k, lane, score) == walk_demand_t::sated_k)
                return;
            // A dense kit pays per score call, not per idle lane, so the round scores what it
            // staged.
            if (pending && score(pending) == walk_demand_t::sated_k) return;
        }
    }
    telemetry.locked_drops += !(held == vram_scored_node<node_id_t>::sentinel());
}

/** Greedy best-first walk from random published landmarks toward the query: a fresh frontier
 *  handed to @ref vram_search_ahead. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, vram_frontier frontier_,
          typename fold_ = dummy_fold_t, typename rescoring_ = vram_no_rescoring>
__device__ __forceinline__ void vram_search_around( //
    vram_flat_view<kit_> const& storage, typename kit_::fragment_t const* query_span,
    typename kit_::norms_t query_state, node_id_t sampling_bound, node_id_t first_invisible_id,
    vram_visited_set<typename schedule_::warp_t, node_id_t>& visited,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_warp_context<frontier_>& context, u32_t lane,
    fold_&& fold = fold_ {}, rescoring_ const& rescoring = rescoring_ {}) noexcept {
    context.frontier.clear(lane);
    vram_search_ahead<kit_, schedule_, resident_blocks_>( //
        storage, query_span, query_state, 0, sampling_bound, first_invisible_id, visited, gathered, context, lane, fold,
        rescoring);
}

/**
 *  @brief The greedy diversity prefix over the @p candidates_count candidates @p sorted_candidates
 *      ranks, ascending by distance with ties on the ring, each pair scored as the greedy meets it.
 *
 *  A candidate is kept only when it is closer to the row than to every kept one, and at most one
 *  zero-distance candidate is kept, which @p kept_zero tells. The low @p position_mask bits of a
 *  ranked ID are its cache row, @c selection.candidate_distances holds each cache row's distance to
 *  the row, and @p candidate_norms its norms. Schedules without a tensor-core step run this.
 *
 *  @return The kept count; @c selection.selected_ids holds their cache rows in kept order.
 */
template <vram_quads_kit kit_, typename schedule_, typename storage_>
__device__ __forceinline__ u32_t vram_keep_diverse( //
    storage_ const& storage, vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_sorted_buffer<typename schedule_::warp_t, node_id_t> const& sorted_candidates, u32_t position_mask,
    u32_t candidates_count, vram_selection_scratch<node_id_t> selection, typename kit_::norms_t const* candidate_norms,
    bool& kept_zero, vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    u32_t* const selected_ids = selection.selected_ids;
    u32_t selected_count = 0, staged_through = 0;
    // Rolled: unrolled, every candidate would carry its own pairwise dot and metric dispatch.
#pragma unroll 1
    for (u32_t order = 0; order < candidates_count && selected_count < storage.connectivity; ++order) {
        u32_t const candidate_index = sorted_candidates.at(order).id() & position_mask;
        // Ties visit rows out of staging order, so the wait covers every row visited so far, the
        // kept ones included.
        staged_through = max_of(staged_through, candidate_index + 1);
        schedule.await_rows(staged_through);
        f32_t const to_query = selection.candidate_distances[candidate_index];
        // A second duplicate of the row would only repeat the first one's links.
        bool diverse = !(to_query == 0 && kept_zero);
        for (u32_t base = 0; base < selected_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
            bool lane_diverse = true;
            if (base + lane < selected_count) {
                u32_t const kept_position = selected_ids[base + lane];
                typename kit_::fragment_t const* const candidate_row = cache.staged_row(storage, candidate_index);
                typename kit_::fragment_t const* const kept_row = cache.staged_row(storage, kept_position);
                typename kit_::dot_t accumulator {};
                // Unrolled deeper, the pairwise loops grow the builders by thousands of
                // `accumulate_dot` copies.
#pragma unroll 4
                for (u32_t fragment = 0; fragment != storage.geometry.row_fragments; ++fragment)
                    accumulator = kit_::accumulate_dot(candidate_row[fragment], kept_row[fragment], accumulator);
                f32_t const between = kit_::distance(storage.geometry.kernel, accumulator,
                                                     candidate_norms[candidate_index], candidate_norms[kept_position]);
                lane_diverse = between >= to_query;
                ++telemetry.scored;
            }
            diverse = warp_t::all(lane_diverse);
        }
        if (diverse) {
            kept_zero |= to_query == 0;
            if (lane == 0) selected_ids[selected_count] = candidate_index;
            ++selected_count;
            warp_t::sync();
        }
    }
    return selected_count;
}

/**
 *  @brief Greedy diversity selection over all of the walk's @p selection_nodes candidates,
 *      ascending, for the row @p node_id.
 *
 *  A candidate is kept only when it is closer to the query than to every kept neighbor, so
 *  neighbors spread across directions; the kept neighbors are ascending by construction. Equal
 *  distances order on the hashed ring around @p node_id, and at most one zero-distance candidate
 *  is kept, by diversity or backfill.
 *
 *  @return The kept count; @p selection.selected_ids holds that many node IDs on return.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, vram_frontier frontier_>
__device__ __forceinline__ u32_t vram_select_diverse( //
    vram_flat_view<kit_> const& storage, node_id_t node_id, vram_vector_cache<typename kit_::fragment_t> cache,
    u32_t selection_nodes, schedule_& schedule, typename kit_::norms_t* candidate_norms,
    vram_selection_scratch<node_id_t> selection, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    using scored_node_t = vram_scored_node<node_id_t>;
    u32_t* const selected_ids = selection.selected_ids;
    u32_t* const candidate_ids = selection.candidate_ids;

    // Stage the candidates' IDs and distances by rank & cache their vectors and norms.
    u32_t candidates_count = 0;
    context.frontier.for_each_rank(selection_nodes, lane, [&](u32_t rank, scored_node_t entry) noexcept {
        bool const live = rank < selection_nodes && !(entry == scored_node_t::sentinel());
        if (live) {
            candidate_ids[rank] = entry.id();
            selection.candidate_distances[rank] = entry.distance();
        }
        candidates_count += warp_t::count(warp_t::ballot(live));
    });
    warp_t::sync();
    schedule.stage_rows(storage, cache, candidate_ids, candidates_count, lane);
    for (u32_t row = lane; row < candidates_count; row += warp_t::width_k)
        candidate_norms[row] = storage.norm_for(candidate_ids[row]);
    warp_t::sync();

    // Re-sort by distance, ties on the ring around the row, while the rows copy in; the low bits
    // keep the cache position.
    u32_t const position_mask = ~0u >> __clz(selection_nodes);
    vram_sorted_buffer<warp_t, node_id_t> sorted_candidates {selection.scored_entries, 0, selection_nodes};
    for (u32_t base = 0; base < candidates_count; base += warp_t::width_k) {
        u32_t const position = base + lane, batch_count = min_of(candidates_count - base, warp_t::width_k);
        scored_node_t const keyed =
            lane < batch_count
                ? scored_node_t::make(selection.candidate_distances[position],
                                      (ring_offset(candidate_ids[position], node_id) & ~position_mask) | position)
                : scored_node_t::sentinel();
        sorted_candidates.insert_batch(sort_lanes<warp_t>(keyed, lane), batch_count, lane);
    }

    // The greedy prefix, on tensor-core tiles where the kit has them.
    bool kept_zero = false;
    u32_t selected_count = schedule.template keep_diverse<kit_>( //
        storage, cache, sorted_candidates, position_mask, candidates_count, selection, candidate_norms, kept_zero,
        context.telemetry, lane);
    // The cache outlives this selection, so no staged copy may still be writing into it.
    schedule.await_rows(candidates_count);
    // The slots diversity leaves empty take the nearest rejected candidates, as the CPU's `select_diverse` does:
    // on a dot graph, whose hubs reject one another, a row of diverse picks alone is a handful of links.
    u32_t const diverse_count = selected_count;
    for (u32_t order = 0, matched = 0; order < candidates_count && selected_count < storage.connectivity; ++order) {
        u32_t const candidate_index = sorted_candidates.at(order).id() & position_mask;
        if (matched != diverse_count && selected_ids[matched] == candidate_index) {
            ++matched;
            continue;
        }
        if (kept_zero && selection.candidate_distances[candidate_index] == 0) continue;
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

    /** Candidates the launch's warps cache, as the host sized them: all of an insert's walk
     *  candidates, or a back-link round's re-selection rows. */
    u32_t selection_nodes;

    /** The batch's base ID: staging row 0 goes there, and the warps assign themselves offsets
     *  within @b [first_id,end_id). */
    node_id_t first_id;

    /** The traversal arena every warp carves its slice from. */
    byte_t* arena;

    /** Shape of one warp's slice: the exact visited set, the arena frontier's tail on the rung past
     *  the register ladder, and the builders' scratch when shared memory cannot fit it. */
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
    u32_t* assigned_nodes;

    /** Each node's last pick an appended node of the launch made, @b [rows], unknown for none; pick
     *  @b p is appended node @c first_id + p / connectivity, its slot p % connectivity. */
    u32_t* link_heads;

    /** The pick each pick was threaded before, @b [launch rows,connectivity]. */
    u32_t* link_next;

    /** The nodes picked by any appended node, each once, and their count, zeroed per launch. */
    u32_t* touched_ids;
    u32_t* touched_count;
};

/** Inputs & outputs of one search launch: a batch of queries in, the nearest IDs and - when
 *  requested - their distances out, both ascending per query. */
template <vram_quads_kit kit_>
struct vram_search_view {
    vram_flat_view<kit_> storage;

    /** The query rows, @b [queries,row_fragments], device-resident. */
    typename kit_::fragment_t const* queries;

    u32_t queries_count;

    /** The watermark snapshotted at launch: landmarks are drawn below it and no neighbor at or past
     *  it is followed, while the walk still gates on slot-0 words, which concurrent insertions
     *  publish. */
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

    /** Walks running now, zeroed with @c assigned_queries, which @ref walk_width divides
     *  @c concurrent_walks among. */
    u32_t* live_walks;

    /** Walks the device runs at once on the launch's rung: multiprocessors × resident blocks ×
     *  warps per block. */
    u32_t concurrent_walks;

    /** Rows whose neighbors fill a walk's candidates, the widest round a walk expands. */
    u32_t filling_nodes;
};

/** Inserts the node identified by @p new_node_id: writes the payload and norms, walks the published
 *  graph, selects diverse neighbors, publishes with one release store, then back-links them - one
 *  warp per node. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, vram_frontier frontier_>
__device__ __forceinline__ void vram_insert_one( //
    vram_build_view<kit_> const& view, node_id_t new_node_id,
    vram_visited_set<typename schedule_::warp_t, node_id_t>& visited, typename kit_::fragment_t* query_span,
    vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_selection_scratch<node_id_t> selection, typename kit_::norms_t* candidate_norms,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    vram_flat_view<kit_> const& storage = view.storage;

    // One pass writes the payload, derives the norms, and parks the query in its span.
    typename kit_::fragment_t const* staging_row = vram_row(view.staging, new_node_id - view.first_id,
                                                            storage.geometry.row_fragments);
    typename kit_::fragment_t* owned_row = storage.vector_span_for(new_node_id);
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < storage.geometry.row_fragments; fragment += warp_t::width_k) {
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
        vram_search_around<kit_, schedule_, resident_blocks_>( //
            storage, query_span, query_state, new_node_id, view.end_id, visited, gathered, context, lane,
            dummy_fold_t {}, vram_single_zero_rescoring<frontier_> {context.frontier});
    else context.frontier.clear(lane);

    // The own row needs no lock: no reader arrives before the slot-0 release below flips the sentinel.
    u32_t const selected_count = vram_select_diverse<kit_>( //
        storage, new_node_id, cache, view.selection_nodes, schedule, candidate_norms, selection, context, lane);
    vram_neighbors_ref<warp_t, node_id_t> const own_neighbors {storage.neighbors_for(new_node_id),
                                                               storage.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    warp_t::sync();
    node_id_t const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) {
        own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
        // A tally for the host; the slot-0 release above is what publishes the row.
        warp_t::fetch_add_relaxed(*view.published_rows, 1u);
    }

    // Only now - published, no locks held - thread the picks onto their targets' chains, which the
    // wave's back-link launch merges.
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k) {
        node_id_t const target = selection.selected_ids[slot];
        u32_t const pick = (new_node_id - view.first_id) * storage.connectivity + slot;
        u32_t const previous = warp_t::exchange_acquire(view.link_heads[target], pick);
        view.link_next[pick] = previous;
        if (previous == node_id_unknown_k)
            view.touched_ids[warp_t::fetch_add_relaxed(*view.touched_count, 1u)] = target;
    }
    warp_t::sync();
}

/** Whether @p bitset - a reachability audit's reached set or frontier, a refinement pass's walked
 *  rows - holds @p row. Every access is relaxed: the launch boundary orders the marks before any
 *  later launch or host copy reads them. */
template <typename warp_type_>
__device__ __forceinline__ bool vram_bitset_contains(vram_bitset_t bitset, u32_t row) noexcept {
    return (warp_type_::load_relaxed(bitset.words[row >> 5]) >> (row & 31u)) & 1u;
}

/** Sets the bit of @p row, answering whether this call was the one that set it. */
template <typename warp_type_>
__device__ __forceinline__ bool vram_bitset_insert(vram_bitset_t bitset, u32_t row) noexcept {
    u32_t const bit = 1u << (row & 31u);
    return !(warp_type_::fetch_or_relaxed(bitset.words[row >> 5], bit) & bit);
}

/** Geometry & storage pointers one refinement launch operates on: the live storage it reads, and
 *  the shadow neighbors matrix it writes - same @b [rows,connectivity] shape, observed by no
 *  reader. */
template <vram_quads_kit kit_>
struct vram_refine_view {
    vram_flat_view<kit_> storage;

    /** Refined neighbors go here, @b [rows,connectivity] IDs - plain stores, published wholesale
     *  by the host after the launch drains. */
    u32_t* shadow_neighbors;

    /** IDs strictly below this bound are refined by this launch. */
    u32_t end_id;

    /** Bytes of the per-warp dynamic-shared slice - the insertion path's shape. */
    u32_t scratch_bytes_per_warp;

    /** Candidates the launch's warps cache, as the host sized them: all of a refine's walk
     *  candidates, or a merge's re-selection rows. */
    u32_t selection_nodes;

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
    u32_t* assigned_nodes;

    /** Optional dense ID array to refine instead of the @b [first_id,end_id) range; null refines
     *  the range, and under an array @c end_id bounds its length and offsets index it. */
    u32_t const* node_ids = nullptr;

    /** Published-node ceiling for the walk's landmark sampling: always the graph size, never
     *  @c end_id, which under an ID array is its length. */
    u32_t sampling_bound = 0;

    /** The ascending first rows of the @c parts parts a shard holds, and the @c round of their
     *  merge: each node first walks across the neighboring group of parts alone, as @c part_across
     *  names it. Fewer than two parts for a plain refine. */
    u32_t const* part_bases = nullptr;
    u32_t parts = 0;
    u32_t round = 0;

    /** Each node's last walker of the pass that picked it, @b [rows], unknown for none: the chain
     *  the merge folds back, as an insertion's back-link does. */
    u32_t* link_heads = nullptr;

    /** The walker each walker was chained before, beside its pick, @b [rows,connectivity] like the
     *  shadow neighbors. */
    u32_t* link_next = nullptr;

    /** The rows the pass walked, whose forward picks the merge folds in first. */
    vram_bitset_t walked;
};

/** Re-selects the neighbors of @p node_id: the node's own vector is the query, its current
 *  neighbors seed the walk so every one already wired competes in the re-selection, the diverse
 *  selection goes into the shadow slots, and the node chains itself onto every pick for the merge.
 *  Over several parts the node first walks across the neighboring group of parts from empty
 *  candidates, so that group's nearest compete too. Read-only on the live storage. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, vram_frontier frontier_>
__device__ __forceinline__ void vram_refine_one( //
    vram_refine_view<kit_> const& view, node_id_t node_id,
    vram_visited_set<typename schedule_::warp_t, node_id_t>& visited, typename kit_::fragment_t* query_span,
    vram_vector_cache<typename kit_::fragment_t> cache, schedule_& schedule,
    vram_selection_scratch<node_id_t> selection, typename kit_::norms_t* candidate_norms,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    vram_flat_view<kit_> const& storage = view.storage;

    // The query is the node's own stored vector; its norms are already derived state.
    for (u32_t fragment = lane; fragment < storage.geometry.row_fragments; fragment += warp_t::width_k)
        query_span[fragment] = storage.vector_for(node_id, fragment);
    typename kit_::norms_t const query_state = storage.norm_for(node_id);
    warp_t::sync();

    // The node itself never enters its own candidates - visited before anything stages.
    context.frontier.clear(lane);
    visited.test_and_set(node_id, lane);

    // The walk across runs first, with landmarks from the neighboring group alone: the live graph
    // links no group to another before this round's merge, so the walk stays there, and its nearest
    // would lose to the own neighbors. Both walks share one call site, so the walk inlines once and
    // keeps a plain refine's registers.
    iota_view<node_id_t> const across = view.parts > 1 ? part_across(view.part_bases, std::size_t {view.parts},
                                                                     std::size_t {view.round},
                                                                     node_id_t {view.sampling_bound}, node_id)
                                                       : iota_view<node_id_t> {};
    node_id_t sampling_first = across.first;
    node_id_t sampling_bound = across.bound;
    vram_single_zero_rescoring<frontier_> const single_zero {context.frontier};
#pragma unroll 1
    for (u32_t walk_number = across.empty() ? 1 : 0; walk_number != 2; ++walk_number) {
        if (walk_number) {
            context.frontier.renew();
            sampling_first = 0, sampling_bound = view.sampling_bound;
            // Seed the frontier with the current neighbors, staged like a traversal expansion minus
            // the hop bookkeeping.
            vram_neighbors_ref<warp_t, node_id_t> const own_neighbors {storage.neighbors_for(node_id),
                                                                       storage.connectivity};
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
                        vram_score_into_frontier<kit_, schedule_, resident_blocks_>( //
                            storage, query_span, query_state, gathered, gathered_count, context, lane, dummy_fold_t {},
                            single_zero);
                }
        }
        vram_search_ahead<kit_, schedule_, resident_blocks_>( //
            storage, query_span, query_state, sampling_first, sampling_bound, view.sampling_bound, visited, gathered,
            context, lane, dummy_fold_t {}, single_zero);
    }

    // Re-select the neighbors from all of its candidates.
    u32_t const selected_count = vram_select_diverse<kit_>( //
        storage, node_id, cache, view.selection_nodes, schedule, candidate_norms, selection, context, lane);

    // The insertion's slot convention: neighbors ascending, sentinel tail, a slot-0 self-loop meaning published-empty.
    u32_t* const shadow_slots = vram_row(view.shadow_neighbors, node_id, storage.connectivity);
    for (u32_t slot = lane; slot < storage.connectivity; slot += warp_t::width_k)
        shadow_slots[slot] = slot < selected_count ? selection.selected_ids[slot]
                             : slot == 0           ? node_id
                                                   : node_id_unknown_k;
    if (lane == 0) vram_bitset_insert<warp_t>(view.walked, node_id);
    // Chain this walker onto every node it picked, beside that pick, for the merge to fold back. A
    // heal's islands chain nowhere: re-judging the rows they pick would drop the links that
    // reconnected earlier islands.
    u32_t* const own_links = vram_row(view.link_next, node_id, storage.connectivity);
    if (!view.node_ids)
        for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
            own_links[slot] = warp_t::exchange_acquire(view.link_heads[selection.selected_ids[slot]], node_id);
    warp_t::sync();
}

/**
 *  @brief Re-selects one node's neighbors from the @p candidates_count distinct candidates staged
 *      in @c selection.candidate_ids.
 *
 *  The diversity heuristic runs over them ascending by distance to the node, backfilled with the
 *  nearest pruned ones, and writes them in its live row slot by slot, so a concurrent walk reads
 *  every slot whole. The node's own row is the cache's @p query_row, its last, and the first
 *  @p incumbents candidates are the members its live row holds.
 *
 *  @return Whether a candidate past them was kept: a member the row lacked.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, typename storage_>
__device__ __forceinline__ bool vram_reselect_staged( //
    storage_ const& storage, node_id_t node_id, u32_t candidates_count, u32_t incumbents,
    vram_vector_cache<typename kit_::fragment_t> cache, u32_t query_row, schedule_& schedule,
    typename kit_::norms_t* candidate_norms, vram_selection_scratch<node_id_t> selection,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    u32_t* const candidate_ids = selection.candidate_ids;
    u32_t* const selected_ids = selection.selected_ids;
    typename kit_::norms_t const query_state = storage.norm_for(node_id);

    // Cache the candidate vectors & norms for the pairwise diversity checks, and the node's own
    // row, so the copies finish while the candidates are scored.
    schedule.stage_rows(storage, cache, candidate_ids, candidates_count, lane);
    cache.template place_row<warp_t>(storage, query_row, node_id, lane);
    for (u32_t cache_row = lane; cache_row < candidates_count; cache_row += warp_t::width_k)
        candidate_norms[cache_row] = storage.norm_for(candidate_ids[cache_row]);
    warp_t::sync();
    typename kit_::fragment_t const* const node_span = cache.staged_row(storage, query_row);

    // Score every candidate against the node & sort ascending, ties on the ring around the node;
    // the IDs' low bits are cache-row indices.
    telemetry.scored += candidates_count;
    u32_t const position_mask = ~0u >> __clz(query_row);
    vram_sorted_buffer<warp_t, node_id_t> sorted_candidates {selection.scored_entries, 0, query_row};
    for (u32_t base = 0; base < candidates_count; base += warp_t::width_k) {
        u32_t const position = base + lane, batch_count = min_of(candidates_count - base, warp_t::width_k);
        vram_scored_node<node_id_t> const scored = vram_score_batch<kit_, warp_t, resident_blocks_>( //
            storage, node_span, query_state, candidate_ids + base, batch_count, gathered, lane);
        vram_scored_node<node_id_t> const sorted_batch = sort_lanes<warp_t>(
            lane < batch_count
                ? vram_scored_node<node_id_t>::make(
                      scored.distance(), (ring_offset(candidate_ids[position], node_id) & ~position_mask) | position)
                : vram_scored_node<node_id_t>::sentinel(),
            lane);
        sorted_candidates.insert_batch(sorted_batch, batch_count, lane);
        if (lane < batch_count) selection.candidate_distances[position] = scored.distance();
    }

    // The diversity checks visit the rows by distance, so every row is copied first.
    schedule.await_rows(candidates_count);
    warp_t::sync();

    // The greedy prefix, on tensor-core tiles where the kit has them.
    bool kept_zero = false;
    u32_t refined_count = schedule.template keep_diverse<kit_>( //
        storage, cache, sorted_candidates, position_mask, candidates_count, selection, candidate_norms, kept_zero,
        telemetry, lane);

    // Fill the remaining slots with the nearest pruned candidates, past the node's other
    // duplicates. The diverse prefix is a subsequence of the sorted candidates, so one cursor tells
    // a kept candidate from a rejected one.
    u32_t const diverse_count = refined_count;
    for (u32_t merge_index = 0, matched = 0;
         merge_index != sorted_candidates.size_ && refined_count != storage.connectivity; ++merge_index) {
        vram_scored_node<node_id_t> const entry = sorted_candidates.at(merge_index);
        u32_t const candidate_position = entry.id() & position_mask;
        if (matched != diverse_count && selected_ids[matched] == candidate_position) {
            ++matched;
            continue;
        }
        if (kept_zero && entry.distance() == 0) continue;
        if (lane == 0) selected_ids[refined_count] = candidate_position;
        ++refined_count;
        warp_t::sync();
    }

    // Write the final neighbors in the live matrix - the insertion's slot convention - rewriting
    // only the slots that differ.
    warp_t::sync();
    vram_neighbors_ref<warp_t, node_id_t> const live_row {storage.neighbors_for(node_id), storage.connectivity};
    typename warp_t::mask_t gained_mask = 0;
    for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) { // Warp-uniform.
        u32_t const slot = slot_base + lane;
        u32_t const value = slot < refined_count ? candidate_ids[selected_ids[slot]]
                            : slot == 0          ? node_id
                                                 : node_id_unknown_k;
        // The incumbents lead the cache rows, so a kept row past them is a member the row lacked.
        gained_mask |= warp_t::ballot(slot < refined_count && selected_ids[slot] >= incumbents);
        if (slot < storage.connectivity && live_row.load_slot(slot) != value) live_row.store_slot(slot, value);
    }
    warp_t::sync();
    return gained_mask != 0;
}

/**
 *  @brief Back-links one node in rounds, so no appended node is dropped.
 *
 *  Each round caches the live row and as many appended nodes as fit before the node's own row at
 *  @p query_row - a refinement's @p forward_picks first, a row whose unknown slots stand for none,
 *  then the chain from the node's @p link_head through @p link_next - skipping any the round
 *  already holds, appends them into the free slots when they all fit, and otherwise re-selects over
 *  both through @ref unum::usearch::dense::vram_reselect_staged; the next round starts from the row
 *  the last one wrote. An insertion's chain links picks, pick @b p naming appended node
 *  @b first_id+p/connectivity; a refinement's links walkers, each found beside its pick in
 *  @p walker_rows, so the chain spans any row count.
 *
 *  @return Whether a round kept a member its live row lacked.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, typename storage_>
__device__ __forceinline__ bool vram_back_link_one( //
    storage_ const& storage, node_id_t node_id, u32_t const* forward_picks, u32_t const* walker_rows, u32_t link_head,
    u32_t const* link_next, node_id_t first_id, vram_vector_cache<typename kit_::fragment_t> cache, u32_t query_row,
    schedule_& schedule, typename kit_::norms_t* candidate_norms, vram_selection_scratch<node_id_t> selection,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    u32_t const connectivity = storage.connectivity;
    u32_t* const candidate_ids = selection.candidate_ids;
    vram_neighbors_ref<warp_t, node_id_t> const live_row {storage.neighbors_for(node_id), connectivity};

    // Every lane follows the serial chain in step, reading the same words; the rounds resume where
    // it stopped.
    u32_t forward_slot = forward_picks ? 0 : connectivity;
    u32_t link = link_head;
    bool gained = false;
    while (forward_slot != connectivity || link != node_id_unknown_k) {
        // The live neighbors first, a self-loop in slot 0 meaning none, then the appended nodes.
        u32_t live_count = 0;
        for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            u32_t const live_id = slot < connectivity ? live_row.load_slot(slot) : node_id_unknown_k;
            bool const usable = live_id != node_id_unknown_k && live_id != node_id;
            typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
            if (usable) candidate_ids[live_count + warp_t::count(usable_mask & warp_t::below(lane))] = live_id;
            live_count += warp_t::count(usable_mask);
            warp_t::sync();
        }
        u32_t candidates_count = live_count;
        while (candidates_count != query_row) {
            u32_t appended_node = node_id_unknown_k;
            if (forward_slot != connectivity) appended_node = forward_picks[forward_slot++];
            else if (link == node_id_unknown_k) break;
            else if (!walker_rows) appended_node = first_id + link / connectivity, link = link_next[link];
            else {
                // The walker threaded its link beside its pick of this node.
                u32_t const* const picks = vram_row(walker_rows, link, connectivity);
                u32_t slot = connectivity;
                for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
                    typename warp_t::mask_t const found = warp_t::ballot(slot_base + lane < connectivity &&
                                                                         picks[slot_base + lane] == node_id);
                    if (found) {
                        slot = slot_base + warp_t::first(found);
                        break;
                    }
                }
                // A walker that no longer holds its pick ends the chain, whose link sat beside it.
                u32_t const* const links = vram_row(link_next, link, connectivity);
                appended_node = link, link = slot != connectivity ? links[slot] : node_id_unknown_k;
            }
            if (appended_node == node_id_unknown_k || appended_node == node_id) continue;
            // Mutual picks reach a node twice: through its live row or forward picks, and through
            // its chain.
            bool held = false;
            for (u32_t index = lane; index < candidates_count; index += warp_t::width_k)
                held |= candidate_ids[index] == appended_node;
            if (warp_t::any(held)) continue;
            if (lane == 0) candidate_ids[candidates_count] = appended_node;
            ++candidates_count;
            warp_t::sync();
        }

        // Room for every appended node left: append them past the live neighbors, a published-empty
        // row's self-loop giving way to the first, and nothing to re-judge.
        if (candidates_count <= connectivity) {
            for (u32_t slot = live_count + lane; slot < candidates_count; slot += warp_t::width_k)
                live_row.store_slot(slot, candidate_ids[slot]);
            warp_t::sync();
            return gained || candidates_count != live_count;
        }
        ++telemetry.reselections;
        gained |= vram_reselect_staged<kit_, schedule_, resident_blocks_>(
            storage, node_id, candidates_count, live_count, cache, query_row, schedule, candidate_norms, selection,
            gathered, telemetry, lane);
    }
    return gained;
}

/** One fold launch's view over the rows of one shard of a bin: the bin as one storage, every
 *  shard's fold buffers, the launching shard's place, the cross results each row holds from every
 *  other shard, and the scratch its re-selections or merges carve. */
template <vram_quads_kit kit_>
struct vram_fold_view {
    vram_bin_view<kit_> bin;

    /** Every shard's fold buffers by its place, resident on the executing device. */
    vram_fold_shard_t const* fold_shards;

    /** The bin's shards, and the launching one's place among them. */
    u32_t shards_count;
    u32_t shard;

    /** Cross results a row holds from every other shard, leading its picks row in bin order. */
    u32_t cross_count;

    /** Candidates a warp caches, the node's own row last: a row's members and its cross results,
     *  or a merge round's live row and its appended nodes. */
    u32_t selection_nodes;

    /** Bytes of the per-warp dynamic-shared slice, zero where the arena holds it. */
    u32_t scratch_bytes_per_warp;
    vram_fold_phase_t phase;

    /** The arena the slices carve from when shared memory cannot hold them, and their shape. */
    byte_t* arena;
    vram_arena_shape_t arena_shape;
    vram_telemetry_block_t* telemetry;

    /** Rows of the launching shard warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_nodes;

    /** Where the chains of the shard at @p source onto the rows of the shard at @p target start in
     *  its heads: past the runs of every shard before @p target, its own aside. */
    __device__ __forceinline__ u32_t heads_offset(u32_t target, u32_t source) const noexcept {
        return bin.shards[target].first - (target > source ? bin.shards[source].rows : 0u);
    }
};

/**
 *  @brief Re-selects the fold row of @p node_id from its live members and its cross results, both
 *      tagged into bin IDs, freezes the picks for the merge, and chains the row onto every pick in
 *      another shard.
 *
 *  The members lead the candidates, so @ref unum::usearch::dense::vram_reselect_staged tells a
 *  gained member apart; the cross results follow, @c cross_count of every other shard in bin order,
 *  the sentinel past what a shard returned. The shards' live rows stay as they were, and every
 *  chain head this row moves belongs to its own shard, so no other device contends for it.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_>
__device__ __forceinline__ void vram_fold_reselect_one( //
    vram_fold_view<kit_> const& view, node_id_t node_id, vram_vector_cache<typename kit_::fragment_t> cache,
    schedule_& schedule, vram_selection_scratch<node_id_t> selection, typename kit_::norms_t* candidate_norms,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    vram_bin_view<kit_> const& bin = view.bin;
    u32_t const connectivity = bin.connectivity, row = bin.row_of(node_id);
    vram_fold_shard_t const own = view.fold_shards[view.shard];
    u32_t* const candidate_ids = selection.candidate_ids;

    // Every lane offers one candidate, packed past the ones staged before it.
    u32_t candidates_count = 0;
    auto const stage = [&](node_id_t candidate) noexcept {
        bool const usable = candidate != node_id_unknown_k;
        typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
        if (usable) candidate_ids[candidates_count + warp_t::count(usable_mask & warp_t::below(lane))] = candidate;
        candidates_count += warp_t::count(usable_mask);
        warp_t::sync();
    };
    // The live members first, a self-loop in slot 0 meaning none.
    u32_t const* const live = vram_row(own.live, row, connectivity);
    for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        u32_t const member = slot < connectivity ? live[slot] : node_id_unknown_k;
        stage(member != node_id_unknown_k && member != row ? node_id - row + member : node_id_unknown_k);
    }
    u32_t const incumbents = candidates_count;
    // The cross results lead the picks row until the picks overwrite them.
    u32_t* const picks = vram_row(own.picks, row, connectivity);
    u32_t const crossed = (view.shards_count - 1) * view.cross_count;
    for (u32_t column_base = 0; column_base < crossed; column_base += warp_t::width_k) {
        u32_t const column = column_base + lane;
        u32_t const block = column / view.cross_count;
        u32_t const found = column < crossed ? picks[column] : node_id_unknown_k;
        u32_t const other = block + (block >= view.shard);
        stage(found != node_id_unknown_k ? (other << bin.row_bits) | found : node_id_unknown_k);
    }
    ++telemetry.reselections;
    vram_reselect_staged<kit_, schedule_, resident_blocks_>( //
        bin, node_id, candidates_count, incumbents, cache, view.selection_nodes - 1, schedule, candidate_norms,
        selection, gathered, telemetry, lane);

    // The picks freeze for the merge, whose walkers are found beside the pick each one threads.
    vram_neighbors_ref<warp_t, node_id_t> const folded {bin.neighbors_for(node_id), connectivity};
    u32_t* const own_links = vram_row(own.links, row, connectivity);
    for (u32_t slot = lane; slot < connectivity; slot += warp_t::width_k) {
        node_id_t const pick = folded.load_slot(slot);
        picks[slot] = pick;
        if (pick == node_id_unknown_k || bin.shard_of(pick) == view.shard) continue;
        u32_t* const head = own.heads + view.heads_offset(bin.shard_of(pick), view.shard) + bin.row_of(pick);
        own_links[slot] = warp_t::exchange_acquire(*head, node_id);
    }
    warp_t::sync();
}

/**
 *  @brief Merges into the fold row of @p node_id the rows of the bin's other shards that picked it,
 *      a row of them at a time through @ref unum::usearch::dense::vram_back_link_one, then rewrites
 *      its bin IDs into the bin's dense ID space.
 *
 *  Each other shard threads its own chain onto the row; a walker is found beside its frozen pick of
 *  the row, as a refinement's merge finds one, and enters the inbox, which a round folds in once it
 *  fills a row. Only this warp reaches the row, so the rewrite races no merge.
 *
 *  @return Whether a round kept a member the row lacked.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_>
__device__ __forceinline__ bool vram_fold_merge_one( //
    vram_fold_view<kit_> const& view, node_id_t node_id, vram_vector_cache<typename kit_::fragment_t> cache,
    schedule_& schedule, vram_selection_scratch<node_id_t> selection, typename kit_::norms_t* candidate_norms,
    vram_gathered<kit_, typename schedule_::warp_t>& gathered, vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    vram_bin_view<kit_> const& bin = view.bin;
    u32_t const connectivity = bin.connectivity, row = bin.row_of(node_id);
    // Neither the back-link nor its re-selection reads the kept positions, so they hold the inbox.
    u32_t* const inbox = selection.kept_positions;
    u32_t inboxed = 0;
    bool gained = false;
    auto const fold_inbox = [&]() noexcept {
        for (u32_t slot = inboxed + lane; slot < connectivity; slot += warp_t::width_k) inbox[slot] = node_id_unknown_k;
        warp_t::sync();
        gained |= vram_back_link_one<kit_, schedule_, resident_blocks_>( //
            bin, node_id, inbox, nullptr, node_id_unknown_k, nullptr, 0, cache, view.selection_nodes - 1, schedule,
            candidate_norms, selection, gathered, telemetry, lane);
        inboxed = 0;
    };
    for (u32_t source = 0; source != view.shards_count; ++source) {
        if (source == view.shard) continue;
        vram_fold_shard_t const from = view.fold_shards[source];
        // Every lane follows the serial chain in step, reading the same words.
        u32_t link = from.heads[view.heads_offset(view.shard, source) + row];
        while (link != node_id_unknown_k) {
            u32_t const walker = bin.row_of(link);
            u32_t const* const walker_picks = vram_row(from.picks, walker, connectivity);
            u32_t slot = connectivity;
            for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
                typename warp_t::mask_t const found = warp_t::ballot(slot_base + lane < connectivity &&
                                                                     walker_picks[slot_base + lane] == node_id);
                if (found) {
                    slot = slot_base + warp_t::first(found);
                    break;
                }
            }
            // A walker missing its pick ends the chain, as in the back-link.
            u32_t const* const walker_links = vram_row(from.links, walker, connectivity);
            if (lane == 0) inbox[inboxed] = link;
            ++inboxed;
            link = slot != connectivity ? walker_links[slot] : node_id_unknown_k;
            warp_t::sync();
            if (inboxed == connectivity) fold_inbox();
        }
    }
    if (inboxed) fold_inbox();

    vram_neighbors_ref<warp_t, node_id_t> const folded {bin.neighbors_for(node_id), connectivity};
    for (u32_t slot = lane; slot < connectivity; slot += warp_t::width_k)
        if (node_id_t const member = folded.load_slot(slot); member != node_id_unknown_k)
            folded.store_slot(slot, bin.shards[bin.shard_of(member)].first + bin.row_of(member));
    warp_t::sync();
    return gained;
}

/** Writes each query's nearest @c count IDs and distances out of a search walk, ascending, with
 *  sentinel IDs past the live prefix. */
template <vram_quads_kit kit_>
struct vram_search_sink {
    using fold_t = dummy_fold_t;

    /** The outputs for the matched IDs, @b [queries,count]. */
    node_id_t* result_ids;

    /** The outputs for the matched distances, @b [queries,count]; null skips the writes. */
    distance_t* result_distances;

    /** Matches kept per query. */
    u32_t count;

    /** The metric the ranked distances convert out of. */
    nk_kernel_kind_t kernel;

    template <vram_frontier frontier_>
    __device__ __forceinline__ fold_t open(u32_t, vram_warp_context<frontier_> const&, u32_t) const noexcept {
        return {};
    }

    template <vram_frontier frontier_>
    __device__ __forceinline__ void close(fold_t const&, vram_warp_context<frontier_>& context, u32_t query_index,
                                          u32_t lane) const noexcept {
        // The rank extraction shuffles - every lane participates, only in-range ones store.
        node_id_t* const ids_row = vram_row(result_ids, query_index, count);
        distance_t* const distances_row = result_distances ? vram_row(result_distances, query_index, count) : nullptr;
        context.frontier.for_each_rank(count, lane, [&](u32_t rank, vram_scored_node<node_id_t> entry) noexcept {
            if (rank >= count) return;
            ids_row[rank] = entry.id();
            if (distances_row) distances_row[rank] = kit_::convert_to_distance(kernel, entry.distance());
        });
    }
};

/** Writes each query's kept node IDs and native dots out of a multiply walk, and the pairs it scored. */
template <vram_quads_kit kit_, typename warp_type_>
struct vram_multiply_sink {
    using fold_t = vram_multiply_fold<kit_, warp_type_>;

    /** Node IDs of the kept pairs, @b [queries,wanted], sentinel past them. */
    node_id_t* ids;

    /** Native dots of the kept pairs, parallel to @c ids. */
    typename kit_::dot_t* products;

    /** Pairs scored per query, above @c wanted when a row was truncated. */
    u32_t* counts;

    /** Pairs kept per query. */
    u32_t wanted;

    template <vram_frontier frontier_>
    __device__ __forceinline__ fold_t open(u32_t query_index, vram_warp_context<frontier_> const&,
                                           u32_t) const noexcept {
        return {vram_row(ids, query_index, wanted), vram_row(products, query_index, wanted), wanted};
    }

    template <vram_frontier frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, vram_warp_context<frontier_>&, u32_t query_index,
                                          u32_t lane) const noexcept {
        for (u32_t slot = fold.written + lane; slot < wanted; slot += warp_type_::width_k)
            fold.ids[slot] = node_id_unknown_k;
        if (lane == 0) counts[query_index] = fold.scored;
    }
};

/** One query's full walk: stages the query into its shared span, walks the graph under the fold the
 *  sink opens and the query's slice of @p rescorings, and lets the sink close the query. It is safe
 *  against concurrent insertions, since the walk gates on every slot-0 word and writes nothing. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, typename sink_,
          typename rescoring_, vram_frontier frontier_>
__device__ __forceinline__ void vram_walk_one( //
    vram_search_view<kit_> const& view, sink_ const& sink, rescoring_ const& rescorings, u32_t query_index,
    typename kit_::fragment_t* query_span, vram_visited_set<typename schedule_::warp_t, node_id_t>& visited,
    frontier_ frontier, vram_gathered<kit_, typename schedule_::warp_t>& gathered, u32_t lane) noexcept {

    using warp_t = typename schedule_::warp_t;
    typename kit_::fragment_t const* query_row = vram_row(view.queries, query_index,
                                                          view.storage.geometry.row_fragments);
    typename kit_::norms_accumulator_t norms_accumulator {};
    for (u32_t fragment = lane; fragment < view.storage.geometry.row_fragments; fragment += warp_t::width_k) {
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
    context.scratch = reinterpret_cast<byte_t*>(query_span + view.storage.geometry.row_fragments);
    // Concurrent walks the batch leaves idle widen this one, never past its filling rows.
    u32_t live = lane == 0 ? warp_t::fetch_add_relaxed(*view.live_walks, 1u) + 1 : 0;
    live = warp_t::shuffle(live, 0);
    context.walk_width = walk_width(view.concurrent_walks, live, view.queries_count - query_index, view.filling_nodes);
    typename sink_::fold_t fold = sink.open(query_index, context, lane);
    rescoring_ const rescoring = rescorings.for_query(query_index, context);
    vram_search_around<kit_, schedule_, resident_blocks_>( //
        view.storage, query_span, query_state, view.sampling_bound, view.sampling_bound, visited, gathered, context,
        lane, fold, rescoring);
    if (lane == 0) warp_t::fetch_sub_relaxed(*view.live_walks, 1u);
    sink.close(fold, context, query_index, lane);
    context.telemetry.tail_compactions = context.frontier.compactions();
    context.telemetry.tail_dropped = context.frontier.dropped();
    context.telemetry.visited_saturations = visited.saturations_;
    context.telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

/** Moves rows @b [first_row,first_row+rows_count) of @p neighbors between ID spaces as the host's
 *  @c rebase_neighbors and @c compact_neighbors do, one warp per row.
 *  An ID from @p source_first below @p source_bound goes to @p shift past its offset, the rest fold
 *  away, the survivors pack to the front, and a row nothing survived in points at itself. */
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

/** A ranking of opaque keys as a sharded search writes it on a device: per query, @c width keys of
 *  @c key_bytes bytes and as many scores, f64 or f32 by @c score_bytes, rows @c keys_stride bytes
 *  apart and scores @c scores_stride bytes apart, and each query's count of ranks held. */
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

    /** Writes @p key and @p score at @p rank of @p query; an f32 row narrows exactly, since every
     *  score began as an f32 distance. */
    __device__ __forceinline__ void store(u32_t query, u32_t rank, byte_t const* key_source,
                                          f64_t score_value) const noexcept {
        store_key(query, rank, key_source);
        byte_t* const row = scores + query * scores_stride;
        f64_t* const wide_row = reinterpret_cast<f64_t*>(row);
        f32_t* const narrow_row = reinterpret_cast<f32_t*>(row);
        if (score_bytes == sizeof(f64_t)) wide_row[rank] = score_value;
        else narrow_row[rank] = static_cast<f32_t>(score_value);
    }

    /** Writes @p key_source at @p rank of @p query, leaving its score. */
    __device__ __forceinline__ void store_key(u32_t query, u32_t rank, byte_t const* key_source) const noexcept {
        // A loop, since `std::memcpy` names the host function under HIP.
        byte_t* const key_target = key(query, rank);
        for (u32_t offset = 0; offset != key_bytes; ++offset) key_target[offset] = key_source[offset];
    }

    /** The score bytes at @p rank of @p query, whatever their type. */
    __device__ __forceinline__ byte_t* score_bytes_at(u32_t query, u32_t rank) const noexcept {
        return scores + query * scores_stride + std::size_t {rank} * score_bytes;
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
 *  the equal ones already held, so a run of equal distances writes reversed; ranks past the width
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

#pragma region Gathers and Products

/**
 *  @brief Copies the row of every entry @p gather lists into its output row, one warp per entry
 *      and one lane per unit of the row's cast, converting the stored scalar into the outputs'
 *      as NumKong's bulk cast does on the SIMT cores.
 *
 *  Every row shares the one plan, which the host built over addresses aligned as every row's are.
 */
template <typename warp_type_>
__device__ __forceinline__ void vram_gather_vectors(vram_gather_t const& gather) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t entry = warp.index; entry < gather.count; entry += warp.count) {
        vram_gathered_row_t const row = gather.rows[entry];
        nk_cast_arguments_t const cast {
            reinterpret_cast<unsigned char const*>(gather.source + std::size_t {row.node} * gather.source_stride),
            reinterpret_cast<unsigned char*>(gather.outputs + std::size_t {row.output_row} * gather.output_stride),
            gather.cast};
        for (nk_size_t unit = warp.lane; unit < gather.cast.units; unit += warp_type_::width_k)
            nk_cast_unit_simt_(&cast, unit);
    }
}

/** Sums the @p queries_count counts at @p counts into @p total on the grid's first warp alone, so
 *  the host copies back one word instead of every query's count. */
template <typename warp_type_>
__device__ __forceinline__ void vram_sum_counts(std::size_t const* counts, u32_t queries_count,
                                                std::size_t* total) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    if (warp.index != 0) return;
    std::size_t sum = 0;
    for (u32_t query = warp.lane; query < queries_count; query += warp_type_::width_k) sum += counts[query];
    for (u32_t mask = warp_type_::width_k / 2; mask; mask >>= 1) sum += warp_type_::shuffle_xor(sum, mask);
    if (warp.lane == 0) *total = sum;
}

/**
 *  @brief Resolves one shard's walked products into @p ranking, one warp per query: the first
 *      @p scored of the query's @p ids, no more than the ranking's width, become their keys
 *      through the shard's device @p keys column, and their @p products come along beside them,
 *      exactly as they are.
 *
 *  The IDs and products, of @p dot_type_, lie @p ids_stride values apart per query; a ranking of
 *  f64 scores widens each product, any other copies it as it is, and a null @p products leaves the
 *  scores. The ranking's counts, where it keeps any, take the whole scored count, so a ranking of
 *  width zero still counts.
 */
template <typename warp_type_, typename dot_type_>
__device__ __forceinline__ void vram_resolve_products(node_id_t const* ids, dot_type_ const* products,
                                                      u32_t const* scored, byte_t const* keys, u32_t ids_stride,
                                                      u32_t queries_count, vram_ranking_view const& ranking) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t query = warp.index; query < queries_count; query += warp.count) {
        node_id_t const* const query_ids = vram_row(ids, query, ids_stride);
        u32_t const kept = min_of(scored[query], ranking.width);
        for (u32_t pair = warp.lane; pair < kept; pair += warp_type_::width_k) {
            ranking.store_key(query, pair, keys + std::size_t {query_ids[pair]} * ranking.key_bytes);
            if (!products) continue;
            dot_type_ const product = products[std::size_t {query} * ids_stride + pair];
            if (ranking.score_bytes == sizeof(f64_t) && sizeof(dot_type_) != sizeof(f64_t))
                *reinterpret_cast<f64_t*>(ranking.score_bytes_at(query, pair)) = static_cast<f64_t>(product);
            else warp_type_::copy_bytes(ranking.score_bytes_at(query, pair), &product, sizeof(dot_type_));
        }
        if (ranking.counts && warp.lane == 0) ranking.counts[query] = scored[query];
    }
}

/** How many pairs query @p query keeps across the @p shards rankings @p block_bytes apart from
 *  @p gathered: the sum of their counts, no more than the width. */
__device__ __forceinline__ std::size_t vram_kept_pairs(vram_ranking_view const& gathered, std::size_t block_bytes,
                                                       u32_t shards, u32_t query) noexcept {
    std::size_t total = 0;
    for (u32_t shard = 0; shard != shards; ++shard) total += gathered.shifted(shard * block_bytes).counts[query];
    return min_of<std::size_t>(total, gathered.width);
}

/**
 *  @brief Packs the @p shards rankings @p block_bytes apart from @p gathered into @p tape on one
 *      block: each query keeps the pairs of the first shard, then of the next, until the width
 *      fills, keys and scores alike, back to back from its offset.
 *
 *  The block's first warp scans the kept counts into the @p queries_count offsets past
 *  @c tape.offsets[0], the base the previous window ended at, and sums every query's counts into
 *  @p totals unless it is null; then every warp of the block copies whole queries. A tape never
 *  overlaps the rankings, whose rows a packed run of an earlier query could otherwise overwrite
 *  before they move. A ranking without scores packs its keys alone.
 */
template <typename warp_type_>
__device__ __forceinline__ void vram_compact_products(vram_ranking_view const& gathered, std::size_t block_bytes,
                                                      u32_t shards, u32_t queries_count,
                                                      vram_products_tape_t const& tape, std::size_t* totals) noexcept {
    u32_t const lane = threadIdx.x % warp_type_::width_k, warp = threadIdx.x / warp_type_::width_k,
                warps = blockDim.x / warp_type_::width_k;
    if (warp == 0) {
        std::size_t carried = tape.offsets[0];
        for (u32_t base = 0; base < queries_count; base += warp_type_::width_k) { // Warp-uniform for the scan.
            u32_t const query = base + lane;
            std::size_t kept = query < queries_count ? vram_kept_pairs(gathered, block_bytes, shards, query) : 0;
            if (totals && query < queries_count) {
                std::size_t total = 0;
                for (u32_t shard = 0; shard != shards; ++shard)
                    total += gathered.shifted(shard * block_bytes).counts[query];
                totals[query] = total;
            }
            for (u32_t delta = 1; delta < warp_type_::width_k; delta <<= 1) {
                std::size_t const below = warp_type_::shuffle_up(kept, delta);
                if (lane >= delta) kept += below;
            }
            if (query < queries_count) tape.offsets[query + 1] = carried + kept;
            carried += warp_type_::shuffle(kept, warp_type_::width_k - 1);
        }
    }
    __syncthreads();
    for (u32_t query = warp; query < queries_count; query += warps) {
        std::size_t const first = tape.offsets[query];
        std::size_t kept = 0;
        for (u32_t shard = 0; shard != shards && kept < gathered.width; ++shard) {
            vram_ranking_view const own = gathered.shifted(shard * block_bytes);
            std::size_t const taking = min_of<std::size_t>(own.counts[query], gathered.width - kept);
            for (u32_t pair = lane; pair < taking; pair += warp_type_::width_k) {
                std::size_t const position = first + kept + pair;
                warp_type_::copy_bytes(tape.keys + position * own.key_bytes, own.key(query, pair), own.key_bytes);
                warp_type_::copy_bytes(tape.scores + position * own.score_bytes, own.score_bytes_at(query, pair),
                                       own.score_bytes);
            }
            kept += taking;
        }
    }
}

#pragma endregion Gathers and Products

#pragma region Exact MaxSim

/** Whether the @p key_bytes bytes at @p first and @p second are one key. */
__device__ __forceinline__ bool vram_same_key(byte_t const* first, byte_t const* second, u32_t key_bytes) noexcept {
    for (u32_t offset = 0; offset != key_bytes; ++offset)
        if (first[offset] != second[offset]) return false;
    return true;
}

/**
 *  @brief Nominates every query's distinct keys, one warp per query: the keys any of the
 *      @p shards rankings @p block_bytes apart from @p gathered holds for any of the query's
 *      token rows, @p query_offsets apart, written to the query's row of @p nominees in the order
 *      first visited, shard by shard, row by row, rank by rank, and counted in its counts.
 *
 *  Each key meets the nominees so far across the warp's lanes before the first lane files it, so
 *  the warp syncs before the next key reads them; keys past the nominees' width drop.
 */
template <typename warp_type_>
__device__ __forceinline__ void vram_nominate_keys(vram_ranking_view const& gathered, std::size_t block_bytes,
                                                   u32_t shards, u32_t const* query_offsets, u32_t queries_count,
                                                   vram_ranking_view const& nominees) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t query = warp.index; query < queries_count; query += warp.count) {
        u32_t held = 0;
        for (u32_t shard = 0; shard != shards; ++shard) {
            vram_ranking_view const own = gathered.shifted(shard * block_bytes);
            for (u32_t row = query_offsets[query]; row != query_offsets[query + 1]; ++row) {
                u32_t const ranked = static_cast<u32_t>(min_of<std::size_t>(own.counts[row], own.width));
                for (u32_t rank = 0; rank != ranked && held != nominees.width; ++rank) {
                    byte_t const* const key = own.key(row, rank);
                    bool present = false;
                    for (u32_t nominee = warp.lane; nominee < held && !present; nominee += warp_type_::width_k)
                        present = vram_same_key(nominees.key(query, nominee), key, own.key_bytes);
                    if (warp_type_::any(present)) continue;
                    if (warp.lane == 0) nominees.store_key(query, held, key);
                    ++held;
                    warp_type_::sync();
                }
            }
        }
        if (warp.lane == 0) nominees.counts[query] = held;
    }
}

/** The query whose nominees the union tape's @p offsets, one past each of @p queries_count
 *  queries, place @p nominee among. */
__device__ __forceinline__ u32_t vram_query_of(std::size_t const* offsets, u32_t queries_count,
                                               std::size_t nominee) noexcept {
    return vram_rank_below(queries_count, nominee, vram_ties_t::included_k,
                           [&](u32_t query) noexcept { return offsets[query + 1]; });
}

/**
 *  @brief Rescores every nominee of the union tape on one shard, one warp per nominee: for each
 *      token row of its query, the largest dot of that row with any of the shard's @p entries
 *      of the nominee, or the dot's lowest value where the shard holds none.
 *
 *  The @p entries list the shard's nodes of each nominee as their output rows, ascending, so a
 *  nominee's run is found by two binary searches. The token rows are the window's staged
 *  @p queries, fragment rows as the storage pads them, @p query_offsets apart per query; the
 *  dots, in the kit's own type, land @p rows_stride apart per nominee in @p partials.
 */
template <vram_quads_kit kit_, typename warp_type_>
__device__ __forceinline__ void vram_maxsim_rescore(vram_flat_view<kit_> const& storage,
                                                    typename kit_::fragment_t const* queries,
                                                    u32_t const* query_offsets, u32_t queries_count,
                                                    std::size_t const* nominee_offsets, u32_t nominees_count,
                                                    vram_gathered_row_t const* entries, u32_t entries_count,
                                                    typename kit_::dot_t* partials, u32_t rows_stride) noexcept {
    using dot_t = typename kit_::dot_t;
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    u32_t const row_fragments = storage.geometry.row_fragments;
    auto output_row_of = [&](u32_t entry) noexcept { return entries[entry].output_row; };
    for (u32_t nominee = warp.index; nominee < nominees_count; nominee += warp.count) {
        u32_t const query = vram_query_of(nominee_offsets, queries_count, nominee);
        u32_t const first_row = query_offsets[query], rows = query_offsets[query + 1] - first_row;
        u32_t const first_entry = vram_rank_below(entries_count, nominee, vram_ties_t::excluded_k, output_row_of);
        u32_t const end_entry = vram_rank_below(entries_count, nominee, vram_ties_t::included_k, output_row_of);
        for (u32_t row = 0; row != rows; ++row) {
            typename kit_::fragment_t const* const token = queries + std::size_t {first_row + row} * row_fragments;
            dot_t best = std::numeric_limits<dot_t>::lowest();
            for (u32_t entry = first_entry; entry != end_entry; ++entry) {
                dot_t partial {};
                for (u32_t fragment = warp.lane; fragment < row_fragments; fragment += warp_type_::width_k)
                    partial = kit_::accumulate_dot(storage.vector_for(entries[entry].node, fragment), token[fragment],
                                                   partial);
                dot_t const dot = vram_reduce<warp_type_>(partial, warp_type_::width_k);
                if (dot > best) best = dot;
            }
            if (warp.lane == 0) partials[std::size_t {nominee} * rows_stride + row] = best;
        }
    }
}

/** Folds every shard's partials into each nominee's score, one warp per nominee: per token row of
 *  its query the largest of the @p shards partials @p block_bytes apart from @p partials, summed
 *  over the rows in f64. */
template <vram_quads_kit kit_, typename warp_type_>
__device__ __forceinline__ void vram_maxsim_fold(typename kit_::dot_t const* partials, std::size_t block_bytes,
                                                 u32_t shards, u32_t rows_stride, u32_t const* query_offsets,
                                                 u32_t queries_count, std::size_t const* nominee_offsets,
                                                 u32_t nominees_count, f64_t* totals) noexcept {
    using dot_t = typename kit_::dot_t;
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t nominee = warp.index; nominee < nominees_count; nominee += warp.count) {
        u32_t const query = vram_query_of(nominee_offsets, queries_count, nominee);
        u32_t const rows = query_offsets[query + 1] - query_offsets[query];
        f64_t sum = 0;
        for (u32_t row = warp.lane; row < rows; row += warp_type_::width_k) {
            dot_t best = std::numeric_limits<dot_t>::lowest();
            for (u32_t shard = 0; shard != shards; ++shard) {
                dot_t const* const own = reinterpret_cast<dot_t const*>(reinterpret_cast<byte_t const*>(partials) +
                                                                        shard * block_bytes);
                dot_t const partial = own[std::size_t {nominee} * rows_stride + row];
                if (partial > best) best = partial;
            }
            sum += static_cast<f64_t>(best);
        }
        for (u32_t mask = warp_type_::width_k / 2; mask; mask >>= 1) sum += warp_type_::shuffle_xor(sum, mask);
        if (warp.lane == 0) totals[nominee] = sum;
    }
}

/**
 *  @brief Appends every query's @p wanted best nominees to @p tape, one warp per query: their
 *      keys from the union tape @p nominated, and their f64 @p totals, largest first, ties broken
 *      by nominee order.
 *
 *  A query's run starts past the runs of every query before it, from @c tape.offsets[0], the base
 *  the previous window ended at, which each warp sums for its own query, so no warp waits on
 *  another; a nominee's rank counts the nominees ahead of it. Every query's nominee count is
 *  written to @p scored.
 */
template <typename warp_type_>
__device__ __forceinline__ void vram_top_k(f64_t const* totals, vram_products_tape_t const& nominated,
                                           u32_t queries_count, std::size_t wanted, u32_t key_bytes,
                                           vram_products_tape_t const& tape, std::size_t* scored) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t query = warp.index; query < queries_count; query += warp.count) {
        std::size_t before = 0;
        for (u32_t earlier = warp.lane; earlier < query; earlier += warp_type_::width_k)
            before += min_of<std::size_t>(nominated.offsets[earlier + 1] - nominated.offsets[earlier], wanted);
        for (u32_t mask = warp_type_::width_k / 2; mask; mask >>= 1) before += warp_type_::shuffle_xor(before, mask);
        std::size_t const first = nominated.offsets[query], count = nominated.offsets[query + 1] - first;
        std::size_t const start = tape.offsets[0] + before, kept = min_of(count, wanted);
        if (warp.lane == 0) tape.offsets[query + 1] = start + kept, scored[query] = count;
        for (std::size_t own = warp.lane; own < count; own += warp_type_::width_k) {
            f64_t const total = totals[first + own];
            std::size_t rank = 0;
            for (std::size_t other = 0; other != count && rank < kept; ++other) {
                f64_t const other_total = totals[first + other];
                rank += other_total > total || (other_total == total && other < own);
            }
            if (rank >= kept) continue;
            warp_type_::copy_bytes(tape.keys + (start + rank) * key_bytes, nominated.keys + (first + own) * key_bytes,
                                   key_bytes);
            f64_t* const ranked_scores = reinterpret_cast<f64_t*>(tape.scores);
            ranked_scores[start + rank] = total;
        }
    }
}

#pragma endregion Exact MaxSim

#pragma region Views

static_assert(sizeof(vram_scored_node<node_id_t>) == gpu_scored_node_bytes<node_id_t>);

/** The shared views and sinks, field by field from the host's records. */
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_flat_view<kit_> vram_view(vram_flat_storage_t const& storage) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t*>(storage.vectors),
            reinterpret_cast<typename kit_::norms_t*>(storage.norms),
            storage.neighbors,
            storage.connectivity,
            {storage.row_fragments, storage.lanes_per_row, storage.steps_per_row, storage.row_fragments_magic,
             storage.kernel}};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_build_view<kit_> vram_view(vram_build_t const& build) noexcept {
    return {vram_view<kit_>(build.storage),
            reinterpret_cast<typename kit_::fragment_t const*>(build.staging),
            build.end_id,
            build.scratch_bytes_per_warp,
            build.selection_nodes,
            build.first_id,
            build.arena,
            vram_view(build.arena_shape),
            build.expansion,
            build.epoch_base,
            build.published_rows,
            build.telemetry,
            build.assigned_nodes,
            build.link_heads,
            build.link_next,
            build.touched_ids,
            build.touched_count};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_search_view<kit_> vram_view(vram_search_t const& search) noexcept {
    return {vram_view<kit_>(search.storage),
            reinterpret_cast<typename kit_::fragment_t const*>(search.queries),
            search.queries_count,
            search.sampling_bound,
            search.scratch_bytes_per_warp,
            search.expansion,
            search.arena,
            vram_view(search.arena_shape),
            search.epoch_base,
            search.telemetry,
            search.assigned_queries,
            search.live_walks,
            search.concurrent_walks,
            search.filling_nodes};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_refine_view<kit_> vram_view(vram_refine_t const& refine) noexcept {
    return {vram_view<kit_>(refine.storage),
            refine.shadow_neighbors,
            refine.end_id,
            refine.scratch_bytes_per_warp,
            refine.selection_nodes,
            refine.first_id,
            refine.arena,
            vram_view(refine.arena_shape),
            refine.expansion,
            refine.epoch_base,
            refine.telemetry,
            refine.assigned_nodes,
            refine.node_ids,
            refine.sampling_bound,
            refine.part_bases,
            refine.parts,
            refine.round,
            refine.link_heads,
            refine.link_next,
            refine.walked};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_search_sink<kit_> vram_view(vram_search_sink_t const& sink) noexcept {
    return {sink.result_ids, sink.result_distances, sink.count, sink.kernel};
}
template <vram_quads_kit kit_, typename warp_type_>
__device__ __forceinline__ vram_multiply_sink<kit_, warp_type_> vram_view(vram_multiply_sink_t const& sink) noexcept {
    return {sink.ids, reinterpret_cast<typename kit_::dot_t*>(sink.products), sink.counts, sink.wanted};
}
template <vram_quads_kit kit_, typename warp_type_>
__device__ __forceinline__ vram_indexer_rescoring<kit_, warp_type_> vram_view(
    vram_rescoring_t const& rescoring) noexcept {
    return {reinterpret_cast<typename kit_::fragment_t const*>(rescoring.heads), rescoring.weights,
            rescoring.heads_count, rescoring.row_fragments};
}
template <vram_quads_kit kit_>
__device__ __forceinline__ vram_fold_view<kit_> vram_view(vram_fold_t const& fold) noexcept {
    return {{fold.bin_shards, fold.storage.connectivity, vram_view<kit_>(fold.storage).geometry, fold.row_bits,
             (1u << fold.row_bits) - 1u},
            fold.fold_shards,
            fold.shards_count,
            fold.shard,
            fold.cross_count,
            fold.selection_nodes,
            fold.scratch_bytes_per_warp,
            fold.phase,
            fold.arena,
            vram_view(fold.arena_shape),
            fold.telemetry,
            fold.assigned_nodes};
}
__device__ __forceinline__ vram_ranking_view vram_view(vram_ranking_t const& ranking) noexcept {
    return {ranking.keys,          ranking.scores, ranking.counts,    ranking.keys_stride,
            ranking.scores_stride, ranking.width,  ranking.key_bytes, ranking.score_bytes};
}

#pragma endregion Views

#pragma region Builders

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
 *  narrower than 32, so no walk waits on its graph while the graph is small. A appended node only
 *  threads its picks onto their chains; the wave's @ref unum::usearch::dense::vram_back_link launch
 *  links them back, so appended nodes of one wave reach each other only through their own picks.
 *  Searches see a batch only once the host has drained it and turned the watermark past it, a whole
 *  batch at a time and in claim order, and follow no neighbor at or past the watermark they
 *  launched under.
 */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t candidate_slots_, u32_t resident_blocks_>
__device__ __forceinline__ void vram_insert(vram_build_view<kit_> const& view) {
    using warp_t = typename schedule_::warp_t;
    // The scratch slice hosts the query span while traversing, then the candidate cache while
    // selecting; the exact visited set lives in the warp's arena slice, not in shared memory.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    u32_t const selection_nodes = view.selection_nodes;
    vram_vector_cache<typename kit_::fragment_t> cache {query_span, view.storage.geometry.row_fragments};
    vram_selection_scratch<node_id_t> const selection = carve_selection_scratch<node_id_t>(cache, selection_nodes);
    cache.ids_ = view.arena_shape.scratch_bytes ? selection.candidate_ids : nullptr;
    typename kit_::norms_t* const candidate_norms = reinterpret_cast<typename kit_::norms_t*>(
        selection.end(selection_nodes));
    schedule_ schedule {schedule_states[warp], lane};
    using frontier_t = std::conditional_t<candidate_slots_ == 0, vram_arena_frontier<warp_t, node_id_t>,
                                          vram_register_frontier<candidate_slots_, warp_t, node_id_t>>;
    while (true) {
        u32_t new_node_id = lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_nodes, 1u) : 0;
        new_node_id = view.first_id + warp_t::shuffle(new_node_id, 0);
        if (new_node_id >= view.end_id) return;

        vram_warp_context<frontier_t> context;
        if constexpr (candidate_slots_ == 0) {
            context.frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_t>(view.arena,
                                                                                             slice.warp_index);
            context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
            context.frontier.expansion_target_ = view.expansion;
        }
        context.random_generator = vram_lane_generator<warp_t>(new_node_id, lane);
        vram_visited_set<warp_t, node_id_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                     view.epoch_base + (new_node_id - view.first_id) + 1};
        vram_insert_one<kit_, schedule_, resident_blocks_>( //
            view, new_node_id, visited, query_span, cache, schedule, selection, candidate_norms, gathered[warp],
            context, lane);
        context.telemetry.tail_compactions = context.frontier.compactions();
        context.telemetry.tail_dropped = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
    }
}

/** Back-links an insertion wave: one warp per node any appended node picked, each assigned from the
 *  launch counter, the only writer of that node's row, so no row is locked. Each node's chain head
 *  resets to unknown once merged, leaving the heads ready for the next wave. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_>
__device__ __forceinline__ void vram_back_link(vram_build_view<kit_> const& view) {
    using warp_t = typename schedule_::warp_t;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    u32_t const selection_nodes = view.selection_nodes;
    vram_vector_cache<typename kit_::fragment_t> cache {reinterpret_cast<typename kit_::fragment_t*>(slice.scratch),
                                                        view.storage.geometry.row_fragments};
    vram_selection_scratch<node_id_t> const selection = carve_selection_scratch<node_id_t>(cache, selection_nodes);
    cache.ids_ = view.arena_shape.scratch_bytes ? selection.candidate_ids : nullptr;
    typename kit_::norms_t* const candidate_norms = reinterpret_cast<typename kit_::norms_t*>(
        selection.end(selection_nodes));
    schedule_ schedule {schedule_states[warp], lane};
    u32_t const touched = *view.touched_count;
    while (true) {
        u32_t offset = lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_nodes, 1u) : 0;
        offset = warp_t::shuffle(offset, 0);
        if (offset >= touched) return;
        node_id_t const node_id = view.touched_ids[offset];
        vram_telemetry_t telemetry;
        vram_back_link_one<kit_, schedule_, resident_blocks_>( //
            view.storage, node_id, nullptr, nullptr, view.link_heads[node_id], view.link_next, view.first_id, cache,
            selection_nodes - 1, schedule, candidate_norms, selection, gathered[warp], telemetry, lane);
        if (lane == 0) view.link_heads[node_id] = node_id_unknown_k;
        telemetry.template flush<warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
    }
}

/** One warp refines one node, assigning itself node IDs from the pass counter; warps only read the
 *  live storage and write disjoint shadow neighbors. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t candidate_slots_, u32_t resident_blocks_>
__device__ __forceinline__ void vram_refine(vram_refine_view<kit_> const& view) {
    using warp_t = typename schedule_::warp_t;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    u32_t const selection_nodes = view.selection_nodes;
    vram_vector_cache<typename kit_::fragment_t> cache {query_span, view.storage.geometry.row_fragments};
    vram_selection_scratch<node_id_t> const selection = carve_selection_scratch<node_id_t>(cache, selection_nodes);
    cache.ids_ = view.arena_shape.scratch_bytes ? selection.candidate_ids : nullptr;
    typename kit_::norms_t* const candidate_norms = reinterpret_cast<typename kit_::norms_t*>(
        selection.end(selection_nodes));
    schedule_ schedule {schedule_states[warp], lane};
    using frontier_t = std::conditional_t<candidate_slots_ == 0, vram_arena_frontier<warp_t, node_id_t>,
                                          vram_register_frontier<candidate_slots_, warp_t, node_id_t>>;
    while (true) {
        u32_t offset = lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_nodes, 1u) : 0;
        offset = warp_t::shuffle(offset, 0);
        // Under an ID array, end_id bounds it; the assigned offset indexes it.
        if (view.first_id + offset >= view.end_id) return;
        u32_t const node_id = view.node_ids ? view.node_ids[offset] : view.first_id + offset;

        vram_warp_context<frontier_t> context;
        if constexpr (candidate_slots_ == 0) {
            context.frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_t>(view.arena,
                                                                                             slice.warp_index);
            context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
            context.frontier.expansion_target_ = view.expansion;
        }
        context.random_generator = vram_lane_generator<warp_t>(node_id, lane);
        vram_visited_set<warp_t, node_id_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                     view.epoch_base + offset + 1};
        vram_refine_one<kit_, schedule_, resident_blocks_>( //
            view, node_id, visited, query_span, cache, schedule, selection, candidate_norms, gathered[warp], context,
            lane);
        context.telemetry.tail_compactions = context.frontier.compactions();
        context.telemetry.tail_dropped = context.frontier.dropped();
        context.telemetry.visited_saturations = visited.saturations_;
        context.telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** One warp folds one node's forward picks and the walkers chained onto it into its live row
 *  through @ref unum::usearch::dense::vram_back_link_one, assigning itself node IDs from the pass
 *  counter; only a walked row that gained a member counts as changed. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_>
__device__ __forceinline__ void vram_merge_back_links(vram_refine_view<kit_> const& view) {
    using warp_t = typename schedule_::warp_t;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    u32_t const selection_nodes = view.selection_nodes;
    vram_vector_cache<typename kit_::fragment_t> cache {reinterpret_cast<typename kit_::fragment_t*>(slice.scratch),
                                                        view.storage.geometry.row_fragments};
    vram_selection_scratch<node_id_t> const selection = carve_selection_scratch<node_id_t>(cache, selection_nodes);
    cache.ids_ = view.arena_shape.scratch_bytes ? selection.candidate_ids : nullptr;
    typename kit_::norms_t* const candidate_norms = reinterpret_cast<typename kit_::norms_t*>(
        selection.end(selection_nodes));
    schedule_ schedule {schedule_states[warp], lane};
    u32_t const connectivity = view.storage.connectivity;
    while (true) {
        u32_t node_id = lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_nodes, 1u) : 0;
        node_id = view.first_id + warp_t::shuffle(node_id, 0);
        if (node_id >= view.end_id) return;
        // A row the pass neither walked nor picked keeps its neighbors.
        bool const walked = vram_bitset_contains<warp_t>(view.walked, node_id);
        if (!walked && view.link_heads[node_id] == node_id_unknown_k) continue;

        vram_telemetry_t telemetry;
        u32_t const* const forward_picks = walked ? vram_row(view.shadow_neighbors, node_id, connectivity) : nullptr;
        bool const gained = vram_back_link_one<kit_, schedule_, resident_blocks_>( //
            view.storage, node_id, forward_picks, view.shadow_neighbors, view.link_heads[node_id], view.link_next, 0,
            cache, selection_nodes - 1, schedule, candidate_norms, selection, gathered[warp], telemetry, lane);
        telemetry.changed_nodes += walked && gained;
        telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
    }
}

/** One warp folds one row of the launching shard at a time, assigning itself rows from the pass
 *  counter: a re-selection reads every shard of the bin and writes only its own rows' fold rows,
 *  picks and links, and a merge writes only its own rows' fold rows. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_>
__device__ __forceinline__ void vram_fold(vram_fold_view<kit_> const& view) {
    using warp_t = typename schedule_::warp_t;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];
    __shared__ typename schedule_::shared_t schedule_states[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    u32_t const selection_nodes = view.selection_nodes;
    vram_vector_cache<typename kit_::fragment_t> cache {reinterpret_cast<typename kit_::fragment_t*>(slice.scratch),
                                                        view.bin.geometry.row_fragments};
    vram_selection_scratch<node_id_t> const selection = carve_selection_scratch<node_id_t>(cache, selection_nodes);
    cache.ids_ = view.arena_shape.scratch_bytes ? selection.candidate_ids : nullptr;
    typename kit_::norms_t* const candidate_norms = reinterpret_cast<typename kit_::norms_t*>(
        selection.end(selection_nodes));
    schedule_ schedule {schedule_states[warp], lane};
    u32_t const rows = view.bin.shards[view.shard].rows;
    node_id_t const base = view.shard << view.bin.row_bits;
    auto const assign = [&]() noexcept -> u32_t {
        u32_t const row = lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_nodes, 1u) : 0;
        return warp_t::shuffle(row, 0);
    };
    // The phase is a launch argument, so the branch sits outside the loop every warp spins in.
    if (view.phase == vram_fold_phase_t::reselect_k)
        for (u32_t row = assign(); row < rows; row = assign()) {
            vram_telemetry_t telemetry;
            vram_fold_reselect_one<kit_, schedule_, resident_blocks_>( //
                view, base | row, cache, schedule, selection, candidate_norms, gathered[warp], telemetry, lane);
            telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
        }
    else
        for (u32_t row = assign(); row < rows; row = assign()) {
            vram_telemetry_t telemetry;
            telemetry.changed_nodes += vram_fold_merge_one<kit_, schedule_, resident_blocks_>( //
                view, base | row, cache, schedule, selection, candidate_norms, gathered[warp], telemetry, lane);
            telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
        }
}

/** Rebuilds the norms column from the vectors matrix for rows @c [first_row,first_row+rows_count),
 *  one warp per row; norms are derived state, never serialized. */
template <vram_quads_kit kit_, typename warp_type_>
__device__ __forceinline__ void vram_derive_norms(vram_flat_view<kit_> const& storage, u32_t first_row,
                                                  u32_t rows_count) {
    using warp_t = warp_type_;
    vram_grid_warp_t const warp = vram_grid_warp<warp_t>();
    for (u32_t row = first_row + warp.index; row < first_row + rows_count; row += warp.count) {
        typename kit_::norms_accumulator_t accumulator {};
        for (u32_t fragment = warp.lane; fragment < storage.geometry.row_fragments; fragment += warp_t::width_k)
            accumulator = kit_::accumulate_norms(storage.vector_for(row, fragment), accumulator);
        accumulator = vram_reduce_norms<warp_t>(accumulator);
        if (warp.lane == 0) storage.norm_ref_for(row) = kit_::partials_to_norms(accumulator);
    }
}

#pragma endregion Builders

#pragma region Walks

/** Grouped-query heads as a device rescoring: the walk navigates by a token's first head and keeps
 *  its dots, the warp reads the batch once more for every further head, every head's dots go into
 *  the warp's scratch for the attend fold, and a node's distance is 1 − maxₕ qₕ · k, so every
 *  head's own region is explored. */
template <vram_quads_kit kit_, typename warp_type_>
struct vram_heads_rescoring {
    using warp_t = warp_type_;
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;

    /** Every token's heads as staged rows, @b [tokens·heads,row_fragments], device-resident. */
    fragment_t const* heads;

    /** The walking warp's scratch, opening with the batch dots, @b [heads,lanes]; set per walk. */
    f32_t* dots;

    /** Heads per token. */
    u32_t heads_count;

    /** Fragments per staged head row. */
    u32_t row_fragments;

    template <vram_frontier frontier_>
    __device__ __forceinline__ vram_heads_rescoring
    for_query(u32_t query_index, vram_warp_context<frontier_> const& context) const noexcept {
        vram_heads_rescoring narrowed = *this;
        narrowed.heads = vram_row(heads, query_index, heads_count * row_fragments);
        narrowed.dots = reinterpret_cast<f32_t*>(context.scratch);
        return narrowed;
    }

    /** Scores the batch against every head past the first as
     *  @ref unum::usearch::dense::vram_score_batch scores it against the query: a sub-warp of
     *  @c lanes_per_row lanes owns one row, and each key fragment it loads meets every head, so no
     *  load window reads @p resident_blocks_. */
    template <u32_t resident_blocks_>
    __device__ __forceinline__ vram_scored_node<node_id_t> rescore(vram_flat_view<kit_> const& storage,
                                                                   vram_gathered<kit_, warp_t>& gathered,
                                                                   u32_t gathered_count, u32_t lane,
                                                                   vram_scored_node<node_id_t> scored) const noexcept {
        constexpr u32_t lanes_k = warp_t::width_k;
        u32_t const lanes_log2 = 31 - __clz(storage.geometry.lanes_per_row);
        u32_t const lane_in_row = lane & (storage.geometry.lanes_per_row - 1);
        bool const live = lane < gathered_count;
        if (live) dots[lane] = static_cast<f32_t>(gathered.dots[lane]);
        for (u32_t row_base = 0; heads_count != 1 && row_base < gathered_count; row_base += lanes_k >> lanes_log2) {
            u32_t const row = row_base + (lane >> lanes_log2);
            for (u32_t step = 0; step != storage.geometry.steps_per_row; ++step) {
                u32_t const fragment_index = lane_in_row + (step << lanes_log2);
                bool const loads = row < gathered_count && fragment_index < storage.geometry.row_fragments;
                fragment_t const key = loads ? storage.vector_for(gathered.ids[row], fragment_index) : fragment_t {};
                for (u32_t head = 1; head != heads_count; ++head) {
                    fragment_t const query = loads ? heads[head * row_fragments + fragment_index] : fragment_t {};
                    dot_t const dot = vram_reduce<warp_t>(kit_::accumulate_dot(key, query, dot_t {}),
                                                          storage.geometry.lanes_per_row);
                    f32_t& slot = dots[head * lanes_k + row];
                    if (lane_in_row == 0 && row < gathered_count) slot = (step ? slot : 0.f) + static_cast<f32_t>(dot);
                }
            }
        }
        warp_t::sync();
        if (!live) return scored;
        f32_t best = dots[lane];
        for (u32_t head = 1; head != heads_count; ++head) best = fmaxf(best, dots[head * lanes_k + lane]);
        scored = vram_scored_node<node_id_t>::make(1 - best, gathered.ids[lane]);
        gathered.scored[lane] = scored;
        return scored;
    }
};

/**
 *  @brief Adds one batch's @p count value rows into @p heads_count accumulator rows in the
 *      warp's scratch: row h of @p accumulators is scaled by rescalesₕ, then takes Σᵣ
 *      weightsₕᵣ · rowᵣ, @p weights being @b [heads,lanes].
 *
 *  Each lane brings its own batch row's start in @p row and owns the accumulator fragments
 *  congruent to it modulo the warp's width: it reads every row's fragment there once and adds it
 *  into every head, so the rows accumulate in place without atomics, and a row's last, partial
 *  fragment is read whole bytes at a time.
 */
template <vram_quads_kit kit_, typename warp_type_>
__device__ __forceinline__ void vram_gather_weighted_rows( //
    typename kit_::fragment_t const* row, u32_t count, f32_t const* weights, f32_t const* rescales, u32_t heads_count,
    u32_t columns, f32_t* accumulators, u32_t lane) noexcept {
    using warp_t = warp_type_;
    using fragment_t = typename kit_::fragment_t;
    constexpr u32_t lanes_k = warp_t::width_k;
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
                warp_t::shuffle(reinterpret_cast<std::uintptr_t>(row), source_lane));
            if (!fragment_columns) continue;
            fragment_t value {};
            if (fragment_columns == columns_per_fragment_k) value = source[fragment];
            else
                warp_t::copy_bytes(&value, source + fragment,
                                   divide_round_up(fragment_columns * kit_::scalar_t::bits_per_dimension(), 8u));
            for (u32_t head = 0; head != heads_count; ++head) {
                f32_t const weight = weights[head * lanes_k + source_lane];
                if (weight == 0) continue;
                f32_t* const slot = slots + head * columns;
                // The sums round-trip through the scratch row, fully unrolled so they never go to
                // local or private memory.
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

/** Resolves @p node_id through the shard's device key column, then the caller's key-level lookup,
 *  whose own @c vram_value_row overload sits beside its record. */
template <typename key_type_, typename value_lookup_type_>
__device__ __forceinline__ void const* vram_value_row(vram_keyed_lookup<key_type_, value_lookup_type_> const& lookup,
                                                      node_id_t node_id) noexcept {
    return vram_value_row(lookup.value_lookup, lookup.keys[node_id]);
}

/**
 *  @brief One token's attention as a walk fold, FlashAttention-style over every node the walk
 *      scores.
 *
 *  The per-head dots @ref vram_heads_rescoring left in the warp's scratch become base-2 weights
 *  against each head's running maximum, and every value row the lookup resolves is read once and
 *  added into every head's accumulator row. The running maxima, partitions and rescales are
 *  warp-uniform, as every lane derives them from the same reductions.
 */
template <vram_quads_kit kit_, typename warp_type_, typename value_lookup_type_>
struct vram_attend_fold {
    using warp_t = warp_type_;
    using fragment_t = typename kit_::fragment_t;

    /** Resolves a node ID to its value row's start, 16-byte aligned, or to null when it has
     *  none. */
    value_lookup_type_ value_lookup;

    /** The warp's scratch, carved in order: the batch dots @b [heads,lanes], each head's running
     *  base-2 maximum, partition and latest rescale, then the accumulator rows
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

    /** Whether the walk visited a node the lookup had no row for, identical on every lane. */
    bool missed = false;

    __device__ __forceinline__ walk_demand_t absorb(vram_gathered<kit_, warp_t> const& gathered, u32_t gathered_count,
                                                    u32_t lane) noexcept {
        constexpr u32_t lanes_k = warp_t::width_k;
        bool const live = lane < gathered_count;
        fragment_t const* const row =
            live ? static_cast<fragment_t const*>(vram_value_row(value_lookup, gathered.ids[lane])) : nullptr;
        // A node without its value row would leave a wrong softmax, so the walk stops there and the
        // call refuses.
        missed = warp_t::any(live && !row);
        if (missed) return walk_demand_t::sated_k;
        for (u32_t head = 0; head != heads_count; ++head) {
            f32_t const logit = live ? logit_scale * dots[head * lanes_k + lane] : std::numeric_limits<f32_t>::lowest();
            f32_t batch_maximum = logit;
            for (u32_t offset = lanes_k / 2; offset; offset >>= 1)
                batch_maximum = fmaxf(batch_maximum, warp_t::shuffle_xor(batch_maximum, offset));
            f32_t const previous_maximum = maxima[head];
            f32_t const maximum = fmaxf(previous_maximum, batch_maximum);
            f32_t const weight = live ? exp2f(logit - maximum) : 0;
            f32_t batch_partition = weight;
            for (u32_t offset = lanes_k / 2; offset; offset >>= 1)
                batch_partition += warp_t::shuffle_xor(batch_partition, offset);
            f32_t const rescale = exp2f(previous_maximum - maximum);
            if (live) dots[head * lanes_k + lane] = weight;
            // Every lane reads the previous state before the leader replaces it.
            warp_t::sync();
            if (lane == 0) {
                maxima[head] = maximum;
                partitions[head] = partitions[head] * rescale + batch_partition;
                rescales[head] = rescale;
            }
        }
        warp_t::sync();
        vram_gather_weighted_rows<kit_, warp_t>(row, gathered_count, dots, rescales, heads_count, columns, accumulators,
                                                lane);
        return walk_demand_t::continue_k;
    }
};

/** Writes each token's attention out of its walk: opens the fold over the warp's scratch, then
 *  writes every head's normalized row and its carry once the walk ends, and raises the miss word if
 *  the walk visited a row-less node. */
template <vram_quads_kit kit_, typename warp_type_, typename value_lookup_type_>
struct vram_attend_sink {
    using warp_t = warp_type_;
    using fold_t = vram_attend_fold<kit_, warp_type_, value_lookup_type_>;

    /** Resolves a node ID to its value row, device-callable. */
    value_lookup_type_ value_lookup;

    /** The attended rows, @b [tokens·heads,columns], device-resident. */
    f32_t* outputs;

    /** Each row's softmax carry, @b [tokens·heads], device-resident. */
    softmax_state_t* states;

    /** Set by any walk that visited a node the lookup had no row for; zeroed, device-resident. */
    u32_t* missed;

    /** The factor turning a dot into a base-2 logit, @c score_scale · log₂e. */
    f32_t logit_scale;

    /** Heads per token, and columns per value and output row. */
    u32_t heads_count, columns;

    template <vram_frontier frontier_>
    __device__ __forceinline__ fold_t open(u32_t, vram_warp_context<frontier_> const& context,
                                           u32_t lane) const noexcept {
        constexpr u32_t lanes_k = warp_t::width_k;
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
        // The first batch rescales the accumulators by zero, which a NaN left in the scratch would
        // survive.
        for (u32_t slot = lane; slot < heads_count * columns; slot += lanes_k) accumulators[slot] = 0;
        warp_t::sync();
        return fold;
    }

    template <vram_frontier frontier_>
    __device__ __forceinline__ void close(fold_t const& fold, vram_warp_context<frontier_>&, u32_t query_index,
                                          u32_t lane) const noexcept {
        constexpr u32_t lanes_k = warp_t::width_k;
        if (fold.missed && lane == 0) warp_t::store_relaxed(*missed, 1u);
        // The lanes accumulated by fragment and write out by column, so the rows cross lanes here.
        warp_t::sync();
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

/** One warp per query over the register frontier - expansions within the ladder - each walk
 *  scored through the schedule under @p rescorings and written out through @p sink; warps take
 *  queries from the batch counter, as builders take rows. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t candidate_slots_, u32_t resident_blocks_,
          typename sink_, typename rescoring_>
__device__ __forceinline__ void vram_walk(vram_search_view<kit_> const& view, sink_ const& sink,
                                          rescoring_ const& rescorings) {
    using warp_t = typename schedule_::warp_t;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    while (true) {
        u32_t query_index = slice.lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_queries, 1u) : 0;
        query_index = warp_t::shuffle(query_index, 0);
        if (query_index >= view.queries_count) return;
        vram_visited_set<warp_t, node_id_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                     view.epoch_base + query_index + 1};
        vram_walk_one<kit_, schedule_, resident_blocks_>( //
            view, sink, rescorings, query_index, query_span, visited,
            vram_register_frontier<candidate_slots_, warp_t, node_id_t> {}, gathered[slice.warp], slice.lane);
    }
}

/** One warp per query past the register ladder's width: the same walk over the arena frontier,
 *  whose budget is the requested expansion itself, ranking everything it kept at the end. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t resident_blocks_, typename sink_,
          typename rescoring_>
__device__ __forceinline__ void vram_walk_wide(vram_search_view<kit_> const& view, sink_ const& sink,
                                               rescoring_ const& rescorings) {
    using warp_t = typename schedule_::warp_t;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ vram_gathered<kit_, warp_t> gathered[vram_warps_per_block_k];

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_t>(view, dynamic_scratch);
    typename kit_::fragment_t* const query_span = reinterpret_cast<typename kit_::fragment_t*>(slice.scratch);
    vram_arena_frontier<warp_t, node_id_t> frontier;
    frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_t>(view.arena, slice.warp_index);
    frontier.tail_.capacity_ = view.arena_shape.tail_entries;
    frontier.expansion_target_ = view.expansion;
    while (true) {
        u32_t query_index = slice.lane == 0 ? warp_t::fetch_add_relaxed(*view.assigned_queries, 1u) : 0;
        query_index = warp_t::shuffle(query_index, 0);
        if (query_index >= view.queries_count) return;
        vram_visited_set<warp_t, node_id_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                     view.epoch_base + query_index + 1};
        vram_walk_one<kit_, schedule_, resident_blocks_>(view, sink, rescorings, query_index, query_span, visited,
                                                         frontier, gathered[slice.warp], slice.lane);
    }
}

/** Runs one walk family on its rung, the arena frontier at slot zero, under @p resident_blocks_
 *  blocks per multiprocessor. */
template <vram_quads_kit kit_, vram_schedule<kit_> schedule_, u32_t slots_, u32_t resident_blocks_, typename sink_,
          typename rescoring_>
__device__ __forceinline__ void vram_walk_rung(vram_search_view<kit_> const& view, sink_ const& sink,
                                               rescoring_ const& rescorings) {
    if constexpr (slots_ == 0) vram_walk_wide<kit_, schedule_, resident_blocks_>(view, sink, rescorings);
    else vram_walk<kit_, schedule_, slots_, resident_blocks_>(view, sink, rescorings);
}

/** Merges the normalized attended rows and carries of @p shards walks into the first's by the flash
 *  rule, a warp per row: each rescales to the joint maximum and weighs by its share of the joint
 *  partition. Every shard holds a block of @b [rows,columns] rows, then @b [rows] carries, the
 *  blocks lying back to back from @p blocks. */
template <typename warp_type_>
__device__ __forceinline__ void vram_merge_attention(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards) {
    constexpr u32_t lanes_k = warp_type_::width_k;
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
    warp_type_::sync();
    if (lane == 0) carry_of(0) = {maximum, partition};
}

#pragma endregion Walks

#pragma region Audits

/** Marks @p seed_count stride-spread rows reached and in the first frontier - the audit's landmark
 *  set, spread over the ID space like the traversal samplers. */
template <typename warp_type_>
__device__ __forceinline__ void vram_mark_landmarks(vram_bitset_t reached, vram_bitset_t frontier, u32_t rows_count,
                                                    u32_t seed_count) {
    u32_t const seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= seed_count) return;
    u32_t const row = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    vram_bitset_insert<warp_type_>(reached, row);
    vram_bitset_insert<warp_type_>(frontier, row);
}

/** One level of forward reachability: every row of @p frontier marks its targets reached, and the
 *  ones it reaches first enter @p next_frontier. Only the frontier expands, so a level is the same
 *  set of rows however the warps interleave, and sweeping until a level expands nothing takes the
 *  landmarks' eccentricity in levels; a level whose @p expanded_before reads zero was queued past
 *  convergence and has nothing to expand. The first expansion of a level counts it in @p levels, so
 *  the host reads the depth from one word. */
template <typename warp_type_>
__device__ __forceinline__ void vram_reachability_sweep(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                        vram_bitset_t reached, vram_bitset_t frontier,
                                                        vram_bitset_t next_frontier, u32_t* expanded,
                                                        u32_t const* expanded_before, u32_t* levels) {
    // The level before finished ahead of this launch, so every thread reads one settled word.
    if (expanded_before && !*expanded_before) return;
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!vram_bitset_contains<warp_type_>(frontier, row)) continue;
        if (warp.lane == 0 && !warp_type_::fetch_add_relaxed(*expanded, 1u)) warp_type_::fetch_add_relaxed(*levels, 1u);
        vram_for_each_link<warp_type_>(neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
            if (vram_bitset_insert<warp_type_>(reached, target)) vram_bitset_insert<warp_type_>(next_frontier, target);
        });
    }
}

/** One level of reachability back toward the landmarks: every row outside @p reaches_back with a
 *  target in @p back_settled, the rows reaching back before this level, enters it. No reverse
 *  adjacency is built, so a warp reads each such row's own links, and a level is the same set of
 *  rows however the warps interleave. Each row it marks counts in @p expanded beside the forward
 *  sweep's, so one counter ends both directions, and stamps @p sweep_number into @p back_levels;
 *  a level whose @p expanded_before reads zero was queued past convergence. */
template <typename warp_type_>
__device__ __forceinline__ void vram_reachability_back_sweep(u32_t const* neighbors, u32_t rows_count,
                                                             u32_t connectivity, vram_bitset_t reaches_back,
                                                             vram_bitset_t back_settled, u32_t* expanded,
                                                             u32_t const* expanded_before, u32_t* back_levels,
                                                             u32_t sweep_number) {
    if (expanded_before && !*expanded_before) return;
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        // Only this warp sets the row's bit, so every lane reads the same answer.
        if (vram_bitset_contains<warp_type_>(reaches_back, row)) continue;
        bool settled_target = false;
        vram_for_each_link<warp_type_>(neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
            settled_target = settled_target || vram_bitset_contains<warp_type_>(back_settled, target);
        });
        if (!warp_type_::any(settled_target) || warp.lane != 0) continue;
        vram_bitset_insert<warp_type_>(reaches_back, row);
        warp_type_::fetch_add_relaxed(*expanded, 1u);
        // Every marking warp of the level stores the same value.
        warp_type_::store_relaxed(*back_levels, sweep_number);
    }
}

/** Counts the islands - the rows @p reached lacks, or @p reaches_back lacks when the audit swept
 *  back - in @p islands_cursor, collecting their IDs into @p island_ids when it is given, and the
 *  unreached among them in @p unreached_cursor when it is given. */
template <typename warp_type_>
__device__ __forceinline__ void vram_reachability_islands(vram_bitset_t reached, u32_t rows_count, u32_t* island_ids,
                                                          u32_t* islands_cursor, vram_bitset_t reaches_back = {},
                                                          u32_t* unreached_cursor = nullptr) {
    u32_t const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows_count) return;
    bool const unreached = !vram_bitset_contains<warp_type_>(reached, row);
    if (unreached && unreached_cursor) warp_type_::fetch_add_relaxed(*unreached_cursor, 1u);
    if (!unreached && (!reaches_back.words || vram_bitset_contains<warp_type_>(reaches_back, row))) return;
    u32_t const slot = warp_type_::fetch_add_relaxed(*islands_cursor, 1u);
    if (island_ids) island_ids[slot] = row;
}

/** Counts, for every row, the reached rows that point at it - the links a reconnection may take
 *  from a reached resident only while another one remains. */
template <typename warp_type_>
__device__ __forceinline__ void vram_reached_in_links(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                      vram_bitset_t reached, u32_t* in_links) {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!vram_bitset_contains<warp_type_>(reached, row)) continue;
        vram_for_each_link<warp_type_>(neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
            warp_type_::fetch_add_relaxed(in_links[target], 1u);
        });
    }
}

/** Reconnects each island, bypassing the diversity judgment that dropped its links. An unreached
 *  island links into the first of its reached forward picks with room for it, or into the landmark
 *  covering its ID. One reaching no landmark back, when the audit swept back, takes an out-link in
 *  its own row to the nearest of the heal's @p shadow_neighbors picks that does, or to that
 *  landmark, unless a live pick already does. A link takes a free slot when one is left, otherwise
 *  a slot spread across the tail. Every link is a compare-and-swap that never displaces an
 *  unreached resident of a reached row, the round's sibling moves, nor a reached resident whose own
 *  @p in_links would drop to zero, so no reconnection leaves another node an island; @p reconnected
 *  counts the links actually written. Exclusive pass: the launch boundary publishes every write, so
 *  each access is relaxed. */
template <typename warp_type_>
__device__ __forceinline__ void vram_reconnect_island(u32_t* neighbors, u32_t connectivity, u32_t const* island_ids,
                                                      u32_t islands_count, vram_bitset_t reached, u32_t rows_count,
                                                      u32_t seed_count, u32_t* in_links, u32_t* reconnected,
                                                      vram_bitset_t reaches_back = {},
                                                      u32_t* shadow_neighbors = nullptr) {
    u32_t const island_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (island_index >= islands_count) return;
    u32_t const island = island_ids[island_index];
    u32_t const spread = heal_spread(connectivity);

    // Whether `linked` now sits in the owner's row, placing it there when a slot allows.
    auto const attach = [&](u32_t owner, u32_t linked) noexcept {
        u32_t* const owner_row = vram_row(neighbors, owner, connectivity);
        bool const owner_reached = vram_bitset_contains<warp_type_>(reached, owner);
        auto const claim = [&](u32_t slot, u32_t resident) noexcept {
            if (!warp_type_::compare_exchange_relaxed(owner_row[slot], resident, linked)) return false;
            warp_type_::fetch_add_relaxed(*reconnected, 1u);
            return true;
        };
        for (u32_t slot = 0; slot != connectivity; ++slot) {
            u32_t const resident = warp_type_::load_relaxed(owner_row[slot]);
            if (resident == linked) return true; // Already wired; the audit raced a heal.
            if (resident != node_id_unknown_k && resident != owner) continue;
            if (claim(slot, resident)) return true;
        }
        for (u32_t attempt = 0; attempt != spread; ++attempt) {
            u32_t const target_slot = connectivity - 1 - (linked + attempt) % spread;
            u32_t const resident = warp_type_::load_relaxed(owner_row[target_slot]);
            if (resident == linked) return true;
            // An unreached row proves no in-link, and only its own island writes it.
            if (!owner_reached) {
                if (claim(target_slot, resident)) return true;
                continue;
            }
            if (resident != node_id_unknown_k && !vram_bitset_contains<warp_type_>(reached, resident))
                continue; // A sibling island linked first.
            // A reached resident gives this link up only while another reached row still points at
            // it.
            bool const evicting = resident != node_id_unknown_k && resident != owner;
            if (evicting && warp_type_::fetch_sub_relaxed(in_links[resident], 1u) <= 1u) {
                warp_type_::fetch_add_relaxed(in_links[resident], 1u);
                continue;
            }
            if (claim(target_slot, resident)) return true;
            if (evicting) warp_type_::fetch_add_relaxed(in_links[resident], 1u);
        }
        return false;
    };
    // The first of the row's picks in `bitset`, or the sentinel.
    auto const first_pick_in = [&](u32_t* picks, vram_bitset_t bitset) noexcept {
        for (u32_t slot = 0; slot != connectivity; ++slot) {
            u32_t const pick = warp_type_::load_relaxed(picks[slot]);
            if (pick != node_id_unknown_k && pick != island && pick < rows_count &&
                vram_bitset_contains<warp_type_>(bitset, pick))
                return pick;
        }
        return node_id_unknown_k;
    };

    // A walk starts at the landmark covering this ID by definition, which is what makes it reached
    // and reaching back.
    u32_t landmark = island;
    if (seed_count) {
        u32_t const seed = static_cast<u32_t>(static_cast<u64_t>(island) * seed_count / rows_count);
        landmark = static_cast<u32_t>(static_cast<u64_t>(seed) * rows_count / seed_count);
    }
    u32_t* const island_row = vram_row(neighbors, island, connectivity);
    if (!vram_bitset_contains<warp_type_>(reached, island)) {
        bool attached = false;
        for (u32_t slot = 0; slot != connectivity && !attached; ++slot) {
            u32_t const pick = island_row[slot];
            attached = pick != node_id_unknown_k && pick != island && vram_bitset_contains<warp_type_>(reached, pick) &&
                       attach(pick, island);
        }
        if (!attached && landmark != island) attach(landmark, island);
    }
    if (!reaches_back.words || vram_bitset_contains<warp_type_>(reaches_back, island)) return;
    if (first_pick_in(island_row, reaches_back) != node_id_unknown_k) return;
    u32_t const nearest = shadow_neighbors
                              ? first_pick_in(vram_row(shadow_neighbors, island, connectivity), reaches_back)
                              : node_id_unknown_k;
    if (nearest != node_id_unknown_k) attach(island, nearest);
    else if (landmark != island) attach(island, landmark);
}

#pragma endregion Audits

#pragma region Splits

/** Places the first @p roots_count parts of a split: part @c p takes row @p roots[p], alone in
 *  @p frontier. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_roots(u32_t const* roots, u32_t roots_count, u32_t* node_parts,
                                                     vram_bitset_t frontier, u32_t* part_sizes) noexcept {
    u32_t const part = blockIdx.x * blockDim.x + threadIdx.x;
    if (part >= roots_count) return;
    node_parts[roots[part]] = part;
    vram_bitset_insert<warp_type_>(frontier, roots[part]);
    part_sizes[part] = 1;
}

/** A growth round's first pass, one warp per frontier row: a row of a part with room proposes
 *  itself to every unplaced row it links to, and each keeps its lowest-ID proposer, so no warp
 *  order sways the round. A round whose @p placed_before reads zero follows one that placed nothing
 *  and leaves at entry, as every later pass of it does. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_propose(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                       u32_t const* node_parts, u32_t* proposals,
                                                       vram_bitset_t frontier, u32_t const* part_sizes,
                                                       u32_t part_capacity, u32_t const* placed_before) noexcept {
    if (placed_before && !*placed_before) return;
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        if (!vram_bitset_contains<warp_type_>(frontier, row) || part_sizes[node_parts[row]] >= part_capacity) continue;
        vram_for_each_link<warp_type_>(neighbors, connectivity, rows_count, row, warp.lane, [&](u32_t target) noexcept {
            if (node_parts[target] != node_id_unknown_k) return;
            for (u32_t held = warp_type_::load_relaxed(proposals[target]); row < held;
                 held = warp_type_::load_relaxed(proposals[target]))
                if (warp_type_::compare_exchange_relaxed(proposals[target], held, row)) return;
        });
    }
}

/** A growth round's second pass, one lane per row: counts the rows proposed to each part in
 *  @p round_counts, the lanes of a warp proposed to one part adding at once. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_count(u32_t const* node_parts, u32_t const* proposals, u32_t rows_count,
                                                     u32_t* round_counts, u32_t const* placed_before) noexcept {
    if (placed_before && !*placed_before) return;
    u32_t const lane = threadIdx.x % warp_type_::width_k, threads = gridDim.x * blockDim.x;
    // The trip count is warp-uniform, so every lane enters each vote.
    for (u32_t first = blockIdx.x * blockDim.x + threadIdx.x - lane; first < rows_count; first += threads) {
        u32_t const row = first + lane;
        u32_t const proposal = row < rows_count ? proposals[row] : node_id_unknown_k;
        u32_t const part = proposal != node_id_unknown_k ? node_parts[proposal] : node_id_unknown_k;
        typename warp_type_::mask_t const peers = warp_type_::match_any(part);
        if (part != node_id_unknown_k && warp_type_::first(peers) == lane)
            warp_type_::fetch_add_relaxed(round_counts[part], warp_type_::count(peers));
    }
}

/** A growth round's third pass, one lane per row: places every row proposed to a part with room for
 *  the round's whole count, in @p next_frontier, counting them in @p placed. The rows a round count
 *  would overflow stay proposed for @ref unum::usearch::dense::vram_partition_fill. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_accept(u32_t* node_parts, u32_t* proposals, u32_t rows_count,
                                                      vram_bitset_t next_frontier, u32_t const* part_sizes,
                                                      u32_t const* round_counts, u32_t part_capacity, u32_t* placed,
                                                      u32_t const* placed_before) noexcept {
    if (placed_before && !*placed_before) return;
    u32_t const lane = threadIdx.x % warp_type_::width_k, threads = gridDim.x * blockDim.x;
    for (u32_t first = blockIdx.x * blockDim.x + threadIdx.x - lane; first < rows_count; first += threads) {
        u32_t const row = first + lane;
        u32_t const proposal = row < rows_count ? proposals[row] : node_id_unknown_k;
        // Proposers were placed in earlier rounds, so no lane rewrites a label another one reads.
        u32_t const part = proposal != node_id_unknown_k ? node_parts[proposal] : node_id_unknown_k;
        bool const placing = part != node_id_unknown_k && round_counts[part] <= part_capacity - part_sizes[part];
        if (placing) {
            node_parts[row] = part;
            proposals[row] = node_id_unknown_k;
            vram_bitset_insert<warp_type_>(next_frontier, row);
        }
        typename warp_type_::mask_t const placed_lanes = warp_type_::ballot(placing);
        if (lane == 0 && placed_lanes) warp_type_::fetch_add_relaxed(*placed, warp_type_::count(placed_lanes));
    }
}

/** A growth round's last pass, one block per part: a part whose round count fit takes it into its
 *  size, and one the round count overflows places its lowest-ID proposed rows up to its room, in
 *  @p next_frontier, and fills up. Either way the part's proposals and round count clear for the
 *  round that follows. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_fill(u32_t* node_parts, u32_t* proposals, u32_t rows_count,
                                                    vram_bitset_t next_frontier, u32_t* part_sizes, u32_t* round_counts,
                                                    u32_t part_capacity, u32_t* placed,
                                                    u32_t const* placed_before) noexcept {
    if (placed_before && !*placed_before) return;
    __shared__ u32_t warp_counts[vram_warps_per_block_k];
    u32_t const part = blockIdx.x, lane = threadIdx.x % warp_type_::width_k, warp = threadIdx.x / warp_type_::width_k;
    u32_t const warps = blockDim.x / warp_type_::width_k;
    u32_t const wanted = round_counts[part], room = part_capacity - part_sizes[part];
    __syncthreads();
    if (wanted > room)
        for (u32_t first = 0, taken = 0; first < rows_count; first += blockDim.x) {
            u32_t const row = first + threadIdx.x;
            u32_t const proposal = row < rows_count ? proposals[row] : node_id_unknown_k;
            bool const proposed_here = proposal != node_id_unknown_k && node_parts[proposal] == part;
            typename warp_type_::mask_t const voted = warp_type_::ballot(proposed_here);
            if (lane == 0) warp_counts[warp] = warp_type_::count(voted);
            __syncthreads();
            u32_t rank = taken + warp_type_::count(voted & warp_type_::below(lane)), total = 0;
            for (u32_t other = 0; other != warps; ++other) {
                if (other < warp) rank += warp_counts[other];
                total += warp_counts[other];
            }
            if (proposed_here) {
                if (rank < room) {
                    node_parts[row] = part;
                    vram_bitset_insert<warp_type_>(next_frontier, row);
                }
                proposals[row] = node_id_unknown_k;
            }
            taken += total;
            __syncthreads();
        }
    if (threadIdx.x != 0) return;
    part_sizes[part] += wanted > room ? room : wanted;
    round_counts[part] = 0;
    if (wanted > room) warp_type_::fetch_add_relaxed(*placed, room);
}

/** Places every row no growth round reached, one block walking them in ID order: each enters the
 *  part smallest at its turn, the lowest-indexed among equals. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_islands(u32_t* node_parts, u32_t rows_count, u32_t* part_sizes,
                                                       u32_t parts_count) noexcept {
    __shared__ u32_t warp_counts[vram_warps_per_block_k];
    __shared__ u32_t picks[vram_warps_per_block_k * warp_type_::width_k];
    u32_t const lane = threadIdx.x % warp_type_::width_k, warp = threadIdx.x / warp_type_::width_k;
    u32_t const warps = blockDim.x / warp_type_::width_k;
    for (u32_t first = 0; first < rows_count; first += blockDim.x) {
        u32_t const row = first + threadIdx.x;
        bool const unplaced = row < rows_count && node_parts[row] == node_id_unknown_k;
        typename warp_type_::mask_t const voted = warp_type_::ballot(unplaced);
        if (lane == 0) warp_counts[warp] = warp_type_::count(voted);
        __syncthreads();
        u32_t rank = warp_type_::count(voted & warp_type_::below(lane)), total = 0;
        for (u32_t other = 0; other != warps; ++other) {
            if (other < warp) rank += warp_counts[other];
            total += warp_counts[other];
        }
        if (threadIdx.x == 0)
            for (u32_t pick = 0; pick != total; ++pick) {
                u32_t smallest = 0;
                for (u32_t part = 1; part != parts_count; ++part)
                    if (part_sizes[part] < part_sizes[smallest]) smallest = part;
                ++part_sizes[smallest];
                picks[pick] = smallest;
            }
        __syncthreads();
        if (unplaced) node_parts[row] = picks[rank];
        __syncthreads();
    }
}

/** Marks in @p picked every row linking into another part, one warp per row. */
template <typename warp_type_>
__device__ __forceinline__ void vram_partition_picked(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                      u32_t const* node_parts, vram_bitset_t picked) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        u32_t const part = node_parts[row];
        bool across = false;
        vram_for_each_link<warp_type_>(neighbors, connectivity, rows_count, row, warp.lane,
                                       [&](u32_t target) noexcept { across = across || node_parts[target] != part; });
        if (warp_type_::any(across) && warp.lane == 0) vram_bitset_insert<warp_type_>(picked, row);
    }
}

/** Moves every row's neighbors into its part's receiver, one warp per row, at the row its slot
 *  names past the part's first slot, each part @p part_capacity slots wide: the links inside the
 *  part rebased through their own slots and packed to the front, the links across dropped, and a
 *  row nothing survived in pointing at itself. */
template <typename warp_type_>
__device__ __forceinline__ void vram_split_links(u32_t const* neighbors, u32_t rows_count, u32_t connectivity,
                                                 u32_t const* slots, u32_t part_capacity,
                                                 u32_t* const* part_neighbors) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        u32_t const part = slots[row] / part_capacity, position = slots[row] % part_capacity;
        u32_t const* const from = vram_row(neighbors, row, connectivity);
        u32_t* const into = vram_row(part_neighbors[part], position, connectivity);
        u32_t kept = 0;
        for (u32_t base = 0; base < connectivity; base += warp_type_::width_k) { // Warp-uniform trip count.
            u32_t const slot = base + warp.lane;
            u32_t const id = slot < connectivity ? from[slot] : node_id_unknown_k;
            u32_t const target = id < rows_count && id != row ? slots[id] : node_id_unknown_k;
            bool const inside = target != node_id_unknown_k && target / part_capacity == part;
            typename warp_type_::mask_t const survivors = warp_type_::ballot(inside);
            if (inside)
                into[kept + warp_type_::count(survivors & warp_type_::below(warp.lane))] = target % part_capacity;
            kept += warp_type_::count(survivors);
        }
        for (u32_t slot = kept + warp.lane; slot < connectivity; slot += warp_type_::width_k)
            into[slot] = node_id_unknown_k;
        if (!kept && warp.lane == 0) into[0] = position;
    }
}

/** Moves every row's ragged value into its part's receiver, one warp per row, from the bytes
 *  @p offsets bound into the receiver's tape at its own @p part_offsets, laid out in slot order. */
template <typename warp_type_>
__device__ __forceinline__ void vram_split_tape(byte_t const* values, u64_t const* offsets, u32_t rows_count,
                                                u32_t const* slots, u32_t part_capacity, byte_t* const* part_values,
                                                u64_t const* const* part_offsets) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        u32_t const part = slots[row] / part_capacity, position = slots[row] % part_capacity;
        byte_t const* const from = values + offsets[row];
        byte_t* const into = part_values[part] + part_offsets[part][position];
        u64_t const bytes = offsets[row + 1] - offsets[row];
        for (u64_t byte = warp.lane; byte < bytes; byte += warp_type_::width_k) into[byte] = from[byte];
    }
}

/**
 *  @brief Copies the ragged values of the @p listed_count nodes at @p listed back to back into
 *      @p outputs on one block, the bytes @p offsets bound in @p values.
 *
 *  The block's first warp scans their lengths into @p output_offsets, one past each listed node
 *  from the base at its first, the previous call's end; then every warp copies whole values,
 *  one lane after another.
 */
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ void vram_gather_tape(byte_t const* values, u64_t const* offsets,
                                                 node_id_type_ const* listed, u32_t listed_count, byte_t* outputs,
                                                 u64_t* output_offsets) noexcept {
    u32_t const lane = threadIdx.x % warp_type_::width_k, warp = threadIdx.x / warp_type_::width_k,
                warps = blockDim.x / warp_type_::width_k;
    if (warp == 0) {
        u64_t carried = output_offsets[0];
        for (u32_t base = 0; base < listed_count; base += warp_type_::width_k) { // Warp-uniform for the scan.
            u32_t const entry = base + lane;
            u64_t bytes = entry < listed_count ? offsets[listed[entry] + 1] - offsets[listed[entry]] : 0;
            for (u32_t delta = 1; delta < warp_type_::width_k; delta <<= 1) {
                u64_t const below = warp_type_::shuffle_up(bytes, delta);
                if (lane >= delta) bytes += below;
            }
            if (entry < listed_count) output_offsets[entry + 1] = carried + bytes;
            carried += warp_type_::shuffle(bytes, warp_type_::width_k - 1);
        }
    }
    __syncthreads();
    for (u32_t entry = warp; entry < listed_count; entry += warps) {
        byte_t const* const from = values + offsets[listed[entry]];
        byte_t* const into = outputs + output_offsets[entry];
        u64_t const bytes = output_offsets[entry + 1] - output_offsets[entry];
        for (u64_t byte = lane; byte < bytes; byte += warp_type_::width_k) into[byte] = from[byte];
    }
}

/** Moves every fixed-width row of @p row_words words into its part's receiver, one warp per row, at
 *  the row its slot names. */
template <typename warp_type_>
__device__ __forceinline__ void vram_split_rows(u32_t const* rows, u32_t row_words, u32_t rows_count,
                                                u32_t const* slots, u32_t part_capacity,
                                                u32_t* const* part_rows) noexcept {
    vram_grid_warp_t const warp = vram_grid_warp<warp_type_>();
    for (u32_t row = warp.index; row < rows_count; row += warp.count) {
        u32_t const part = slots[row] / part_capacity, position = slots[row] % part_capacity;
        u32_t const* const from = vram_row(rows, row, row_words);
        u32_t* const into = vram_row(part_rows[part], position, row_words);
        for (u32_t word = warp.lane; word < row_words; word += warp_type_::width_k) into[word] = from[word];
    }
}

/** Moves every row's vector, norms and key into its part's receiver at the row its slot names:
 *  @p part_columns holds the receivers' vectors, norms and keys, @p parts_count bases each, and a
 *  zero @p norm_words moves no norms. */
template <typename warp_type_>
__device__ __forceinline__ void vram_split_nodes(u32_t const* vectors, u32_t vector_words, u32_t const* norms,
                                                 u32_t norm_words, u32_t const* keys, u32_t key_words, u32_t rows_count,
                                                 u32_t const* slots, u32_t part_capacity, u32_t parts_count,
                                                 u32_t* const* part_columns) noexcept {
    vram_split_rows<warp_type_>(vectors, vector_words, rows_count, slots, part_capacity, part_columns);
    if (norm_words)
        vram_split_rows<warp_type_>(norms, norm_words, rows_count, slots, part_capacity, part_columns + parts_count);
    vram_split_rows<warp_type_>(keys, key_words, rows_count, slots, part_capacity, part_columns + 2 * parts_count);
}

/** Whether @p kernel partitions or scatters a split, which @ref vram_split_pass runs. */
constexpr bool vram_split_kernel(vram_kernel_t kernel) noexcept {
    return kernel >= vram_kernel_t::partition_roots_k && kernel <= vram_kernel_t::split_nodes_k &&
           kernel != vram_kernel_t::split_tape_k;
}

/** Runs the dense split's pass @p kernel_ over @p graph, as every vendor's dense kernel dispatches
 *  it. */
template <vram_kernel_t kernel_, typename warp_type_>
__device__ __forceinline__ void vram_split_pass(vram_graph_t const& graph) noexcept {
    if constexpr (kernel_ == vram_kernel_t::partition_roots_k)
        vram_partition_roots<warp_type_>(graph.roots, graph.seed_count, graph.node_parts, graph.frontier,
                                         graph.part_sizes);
    else if constexpr (kernel_ == vram_kernel_t::partition_propose_k)
        vram_partition_propose<warp_type_>(graph.neighbors, graph.rows_count, graph.connectivity, graph.node_parts,
                                           graph.proposals, graph.frontier, graph.part_sizes, graph.part_capacity,
                                           graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_count_k)
        vram_partition_count<warp_type_>(graph.node_parts, graph.proposals, graph.rows_count, graph.round_counts,
                                         graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_accept_k)
        vram_partition_accept<warp_type_>(graph.node_parts, graph.proposals, graph.rows_count, graph.next_frontier,
                                          graph.part_sizes, graph.round_counts, graph.part_capacity, graph.counter,
                                          graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_fill_k)
        vram_partition_fill<warp_type_>(graph.node_parts, graph.proposals, graph.rows_count, graph.next_frontier,
                                        graph.part_sizes, graph.round_counts, graph.part_capacity, graph.counter,
                                        graph.previous_counter);
    else if constexpr (kernel_ == vram_kernel_t::partition_islands_k)
        vram_partition_islands<warp_type_>(graph.node_parts, graph.rows_count, graph.part_sizes, graph.parts_count);
    else if constexpr (kernel_ == vram_kernel_t::partition_picked_k)
        vram_partition_picked<warp_type_>(graph.neighbors, graph.rows_count, graph.connectivity, graph.node_parts,
                                          graph.picked);
    else if constexpr (kernel_ == vram_kernel_t::split_links_k)
        vram_split_links<warp_type_>(graph.neighbors, graph.rows_count, graph.connectivity, graph.node_parts,
                                     graph.part_capacity, graph.part_columns);
    else
        vram_split_nodes<warp_type_>(reinterpret_cast<u32_t const*>(graph.storage.vectors), graph.vector_words,
                                     reinterpret_cast<u32_t const*>(graph.storage.norms), graph.norm_words, graph.keys,
                                     graph.key_words, graph.rows_count, graph.node_parts, graph.part_capacity,
                                     graph.parts_count, graph.part_columns + graph.parts_count);
}

#pragma endregion Splits

} // namespace unum::usearch::dense
