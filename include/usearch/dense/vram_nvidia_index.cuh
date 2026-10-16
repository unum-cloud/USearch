/**
 *  @file include/usearch/dense/vram_nvidia_index.cuh
 *  @author Ash Vardanian
 *  @date July 18, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  The kernel side of the index: the exact scan and the one kernel template over five integers -
 *  the kernel, the capability ceiling, the dtype, the register rung and the attend value lookup -
 *  whose instantiations a fatbin carries. Only NVCC's device-only passes include it, once per
 *  ceiling over every compiled dtype; each @c cuda_kernel turns the plain records of
 *  @c vram_index.hpp into the shared views of @c vram_index.cuh, runs the shared bodies there
 *  through @ref cuda_warp_t, and the host finds it by the mangled name @c cuda_kernel_name spells.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy`

#include <type_traits> // `std::is_same_v` - the bit kits' metrics
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

#pragma region Kernels

/** The kit of one dtype under one capability ceiling, both as the kernels' template integers. */
template <u32_t capabilities_, u32_t dtype_>
using cuda_kit =
    cuda_kit_t<static_cast<nk_dtype_t>(dtype_), static_cast<nk_capability_t>(capabilities_) * nk_cap_cuda_k>;

/** Blocks of each kernel resident per multiprocessor, the register budget its launch bounds set:
 *  the builders' saturating count, a walk's by its rung, and one for the light graph sweeps. */
constexpr u32_t cuda_resident_blocks(cuda_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case cuda_kernel_t::insert_k:
    case cuda_kernel_t::refine_k: return cuda_insert_blocks_per_multiprocessor_k;
    case cuda_kernel_t::back_link_k:
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
    cuda_kernel(vram_arguments_t arguments) {
    using kit_t = cuda_kit<capabilities_, dtype_>;
    using schedule_t = cuda_schedule_of<static_cast<nk_capability_t>(capabilities_) * nk_cap_cuda_k>;
    constexpr cuda_kernel_t kernel = static_cast<cuda_kernel_t>(kernel_);
    vram_graph_t const& graph = arguments.graph;
    // Launched overlapping its predecessor's drain, so nothing is read before that predecessor
    // lands; the successor's blocks may then seat as this grid's blocks retire.
    if constexpr (cuda_overlaps_launches(static_cast<nk_capability_t>(capabilities_) * nk_cap_cuda_k))
        asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory");
    if constexpr (kernel == cuda_kernel_t::insert_k)
        vram_insert<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel == cuda_kernel_t::back_link_k)
        vram_back_link<kit_t, schedule_t>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel == cuda_kernel_t::refine_k)
        vram_refine<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.refine));
    else if constexpr (kernel == cuda_kernel_t::search_k)
        vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                  vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    else if constexpr (kernel == cuda_kernel_t::search_indexer_k) {
        // Only the floating dtypes take indexer heads; others export a plain walk never launched.
        if constexpr (floating_scalar<typename kit_t::scalar_t>)
            vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                      vram_view<kit_t>(arguments.search_sink),
                                                      vram_view<kit_t>(arguments.rescoring));
        else
            vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                      vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    }
    else if constexpr (kernel == cuda_kernel_t::scan_k) {
        vram_search_view<kit_t> const view = vram_view<kit_t>(arguments.search);
        vram_search_sink<kit_t> const sink = vram_view<kit_t>(arguments.search_sink);
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
        vram_walk_rung<kit_t, schedule_t, slots_>(vram_view<kit_t>(arguments.search),
                                                  vram_view<kit_t>(arguments.multiply_sink), vram_no_rescoring {});
    else if constexpr (kernel == cuda_kernel_t::attend_k) {
        using value_lookup_t = typename cuda_value_lookup_of<lookup_>::type;
        vram_attend_t const& attend = arguments.attend;
        vram_rescoring_t const& heads = arguments.rescoring;
        value_lookup_t value_lookup;
        std::memcpy(&value_lookup, attend.value_lookup, sizeof(value_lookup_t));
        vram_walk_rung<kit_t, schedule_t, slots_>(
            vram_view<kit_t>(arguments.search),
            vram_attend_sink<kit_t, value_lookup_t> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                     attend.logit_scale, attend.heads_count, attend.columns},
            vram_heads_rescoring<kit_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads), nullptr,
                                         heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel == cuda_kernel_t::merge_back_links_k)
        vram_merge_back_links<kit_t, schedule_t>(vram_view<kit_t>(arguments.refine), graph.bucket_ids,
                                                 graph.bucket_counts, graph.bucket_width, graph.touched);
    else if constexpr (kernel == cuda_kernel_t::scatter_back_links_k)
        vram_scatter_back_links<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count,
                                             graph.node_ids, graph.connectivity, graph.bucket_width, graph.bucket_ids,
                                             graph.bucket_counts, graph.touched);
    else if constexpr (kernel == cuda_kernel_t::derive_norms_k)
        vram_derive_norms<kit_t>(vram_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel == cuda_kernel_t::rebase_links_k)
        vram_rebase_links<cuda_warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                       graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel == cuda_kernel_t::mark_seeds_k)
        vram_mark_seeds<cuda_warp_t>(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel == cuda_kernel_t::reachability_sweep_k)
        vram_reachability_sweep<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                             graph.frontier, graph.next_frontier, graph.counter, graph.previous_counter,
                                             graph.levels);
    else if constexpr (kernel == cuda_kernel_t::reachability_islands_k)
        vram_reachability_islands<cuda_warp_t>(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel == cuda_kernel_t::reached_in_links_k)
        vram_reached_in_links<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                           graph.in_links);
    else if constexpr (kernel == cuda_kernel_t::reconnect_island_k)
        vram_reconnect_island<cuda_warp_t>(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count,
                                           graph.reached, graph.rows_count, graph.seed_count, graph.in_links,
                                           graph.reconnected);
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
                                          vram_view(arguments.matches.ranking));
    else if constexpr (kernel == cuda_kernel_t::merge_matches_k)
        vram_merge_matches<cuda_warp_t>(vram_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                        arguments.matches.shards, arguments.matches.queries_count,
                                        vram_view(arguments.matches.ranking));
    else
        vram_merge_attention<cuda_warp_t>(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
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
