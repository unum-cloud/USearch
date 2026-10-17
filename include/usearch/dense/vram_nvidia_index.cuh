/**
 *  @file include/usearch/dense/vram_nvidia_index.cuh
 *  @author Ash Vardanian
 *  @date July 18, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  The kernel side of the index: the exact scan and the one kernel template over five arguments -
 *  the kernel, the capability ceiling, the dtype, the register rung and the attend value lookup -
 *  whose instantiations a fatbin carries. Only NVCC's device-only passes include it, once per
 *  ceiling over every compiled dtype; each @c cuda_kernel turns the plain records of
 *  @c vram_index.hpp into the shared views of @c vram_index.cuh, runs the shared bodies there
 *  through @ref cuda_warp_t, and the host finds it by the symbol @c vram_kernel_symbol spells.
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
static_assert(vram_nvidia_index<nk_i8_k, nk_cap_cuda_k | nk_cap_ampere_k, void>::max_node_count_k <=
              vram_scored_node<node_id_t>::id_capacity_k);

/** Shared memory a walking warp keeps for the rows @p schedule_ gathers, at its widest window; none
 *  where it gathers none. */
template <typename schedule_>
constexpr std::size_t cuda_gather_bytes_per_warp() noexcept {
    return 0;
}

template <typename schedule_>
    requires requires { schedule_::gather_window_bytes(1u); }
constexpr std::size_t cuda_gather_bytes_per_warp() noexcept {
    return sizeof(typename schedule_::template gather_stages<1>);
}

/** Bytes of a row one gather under @p schedule_ lands: the narrowest window any walk of
 *  @p capability_ over @p dtype_ seats, so one tensor map serves every walk; zero where the
 *  schedule gathers no rows. */
template <typename schedule_, nk_capability_t capability_, nk_dtype_t dtype_>
constexpr u32_t cuda_gather_window_bytes() noexcept {
    return 0;
}

template <typename schedule_, nk_capability_t capability_, nk_dtype_t dtype_>
    requires requires { schedule_::gather_window_bytes(1u); }
constexpr u32_t cuda_gather_window_bytes() noexcept {
    u32_t window = schedule_::gather_window_bytes(1);
    for (cuda_rung_t const rung : cuda_rungs)
        if (cuda_gathers_rows(rung.kernel))
            window = min_of(window, schedule_::gather_window_bytes(
                                        cuda_resident_blocks<capability_, dtype_>(rung.kernel, rung.slots)));
    for (u32_t const slots : cuda_walk_rungs)
        window = min_of(window, schedule_::gather_window_bytes(
                                    cuda_resident_blocks<capability_, dtype_>(vram_kernel_t::attend_k, slots)));
    return window;
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename kernels_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
bool vram_nvidia_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_, neighbors_allocator_type_,
                       traversals_allocator_type_>::walk_scratch_fits(std::size_t fold_bytes) const noexcept {
    using kit_t = typename quads_ladder<scalar_t, capability_>::kit_t;
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    using warp_t = typename schedule_t::warp_t;
    static_assert(std::is_same_v<typename kit_t::norms_t, norms_t> &&
                  sizeof(typename kit_t::fragment_t) == sizeof(fragment_t));
    return std::size_t {vram_warps_per_block_k} * (row_bytes() + fold_bytes + sizeof(vram_gathered<kit_t, warp_t>) +
                                                   cuda_gather_bytes_per_warp<schedule_t>()) <=
           device_.shared_memory_per_block_bytes;
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename kernels_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
cuda_rows_map_t vram_nvidia_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_,
                                  neighbors_allocator_type_, traversals_allocator_type_>::rows_map_() noexcept {
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    constexpr u32_t window_bytes = cuda_gather_window_bytes<schedule_t, capability_, vector_dtype_>();
    if constexpr (window_bytes == 0) return {};
    else {
        standard_atomic_ref<u32_t> state {rows_map_state_};
        if (state.load(std::memory_order_acquire) == 2) return rows_map_kept_;
        // The signed tensor coordinates reach 2^31 - 1 rows at most.
        std::size_t const ceiling_rows = this->vectors_matrix_.reserved_bytes() / row_bytes() - 1;
        u32_t const rows = static_cast<u32_t>(min_of<std::size_t>(ceiling_rows, 0x7FFFFFFFu));
        u32_t const vector_row_bytes = static_cast<u32_t>(row_bytes());
        cuda_rows_map_t const map = cuda_set_device(device_.ordinal) == cuda_success_k
                                        ? cuda_encode_rows_map(storage().vectors, vector_row_bytes, rows,
                                                               min_of(vector_row_bytes, window_bytes))
                                        : cuda_rows_map_t {};
        u32_t idle = 0;
        if (state.compare_exchange_strong(idle, 1, std::memory_order_acquire)) {
            rows_map_kept_ = map;
            state.store(2, std::memory_order_release);
        }
        return map;
    }
}

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename kernels_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_>
bool vram_nvidia_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_, neighbors_allocator_type_,
                       traversals_allocator_type_>::build_scratch_fits() const noexcept {
    using kit_t = typename quads_ladder<scalar_t, capability_>::kit_t;
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    std::size_t const statics = std::size_t {vram_warps_per_block_k} *
                                (vram_cache_rows_k * sizeof(norms_t) +
                                 sizeof(vram_gathered<kit_t, typename schedule_t::warp_t>) +
                                 sizeof(typename schedule_t::shared_t) + cuda_gather_bytes_per_warp<schedule_t>());
    std::size_t const block_bytes = statics +
                                    std::size_t {vram_warps_per_block_k} *
                                        max_of(insert_scratch_bytes_per_warp(), merge_scratch_bytes_per_warp());
    // Every block a builder seats takes its own scratch, so the most any builder seats must fit on
    // one multiprocessor, or the launch seats fewer blocks than its grid and arena were sized for.
    std::size_t const resident_blocks = max_of(resident_blocks_(vram_kernel_t::insert_k, 0),
                                               resident_blocks_(vram_kernel_t::refine_k, 0));
    return block_bytes <= device_.shared_memory_per_block_bytes &&
           resident_blocks * (block_bytes + device_.reserved_shared_memory_per_block_bytes) <=
               device_.shared_memory_per_multiprocessor_bytes;
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
 *  @c vram_scored_node, so a tie in distance keeps the lower row. @p kernel_ fixes the ranking
 *  metric, so the epilogue branches on none, while the matches convert out by the sink's own
 *  metric, as a walk's do. @p resident_blocks_ is the grid's share of every multiprocessor, which
 *  the host split the ranges by.
 */
template <vram_quads_kit kit_, nk_kernel_kind_t kernel_, u32_t resident_blocks_>
__device__ __forceinline__ void cuda_scan(vram_search_view<kit_> const& view, vram_search_sink<kit_> const& sink) {
    using fragment_t = typename kit_::fragment_t;
    using dot_t = typename kit_::dot_t;
    using norms_t = typename kit_::norms_t;
    using tile_t = std::array<dot_t, 4>;
    static_assert(vram_warps_per_block_k == 4, "Four warps split a scanning step two by two");
    constexpr u32_t tile_k = cuda_scan_tile_k, staging_k = 2 * tile_k, rounds_k = staging_k / cuda_warp_t::width_k;
    constexpr u32_t width_k = cuda_warp_t::width_k;
    extern __shared__ vram_scored_node<node_id_t> staged[];
    __shared__ vram_scored_node<node_id_t> worst[tile_k];
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
    u32_t const ranges = cuda_scan_ranges(gridDim.x, resident_blocks_, blocks, visible);
    u32_t const range_rows = round_up_to_multiple(divide_round_up(visible, ranges), tile_k);
    u32_t const spans = divide_round_up(storage.row_fragments, 4u);
    u32_t* const finished_ranges = reinterpret_cast<u32_t*>(view.arena);
    vram_scored_node<node_id_t>* const matches = reinterpret_cast<vram_scored_node<node_id_t>*>(
        finished_ranges + round_up_to_multiple(blocks, 4u));

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
            worst[local] = local < block_queries && count ? vram_scored_node<node_id_t>::sentinel()
                                                          : vram_scored_node<node_id_t> {0};
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
                    vram_scored_node<node_id_t>* const kept =
                        matches + (std::size_t {first_query + local} * ranges + range) * count;
                    vram_scored_node<node_id_t> const* const candidates = staged + local * staging_k;
                    u32_t places[rounds_k];
#pragma unroll
                    for (u32_t round = 0; round != rounds_k; ++round) {
                        u32_t const index = round * width_k + lane;
                        places[round] = 0;
                        if (index >= pending) continue;
                        vram_scored_node<node_id_t> const candidate = candidates[index];
                        for (u32_t other = 0; other != pending; ++other) places[round] += candidates[other] < candidate;
                        places[round] += vram_rank_below(held, candidate.bits_, vram_ties_t::excluded_k,
                                                         [&](u32_t rank) noexcept { return kept[rank].bits_; });
                    }
                    for (u32_t top = held; top;) {
                        u32_t const first = top > width_k ? top - width_k : 0;
                        u32_t const index = first + lane;
                        vram_scored_node<node_id_t> const value = index < top ? kept[index]
                                                                              : vram_scored_node<node_id_t>::sentinel();
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
                        worst[local] = now_held == count ? kept[count - 1] : vram_scored_node<node_id_t>::sentinel();
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
                            vram_scored_node<node_id_t> const candidate = vram_scored_node<node_id_t>::make(distance,
                                                                                                            row);
                            u32_t const local = warp_queries + query_tile * 16 + half * 8 + group;
                            if (!(candidate < worst[local])) continue;
                            u32_t const slot = cuda_warp_t::fetch_add_relaxed(staged_counts[local], 1u);
                            staged[local * staging_k + slot] = candidate;
                            if (slot >= staging_k - tile_k) staging_full[step % 3] = 1;
                        }
            __syncthreads();
        }

        for (u32_t local = threadIdx.x; local < block_queries; local += blockDim.x) {
            vram_scored_node<node_id_t>* const kept = matches +
                                                      (std::size_t {first_query + local} * ranges + range) * count;
            if (kept_counts[local] < count) kept[kept_counts[local]] = vram_scored_node<node_id_t>::sentinel();
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
        vram_scored_node<node_id_t>* const heads = staged + warp * ranges;
        u32_t* const taken = reinterpret_cast<u32_t*>(staged + vram_warps_per_block_k * ranges) + warp * ranges;
        for (u32_t local = warp; local < block_queries; local += vram_warps_per_block_k) {
            u32_t const query = first_query + local;
            vram_scored_node<node_id_t>* const query_matches = matches + std::size_t {query} * ranges * count;
            node_id_t* const ids_row = vram_row(sink.result_ids, query, count);
            distance_t* const distances_row = sink.result_distances ? vram_row(sink.result_distances, query, count)
                                                                    : nullptr;
            for (u32_t of_range = lane; of_range < ranges; of_range += width_k) {
                heads[of_range] = count ? vram_scored_node<node_id_t> {cuda_warp_t::load_relaxed(
                                              query_matches[std::size_t {of_range} * count].bits_)}
                                        : vram_scored_node<node_id_t>::sentinel();
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
                vram_scored_node<node_id_t> const held = chosen < ranges ? heads[chosen]
                                                                         : vram_scored_node<node_id_t>::sentinel();
                vram_scored_node<node_id_t> least = held;
#pragma unroll
                for (u32_t mask = width_k / 2; mask; mask >>= 1) {
                    vram_scored_node<node_id_t> const other = shuffle_xor<cuda_warp_t>(least, mask);
                    if (other < least) least = other;
                }
                if (lane == 0) {
                    ids_row[rank] = least.id();
                    if (distances_row) distances_row[rank] = kit_::convert_to_distance(sink.kernel, least.distance());
                }
                // Every row sits in one range, so one lane at most holds the nearest.
                advanced = held == least && !(least == vram_scored_node<node_id_t>::sentinel());
                if (advanced) {
                    u32_t const position = ++taken[chosen];
                    heads[chosen] = position < count
                                        ? vram_scored_node<node_id_t> {cuda_warp_t::load_relaxed(
                                              query_matches[std::size_t {chosen} * count + position].bits_)}
                                        : vram_scored_node<node_id_t>::sentinel();
                }
            }
        }
    }
}

#pragma region Kernels

/** Every NVIDIA kernel: @p kernel_ picks the body, @p capability_ and @p dtype_ the kit,
 *  @p capability_ alone the schedule and its warp, @p slots_ the rung, and @p lookup_ an attend's
 *  value lookup through @ref cuda_value_lookup_of. The blocks @ref cuda_resident_blocks seats per
 *  multiprocessor bound its registers and reach the body, which sizes its load window by them. The
 *  arguments stay in the parameter space, where a walk's schedule may read the tensor map. */
template <vram_kernel_t kernel_, nk_capability_t capability_, nk_dtype_t dtype_, u32_t slots_, u32_t lookup_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32,
                                  cuda_resident_blocks<capability_, dtype_>(kernel_, slots_))
    cuda_kernel(const __grid_constant__ cuda_arguments_t arguments) {
    using kit_t = typename quads_ladder<typename nk::type_for<dtype_>::type, capability_>::kit_t;
    using schedule_t = typename schedule_ladder<capability_>::schedule_t;
    using warp_t = typename schedule_t::warp_t;
    constexpr u32_t resident_blocks = cuda_resident_blocks<capability_, dtype_>(kernel_, slots_);
    vram_graph_t const& graph = arguments.graph;
    // Launched overlapping its predecessor's drain, so nothing is read before that predecessor
    // lands; the successor's blocks may then seat as this grid's blocks retire.
    if constexpr (cuda_overlaps_launches(capability_))
        asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory");
    if constexpr (cuda_gathers_rows(kernel_) && requires(cuda_rows_map_t const& rows_map) {
                      schedule_t::template bind<resident_blocks>(rows_map, 0u);
                  })
        schedule_t::template bind<resident_blocks>(arguments.rows_map, threadIdx.x % warp_t::width_k);
    if constexpr (kernel_ == vram_kernel_t::insert_k)
        vram_insert<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel_ == vram_kernel_t::back_link_k)
        vram_back_link<kit_t, schedule_t, resident_blocks>(vram_view<kit_t>(arguments.build));
    else if constexpr (kernel_ == vram_kernel_t::refine_k)
        vram_refine<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.refine));
    else if constexpr (kernel_ == vram_kernel_t::search_k)
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
            vram_view<kit_t>(arguments.search), vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    else if constexpr (kernel_ == vram_kernel_t::search_indexer_k) {
        // Only the floating dtypes take indexer heads; others export a plain walk never launched.
        if constexpr (floating_scalar<typename kit_t::scalar_t>)
            vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.search),
                                                                       vram_view<kit_t>(arguments.search_sink),
                                                                       vram_view<kit_t, warp_t>(arguments.rescoring));
        else
            vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
                vram_view<kit_t>(arguments.search), vram_view<kit_t>(arguments.search_sink), vram_no_rescoring {});
    }
    else if constexpr (kernel_ == vram_kernel_t::scan_k) {
        vram_search_view<kit_t> const view = vram_view<kit_t>(arguments.search);
        vram_search_sink<kit_t> const sink = vram_view<kit_t>(arguments.search_sink);
        // Each kit's last metric is the one its `distance` falls back to.
        if constexpr (std::is_same_v<typename kit_t::scalar_t, nk::u1x8_t>) switch (sink.kernel) {
            case nk_kernel_dot_k: cuda_scan<kit_t, nk_kernel_dot_k, resident_blocks>(view, sink); break;
            case nk_kernel_hamming_k: cuda_scan<kit_t, nk_kernel_hamming_k, resident_blocks>(view, sink); break;
            default: cuda_scan<kit_t, nk_kernel_jaccard_k, resident_blocks>(view, sink); break;
            }
        else switch (sink.kernel) {
            case nk_kernel_dot_k: cuda_scan<kit_t, nk_kernel_dot_k, resident_blocks>(view, sink); break;
            case nk_kernel_angular_k: cuda_scan<kit_t, nk_kernel_angular_k, resident_blocks>(view, sink); break;
            default: cuda_scan<kit_t, nk_kernel_euclidean_k, resident_blocks>(view, sink); break;
            }
    }
    else if constexpr (kernel_ == vram_kernel_t::multiply_k)
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(vram_view<kit_t>(arguments.search),
                                                                   vram_view<kit_t, warp_t>(arguments.multiply_sink),
                                                                   vram_no_rescoring {});
    else if constexpr (kernel_ == vram_kernel_t::attend_k) {
        using value_lookup_t = typename cuda_value_lookup_of<lookup_>::type;
        vram_attend_t const& attend = arguments.attend;
        vram_rescoring_t const& heads = arguments.rescoring;
        value_lookup_t value_lookup;
        std::memcpy(&value_lookup, attend.value_lookup, sizeof(value_lookup_t));
        vram_walk_rung<kit_t, schedule_t, slots_, resident_blocks>(
            vram_view<kit_t>(arguments.search),
            vram_attend_sink<kit_t, warp_t, value_lookup_t> {value_lookup, attend.outputs, attend.states, attend.missed,
                                                             attend.logit_scale, attend.heads_count, attend.columns},
            vram_heads_rescoring<kit_t, warp_t> {reinterpret_cast<typename kit_t::fragment_t const*>(heads.heads),
                                                 nullptr, heads.heads_count, heads.row_fragments});
    }
    else if constexpr (kernel_ == vram_kernel_t::merge_back_links_k)
        vram_merge_back_links<kit_t, schedule_t, resident_blocks>(vram_view<kit_t>(arguments.refine), graph.bucket_ids,
                                                                  graph.bucket_counts, graph.bucket_width,
                                                                  graph.touched);
    else if constexpr (kernel_ == vram_kernel_t::scatter_back_links_k)
        vram_scatter_back_links<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.first_id, graph.walk_count,
                                             graph.node_ids, graph.connectivity, graph.bucket_width, graph.bucket_ids,
                                             graph.bucket_counts, graph.touched);
    else if constexpr (kernel_ == vram_kernel_t::derive_norms_k)
        vram_derive_norms<kit_t, warp_t>(vram_view<kit_t>(graph.storage), graph.first_id, graph.rows_count);
    else if constexpr (kernel_ == vram_kernel_t::rebase_links_k)
        vram_rebase_links<cuda_warp_t>(graph.neighbors, graph.first_id, graph.rows_count, graph.connectivity,
                                       graph.source_first, graph.source_bound, graph.shift);
    else if constexpr (kernel_ == vram_kernel_t::mark_seeds_k)
        vram_mark_seeds<cuda_warp_t>(graph.reached, graph.frontier, graph.rows_count, graph.seed_count);
    else if constexpr (kernel_ == vram_kernel_t::reachability_sweep_k)
        vram_reachability_sweep<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                             graph.frontier, graph.next_frontier, graph.counter, graph.previous_counter,
                                             graph.levels);
    else if constexpr (kernel_ == vram_kernel_t::reachability_islands_k)
        vram_reachability_islands<cuda_warp_t>(graph.reached, graph.rows_count, graph.island_ids, graph.counter);
    else if constexpr (kernel_ == vram_kernel_t::reached_in_links_k)
        vram_reached_in_links<cuda_warp_t>(graph.neighbors, graph.rows_count, graph.connectivity, graph.reached,
                                           graph.in_links);
    else if constexpr (kernel_ == vram_kernel_t::reconnect_island_k)
        vram_reconnect_island<cuda_warp_t>(graph.neighbors, graph.connectivity, graph.island_ids, graph.islands_count,
                                           graph.reached, graph.rows_count, graph.seed_count, graph.in_links,
                                           graph.reconnected);
    else if constexpr (kernel_ == vram_kernel_t::cast_k) {
        nk_size_t const threads = static_cast<nk_size_t>(gridDim.x) * blockDim.x;
        for (nk_size_t unit = static_cast<nk_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             unit < arguments.cast.plan.units; unit += threads)
            nk_cast_unit_simt_(&arguments.cast, unit);
    }
    else if constexpr (kernel_ == vram_kernel_t::rmscast_k) {
        // Only narrow floats take a unit-RMS import; every other dtype exports one never launched.
        if constexpr (dtype_ == nk_f16_k) nk_each_rmscast_f16_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e4m3_k) nk_each_rmscast_e4m3_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e5m2_k) nk_each_rmscast_e5m2_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e2m3_k) nk_each_rmscast_e2m3_cuda_vectors_(arguments.rmscast);
        else if constexpr (dtype_ == nk_e3m2_k) nk_each_rmscast_e3m2_cuda_vectors_(arguments.rmscast);
    }
    else if constexpr (kernel_ == vram_kernel_t::resolve_matches_k)
        vram_resolve_matches<cuda_warp_t>(arguments.matches.ids, arguments.matches.distances, arguments.matches.walked,
                                          arguments.matches.keys, arguments.matches.queries_count,
                                          vram_view(arguments.matches.ranking));
    else if constexpr (kernel_ == vram_kernel_t::merge_matches_k)
        vram_merge_matches<cuda_warp_t>(vram_view(arguments.matches.gathered), arguments.matches.block_bytes,
                                        arguments.matches.shards, arguments.matches.queries_count,
                                        vram_view(arguments.matches.ranking));
    else
        vram_merge_attention<cuda_warp_t>(reinterpret_cast<byte_t*>(arguments.attend.outputs), arguments.attend.rows,
                                          arguments.attend.columns, arguments.attend.shards);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<cuda_rungs[rungs_].kernel, capability_, dtype_, cuda_rungs[rungs_].slots, 0>), ...);
}

template <nk_capability_t capability_, nk_dtype_t dtype_, u32_t lookup_, std::size_t... rungs_>
__device__ void cuda_instantiate_attends(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<vram_kernel_t::attend_k, capability_, dtype_, cuda_walk_rungs[rungs_], lookup_>),
     ...);
}

/** Instantiates every kernel one dtype exports under one ceiling, and its attends over each of
 *  @p lookups_; a unit compiling a fatbin names it once per compiled dtype. Taking the kernels'
 *  addresses in device code makes NVCC emit them, while a host-only pass emits no launch stub. */
template <nk_capability_t capability_, nk_dtype_t dtype_, u32_t... lookups_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_rungs<capability_, dtype_>(
        std::make_index_sequence<sizeof(cuda_rungs) / sizeof(cuda_rungs[0])> {});
    (cuda_instantiate_attends<capability_, dtype_, lookups_>(
         std::make_index_sequence<sizeof(cuda_walk_rungs) / sizeof(cuda_walk_rungs[0])> {}),
     ...);
}

#pragma endregion Kernels

} // namespace unum::usearch::dense
