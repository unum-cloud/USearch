/**
 *  @file include/usearch/geospatial/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over observations on NVIDIA GPUs: the basic VRAM index over the chord
 *      kit of @c vram_nvidia_metric.cuh, and the kernel judging each walk's matches.
 *
 *  Plain C++, so the host classes build with any compiler: the kit's device members and the
 *  judging kernel compile only into the fatbin, while the kit's host facts pick a launch's kit and
 *  size its spans. Rows are the 32-byte @c earth_observation_t, in the caller's tape and in the
 *  index alike.
 */
#pragma once

#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/geospatial/vram_nvidia_metric.cuh"

namespace unum::usearch::geospatial {

/** Similarity Search over observations on NVIDIA GPUs, under the squared chord: a lane scores each
 *  candidate, a warp walks each query, and rows keep their neighbors by tangent-plane cone. */
using vram_nvidia_index = basic::vram_nvidia_index<i32_ampere_chord_kit_t>;

/** One launch of @c cuda_admit, which keeps the first @c wanted entries of each walk's pool that
 *  the relation admits, ascending, beside their chords in metres and their count. */
struct cuda_admit_t {
    basic::cuda_ragged_storage_t storage;

    /** The queries, @b [queries], and the pools their walks landed, @b [queries,pool] IDs. */
    earth_observation_t const* queries = nullptr;
    u32_t const* pool_ids = nullptr;
    u32_t queries_count = 0;
    u32_t pool = 0;
    u32_t wanted = 0;

    relation_t relation = relation_t::near_k;
    f64_t radius_metres = 0;
    f64_t slack_milliseconds = 0;

    /** The landing, @b [queries,wanted] IDs and metres, and a count per query. */
    u32_t* matched_ids = nullptr;
    f64_t* matched_metres = nullptr;
    u64_t* matched_counts = nullptr;
};

/** The Itanium mangled name of @c cuda_admit<i32_ampere_chord_kit_t>, as NVCC emits it. */
inline constexpr char cuda_admit_name_k[] =
    "_ZN4unum7usearch10geospatial10cuda_admitINS1_22i32_ampere_chord_kit_tEEEvNS1_12cuda_admit_tE";

#if defined(__CUDACC__)

/** One warp per query keeps, in rank order, the pool entries the relation admits, a warp-width of
 *  ranks at a time, until @c wanted landed or the pool ran out. */
template <typename kit_type_>
__global__ void __launch_bounds__(dense::vram_warps_per_block_k * 32) cuda_admit(cuda_admit_t arguments) {
    using warp_t = typename kit_type_::warp_t;
    u32_t const lane = threadIdx.x % warp_t::width_k;
    u32_t const warps = gridDim.x * dense::vram_warps_per_block_k;
    for (u32_t query_index = blockIdx.x * dense::vram_warps_per_block_k + threadIdx.x / warp_t::width_k;
         query_index < arguments.queries_count; query_index += warps) {
        earth_observation_t const query = arguments.queries[query_index];
        u32_t const* const pool_ids = arguments.pool_ids + u64_t {query_index} * arguments.pool;
        u32_t* const matched_ids = arguments.matched_ids + u64_t {query_index} * arguments.wanted;
        f64_t* const matched_metres = arguments.matched_metres + u64_t {query_index} * arguments.wanted;
        u32_t landed = 0;
        for (u32_t base = 0; base < arguments.pool && landed < arguments.wanted; base += warp_t::width_k) {
            u32_t const rank = base + lane;
            dense::node_id_t const node_id = rank < arguments.pool ? pool_ids[rank] : dense::node_id_unknown_k;
            bool admitted = false;
            i64_t chord = 0;
            if (node_id != dense::node_id_unknown_k) {
                earth_observation_t const entry = *reinterpret_cast<earth_observation_t const*>(
                    arguments.storage.values + arguments.storage.offsets[node_id]);
                chord = chord_squared(query.position, entry.position);
                admitted = cuda_admits(arguments.relation, arguments.radius_metres, arguments.slack_milliseconds, query,
                                       entry, chord);
            }
            typename warp_t::mask_t const admitted_mask = warp_t::ballot(admitted);
            u32_t const position = landed + warp_t::count(admitted_mask & warp_t::below(lane));
            if (admitted && position < arguments.wanted) {
                matched_ids[position] = node_id;
                matched_metres[position] = nk_sqrt_f64_cuda_(static_cast<f64_t>(chord)) / centimetres_per_metre_k;
            }
            landed += warp_t::count(admitted_mask);
        }
        if (lane == 0) arguments.matched_counts[query_index] = min_of(landed, arguments.wanted);
    }
}

#endif

} // namespace unum::usearch::geospatial
