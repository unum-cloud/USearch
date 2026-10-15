/**
 *  @file include/usearch/geospatial/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The device kit, the octant selector and the relation tests of the geospatial VRAM index,
 *      the GPU twins of @c ram_metric.hpp and of @c ram_index.hpp's selector and @c admits.
 *
 *  A stored row is one @c earth_observation_t, and a lane scores a candidate alone: the squared
 *  chord from the three 64-bit products of centimetre differences, exact in integers as the CPU
 *  kit scores it, rounded once to the walk's distance. Nothing past the @c ampere floor is needed.
 *  A row prepares into no span at all, since its position is its query.
 *
 *  Any compiler reads the kit's host facts, so the host index picks it and sizes its spans from the
 *  kit itself; the device members compile only where CUDA does.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/capabilities.h> // `nk_kernel_sqeuclideans3d_k` - the kit's name in the kernels

#include "usearch/utilities/types.hpp"
#include "usearch/geospatial/ram_index.hpp"

#if defined(__CUDACC__)
#include <limits> // `std::numeric_limits` - an unknown pitch, field or range

#include <numkong/geospatial/cuda.cuh> // `nk_terrain_outside_view_f32_cuda_`, `nk_sqrt_f64_cuda_`

#include "usearch/dense/vram_nvidia_metric.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#endif

namespace unum::usearch::geospatial {
using namespace unum::usearch::utilities;

/** Squared chords between stored observations, a lane to a candidate. */
struct i32_ampere_chord_kit_t {

    /** The integer naming it in the kernels: NumKong's dtype and kernel kind and one lane per row,
     *  in the fields the other modalities' kits use, so no two kits share one. */
    inline static constexpr u32_t kit_k = (static_cast<u32_t>(nk_i32_k) << 16) |
                                          (static_cast<u32_t>(nk_kernel_sqeuclideans3d_k) << 8) | 1u;
    inline static constexpr u32_t lanes_per_row_k = 1;

    static constexpr bool serves(std::size_t bytes) noexcept { return bytes <= sizeof(earth_observation_t); }

    /** A position is its own query, so a row prepares into no span. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t) noexcept { return 0; }

#if defined(__CUDACC__)
    using warp_t = dense::cuda_warp_t;
    using query_t = earth_fixed_t;

    /** The position at the front of the row, in one 16-byte load. */
    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t*, u32_t) noexcept {
        int4 const words = *reinterpret_cast<int4 const*>(row.data);
        return {words.x, words.y, words.z};
    }

    /** No header bounds a chord. */
    static __device__ __forceinline__ dense::distance_t bound(query_t const&, basic::vram_row_t) noexcept { return 0; }

    static __device__ __forceinline__ dense::distance_t score(query_t const& query, basic::vram_row_t row,
                                                              dense::distance_t, u32_t lane) noexcept {
        return static_cast<dense::distance_t>(chord_squared(query, prepare(row, nullptr, lane)));
    }
#endif
};

#if defined(__CUDACC__)

/** The integer tangent frame at @p base, as @c tangent_frame builds it on the host, in scalars so
 *  that no axis spills to the stack. */
__device__ __forceinline__ tangent_frame_t cuda_tangent_frame(earth_fixed_t base) noexcept {
    f64_t const x = base.x_centimetres, y = base.y_centimetres, z = base.z_centimetres;
    f64_t const norm = nk_sqrt_f64_cuda_(x * x + y * y + z * z);
    f64_t east_x = 1, east_y = 0, east_z = 0, north_x = 0, north_y = 1, north_z = 0;
    if (norm != 0) {
        f64_t const up_x = x / norm, up_y = y / norm, up_z = z / norm;
        // The axis least aligned with the position keeps the cross product's length above 0.816.
        f64_t const magnitude_x = fabs(up_x), magnitude_y = fabs(up_y), magnitude_z = fabs(up_z);
        bool const along_y = magnitude_y < magnitude_x;
        bool const along_z = magnitude_z < (along_y ? magnitude_y : magnitude_x);
        f64_t const axis_x = !along_y && !along_z, axis_y = along_y && !along_z, axis_z = along_z;
        east_x = axis_y * up_z - axis_z * up_y;
        east_y = axis_z * up_x - axis_x * up_z;
        east_z = axis_x * up_y - axis_y * up_x;
        f64_t const east_norm = nk_sqrt_f64_cuda_(east_x * east_x + east_y * east_y + east_z * east_z);
        east_x /= east_norm, east_y /= east_norm, east_z /= east_norm;
        north_x = up_y * east_z - up_z * east_y;
        north_y = up_z * east_x - up_x * east_z;
        north_z = up_x * east_y - up_y * east_x;
    }
    constexpr f64_t scale_k = static_cast<f64_t>(frame_scale_k);
    auto const quantized = [](f64_t component) noexcept {
        return static_cast<i32_t>(__double2ll_rn(component * scale_k));
    };
    return {{quantized(east_x), quantized(east_y), quantized(east_z)},
            {quantized(north_x), quantized(north_y), quantized(north_z)}};
}

/**
 *  @brief The rule of the CPU's @c surface_keep_octants_t: the nearest candidate in each of eight
 *      tangent-plane cones around the node, then the farthest of the rest, up to the row's
 *      connectivity.
 *
 *  Each lane classifies a candidate by @c octant_of, two integer dot products against the node's
 *  frame, so no distance between candidates is scored. Candidates arrive ascending, so the first
 *  one a cone sees is its nearest, and the kept positions go out ascending.
 */
struct vram_keep_octants_t {

    template <basic::vram_basic_kit kit_, typename schedule_>
    static __device__ __forceinline__ u32_t select( //
        basic::vram_ragged_view const& storage, schedule_&, typename kit_::query_t const& query, byte_t*,
        u32_t const* candidate_ids, f32_t const*, u32_t candidates_count, u32_t* kept_positions,
        dense::vram_telemetry_t&, u32_t lane) noexcept {

        using warp_t = typename kit_::warp_t;
        static_assert(dense::vram_cache_rows_k <= 64, "One bit per candidate position");
        constexpr u32_t cones_k = static_cast<u32_t>(surface_keep_octants_t::cones_k);
        tangent_frame_t const frame = cuda_tangent_frame(query);
        u64_t kept = 0;
        u32_t taken_cones = 0;
        for (u32_t base = 0; base < candidates_count; base += warp_t::width_k) { // Warp-uniform bounds.
            u32_t const position = base + lane;
            u32_t const cone = position < candidates_count
                                   ? octant_of(frame, query,
                                               kit_::prepare(storage.row(candidate_ids[position]), nullptr, lane))
                                   : cones_k;
            for (u32_t candidate_cone = 0; candidate_cone != cones_k; ++candidate_cone) {
                typename warp_t::mask_t const in_cone = warp_t::ballot(cone == candidate_cone);
                if (!in_cone || (taken_cones >> candidate_cone) & 1u) continue;
                kept |= u64_t {1} << (base + warp_t::first(in_cone));
                taken_cones |= 1u << candidate_cone;
            }
        }

        // The farthest non-winners fill the rest of the row, as the CPU's long links do.
        u32_t const budget = min_of(storage.connectivity, candidates_count);
        u32_t const winners = static_cast<u32_t>(__popcll(static_cast<long long>(kept)));
        u32_t retained = budget > winners ? budget - winners : 0;
        for (u32_t position = candidates_count; position-- != 0 && retained;)
            if (!((kept >> position) & 1u)) kept |= u64_t {1} << position, --retained;

        u32_t kept_count = 0;
        for (; kept && kept_count != storage.connectivity; kept &= kept - 1, ++kept_count)
            if (lane == 0) kept_positions[kept_count] = static_cast<u32_t>(__ffsll(static_cast<long long>(kept)) - 1);
        warp_t::sync();
        return kept_count;
    }
};

/** Whether @p target lies inside the view cone of @p viewer, as @c within_view tests it on the
 *  host: the offset turns into the viewer's east, north and up by its geodetic latitude and
 *  longitude, taken from Bowring's closed form through ratios alone. */
__device__ __forceinline__ bool cuda_within_view(earth_observation_t const& viewer,
                                                 earth_observation_t const& target) noexcept {
    constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
    constexpr f64_t equatorial_k = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS * centimetres_per_metre_k;
    constexpr f64_t polar_k = wgs84_semi_minor_axis_k * centimetres_per_metre_k;
    f64_t const x = viewer.position.x_centimetres, y = viewer.position.y_centimetres, z = viewer.position.z_centimetres;
    f64_t const offset_x = static_cast<f64_t>(target.position.x_centimetres) - x;
    f64_t const offset_y = static_cast<f64_t>(target.position.y_centimetres) - y;
    f64_t const offset_z = static_cast<f64_t>(target.position.z_centimetres) - z;
    f64_t const horizontal = nk_sqrt_f64_cuda_(x * x + y * y);
    f64_t const cos_longitude = horizontal > 0 ? x / horizontal : 1,
                sin_longitude = horizontal > 0 ? y / horizontal : 0;
    f64_t const parametric_sine = z * equatorial_k, parametric_cosine = horizontal * polar_k;
    f64_t const parametric_norm = nk_sqrt_f64_cuda_(parametric_sine * parametric_sine +
                                                    parametric_cosine * parametric_cosine);
    f64_t const sin_parametric = parametric_norm > 0 ? parametric_sine / parametric_norm : 0;
    f64_t const cos_parametric = parametric_norm > 0 ? parametric_cosine / parametric_norm : 1;
    f64_t const latitude_sine = z + wgs84_second_eccentricity_squared_k * polar_k * sin_parametric * sin_parametric *
                                        sin_parametric;
    f64_t const latitude_cosine = horizontal - wgs84_eccentricity_squared_k * equatorial_k * cos_parametric *
                                                   cos_parametric * cos_parametric;
    f64_t const latitude_norm = nk_sqrt_f64_cuda_(latitude_sine * latitude_sine + latitude_cosine * latitude_cosine);
    f64_t const sin_latitude = latitude_norm > 0 ? latitude_sine / latitude_norm : 0;
    f64_t const cos_latitude = latitude_norm > 0 ? latitude_cosine / latitude_norm : 1;
    f64_t const east = -sin_longitude * offset_x + cos_longitude * offset_y;
    f64_t const north = -sin_latitude * cos_longitude * offset_x - sin_latitude * sin_longitude * offset_y +
                        cos_latitude * offset_z;
    f64_t const up = cos_latitude * cos_longitude * offset_x + cos_latitude * sin_longitude * offset_y +
                     sin_latitude * offset_z;
    constexpr f64_t turn_k = 2 * 3.14159265358979323846;
    bool const limited = viewer.range_metres != std::numeric_limits<u16_t>::max();
    bool const pitched = viewer.elevation != std::numeric_limits<i8_t>::min();
    return !nk_terrain_outside_view_f32_cuda_(east, north, up,
                                              viewer.field ? viewer.azimuth * turn_k / azimuth_steps_k : nan_k,
                                              pitched ? viewer.elevation * turn_k / elevation_steps_k : nan_k,
                                              viewer.field ? viewer.field * turn_k / elevation_steps_k : nan_k,
                                              limited ? viewer.range_metres * centimetres_per_metre_k : nan_k);
}

/** Whether @p entry, @p chord square centimetres from @p query, matches it under @p relation short
 *  of any line of sight, as @c admits tests it on the host. */
__device__ __forceinline__ bool cuda_admits(relation_t relation, f64_t radius_metres, f64_t slack_milliseconds,
                                            earth_observation_t const& query, earth_observation_t const& entry,
                                            i64_t chord) noexcept {
    constexpr i64_t timeless_k = std::numeric_limits<i64_t>::min();
    constexpr u32_t open_ended_k = std::numeric_limits<u32_t>::max();
    constexpr f64_t infinity_k = std::numeric_limits<f64_t>::infinity();
    if (query.begin_milliseconds != timeless_k && entry.begin_milliseconds != timeless_k) {
        f64_t const query_begin = static_cast<f64_t>(query.begin_milliseconds);
        f64_t const query_end = query.duration_milliseconds == open_ended_k ? infinity_k
                                                                            : query_begin + query.duration_milliseconds;
        f64_t const entry_begin = static_cast<f64_t>(entry.begin_milliseconds);
        f64_t const entry_end = entry.duration_milliseconds == open_ended_k ? infinity_k
                                                                            : entry_begin + entry.duration_milliseconds;
        if (entry_begin > query_end + slack_milliseconds || entry_end < query_begin - slack_milliseconds) return false;
    }
    if (radius_metres > 0) {
        f64_t const deviations = nk_sqrt_f64_cuda_(static_cast<f64_t>(query.deviation_metres) * query.deviation_metres +
                                                   static_cast<f64_t>(entry.deviation_metres) * entry.deviation_metres);
        f64_t const reach = (radius_metres + 3 * deviations) * centimetres_per_metre_k;
        if (static_cast<f64_t>(chord) > reach * reach) return false;
    }
    if (relation == relation_t::sees_k) return cuda_within_view(entry, query);
    if (relation == relation_t::seen_k) return cuda_within_view(query, entry);
    return true;
}

#endif

} // namespace unum::usearch::geospatial
