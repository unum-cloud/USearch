/**
 *  @file include/usearch/geospatial/vram_nvidia_metric.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The device kit, its ladder, the octant selector, the relation tests, the kernels, and the
 *      members of @c vram_nvidia_index.hpp reading the kit: the GPU twins of @c ram_metric.hpp and
 *      of @c ram_index.hpp's selector and @c admits.
 *
 *  A stored row is one @c earth_observation_t, and a lane scores a candidate alone: the squared
 *  chord from the three 64-bit products of centimetre differences, exact in integers as the CPU
 *  kit scores it, rounded once to the walk's distance. Nothing past the @c ampere floor is needed.
 *  A row prepares into no span at all, since its position is its query. Only the modality's NVIDIA
 *  unit compiles this file, so dispatch never names the kit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <iterator> // `std::size` - every rung's kernels
#include <limits>   // `std::numeric_limits` - an unknown pitch, field or range
#include <utility>  // `std::index_sequence` - every rung's kernels, `std::swap` - the cone winners

#include <numkong/geospatial.h> // `nk_terrain_outside_view_f32_cuda_`, `nk_sqrt_f64_cuda_` - with its CUDA backend
#include <numkong/types.h>      // `NUMKONG_ARCH_CUDA_AMPERE_` - the kit this unit carries

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp"        // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t` - the NVIDIA warp policy
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.cuh" // `basic::cuda_kernel_body`
#include "usearch/geospatial/ram_index.hpp"
#include "usearch/geospatial/ram_metric.hpp"        // `geospatial_ladder`
#include "usearch/geospatial/vram_nvidia_index.hpp" // `cuda_admit_arguments` - the judging kernel's record

namespace unum::usearch::geospatial {
using namespace unum::usearch::utilities;

#pragma region Kits

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

    template <basic::vram_basic_kit kit_, typename schedule_, typename node_id_type_>
    static __device__ __forceinline__ u32_t select( //
        basic::vram_ragged_view<node_id_type_> const& storage, schedule_&, typename kit_::query_t const& query, byte_t*,
        node_id_type_ const* candidate_ids, f32_t const*, u32_t candidates_count, u32_t* kept_positions,
        vram_telemetry_t&, u32_t lane) noexcept {

        using warp_t = typename kit_::warp_t;
        constexpr u32_t cones_k = static_cast<u32_t>(surface_keep_octants_t::cones_k);
        tangent_frame_t const frame = cuda_tangent_frame(query);
        // A cone's first candidate is its nearest, so at most a winner per cone, ascending once
        // sorted; every lane holds the same ones.
        u32_t winners[cones_k];
        u32_t winners_count = 0, taken_cones = 0;
        for (u32_t base = 0; base < candidates_count; base += warp_t::width_k) { // Warp-uniform bounds.
            u32_t const position = base + lane;
            u32_t const cone = position < candidates_count
                                   ? octant_of(frame, query,
                                               kit_::prepare(storage.row(candidate_ids[position]), nullptr, lane))
                                   : cones_k;
            for (u32_t candidate_cone = 0; candidate_cone != cones_k; ++candidate_cone) {
                typename warp_t::mask_t const in_cone = warp_t::ballot(cone == candidate_cone);
                if (!in_cone || (taken_cones >> candidate_cone) & 1u) continue;
                winners[winners_count++] = base + warp_t::first(in_cone);
                taken_cones |= 1u << candidate_cone;
            }
        }
        for (u32_t sorted = 1; sorted < winners_count; ++sorted)
            for (u32_t index = sorted; index && winners[index - 1] > winners[index]; --index)
                std::swap(winners[index - 1], winners[index]);
        auto const won = [&](u32_t position) noexcept {
            for (u32_t index = 0; index != winners_count; ++index)
                if (winners[index] == position) return true;
            return false;
        };

        // The farthest non-winners fill the rest of the row, as the CPU's long links do: every
        // position from the lowest of them on that no cone won.
        u32_t const budget = min_of(storage.connectivity, candidates_count);
        u32_t retained = budget > winners_count ? budget - winners_count : 0;
        u32_t first_long = candidates_count;
        while (retained && first_long)
            if (!won(--first_long)) --retained;

        // Winners and long links merge ascending, the long links being every unwon position from
        // the first on.
        u32_t kept_count = 0;
        if (lane == 0) {
            u32_t next_winner = 0;
            for (u32_t position = 0; position != candidates_count && kept_count != storage.connectivity; ++position) {
                bool const winner = next_winner != winners_count && winners[next_winner] == position;
                next_winner += winner;
                if (winner || position >= first_long) kept_positions[kept_count++] = position;
            }
        }
        kept_count = warp_t::shuffle(kept_count, 0);
        warp_t::sync();
        return kept_count;
    }
};

/** Squared chords between stored observations, a lane to a candidate; rows keep their neighbors by
 *  tangent-plane cone. */
struct i32_ampere_geospatial_kit_t {
    inline static constexpr nk_capability_t capability_k = nk_cap_ampere_k;
    inline static constexpr bool compiled_k = NUMKONG_ARCH_CUDA_AMPERE_;
    inline static constexpr u32_t lanes_per_row_k = 1;

    using warp_t = cuda_warp_t;
    using query_t = earth_fixed_t;
    using selector_t = vram_keep_octants_t;

    static constexpr bool serves(std::size_t bytes) noexcept { return bytes <= sizeof(earth_observation_t); }

    /** A position is its own query, so a row prepares into no span. */
    static constexpr std::size_t span_bytes(byte_t const*, std::size_t) noexcept { return 0; }

    /** The position at the front of the row, in one 16-byte load. */
    static __device__ __forceinline__ query_t prepare(basic::vram_row_t row, byte_t*, u32_t) noexcept {
        int4 const words = *reinterpret_cast<int4 const*>(row.data);
        return {words.x, words.y, words.z};
    }

    /** No header bounds a chord. */
    static __device__ __forceinline__ distance_t bound(nk_kernel_kind_t, query_t const&, basic::vram_row_t) noexcept {
        return 0;
    }

    static __device__ __forceinline__ distance_t score(nk_kernel_kind_t, query_t const& query, basic::vram_row_t row,
                                                       distance_t, u32_t lane) noexcept {
        return static_cast<distance_t>(chord_squared(query, prepare(row, nullptr, lane)));
    }
};

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cuda_k) != 0)
struct geospatial_ladder<caps_> {
    using kit_t = typename highest_kit<caps_, i32_ampere_geospatial_kit_t>::type;
    static constexpr nk_capability_t picked_k = caps_ & kit_t::capability_k;
};

#pragma endregion Kits

#pragma region Face

template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
std::size_t vram_nvidia_index<node_id_type_, capability_, kernels_type_>::rung_serving_(std::size_t longest) noexcept {
    using kit_t = typename geospatial_ladder<capability_>::kit_t;
    return kit_t::serves(longest) ? 0 : rungs_k;
}

template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
u32_t vram_nvidia_index<node_id_type_, capability_, kernels_type_>::lanes_per_row_(std::size_t) noexcept {
    using kit_t = typename geospatial_ladder<capability_>::kit_t;
    return kit_t::lanes_per_row_k;
}

template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
std::size_t vram_nvidia_index<node_id_type_, capability_, kernels_type_>::span_bytes_(byte_t const* data,
                                                                                      std::size_t bytes) noexcept {
    using kit_t = typename geospatial_ladder<capability_>::kit_t;
    return kit_t::span_bytes(data, bytes);
}

template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
status_t vram_nvidia_index<node_id_type_, capability_, kernels_type_>::load_kernels_(int device) noexcept {
    auto [module, status] = cuda_module_t::load(device, kernels_type_::bytes());
    if (failed(status)) return status;
    module_ = std::move(module);
    vram_kernel_symbol_t symbol;
    for (basic::cuda_rung_t const instantiated : basic::cuda_rungs) {
        vram_kernel_symbol(symbol, "geospatial", "cuda_kernel", {"vram_arguments", 2, itanium_scope_t::basic_k},
                           instantiated.kernel, capability_, itanium_type<node_id_type_> {}, instantiated.slots);
        cuda_function_handle_t& function =
            kernels_[0][static_cast<u32_t>(instantiated.kernel)][gpu_rung_index(instantiated.slots)];
        if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
    }
    return status_t::success_k;
}

#pragma endregion Face

#pragma region Kernels

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

/** One warp per query keeps, in rank order, the pool entries the relation admits, a warp-width of
 *  ranks at a time, until @c wanted landed or the pool ran out. */
template <nk_capability_t capability_, typename node_id_type_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32)
    cuda_admit(cuda_admit_arguments<node_id_type_> arguments) {
    constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
    using warp_t = typename geospatial_ladder<capability_>::kit_t::warp_t;
    u32_t const lane = threadIdx.x % warp_t::width_k;
    u32_t const warps = gridDim.x * vram_warps_per_block_k;
    for (u32_t query_index = blockIdx.x * vram_warps_per_block_k + threadIdx.x / warp_t::width_k;
         query_index < arguments.queries_count; query_index += warps) {
        earth_observation_t const query = arguments.queries[query_index];
        node_id_type_ const* const pool_ids = arguments.pool_ids + u64_t {query_index} * arguments.pool;
        node_id_type_* const matched_ids = arguments.matched_ids + u64_t {query_index} * arguments.wanted;
        f64_t* const matched_metres = arguments.matched_metres + u64_t {query_index} * arguments.wanted;
        u32_t landed = 0;
        for (u32_t base = 0; base < arguments.pool && landed < arguments.wanted; base += warp_t::width_k) {
            u32_t const rank = base + lane;
            node_id_type_ const node_id = rank < arguments.pool ? pool_ids[rank] : unknown_k;
            bool admitted = false;
            i64_t chord = 0;
            if (node_id != unknown_k) {
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

/** Every NVIDIA geospatial kernel: @p kernel_ picks the walk, @p capability_ the kit through
 *  @c geospatial_ladder, @p node_id_type_ the IDs of the records, and @p slots_ the register
 *  rung. */
template <vram_kernel_t kernel_, nk_capability_t capability_, typename node_id_type_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32, basic::cuda_resident_blocks(kernel_, slots_))
    cuda_kernel(basic::vram_arguments<node_id_type_> arguments) {
    using kit_t = typename geospatial_ladder<capability_>::kit_t;
    basic::cuda_kernel_body<kernel_, kit_t, 0, slots_>(arguments);
}

template <nk_capability_t capability_, typename node_id_type_, std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(
         &cuda_kernel<basic::cuda_rungs[rungs_].kernel, capability_, node_id_type_, basic::cuda_rungs[rungs_].slots>),
     ...);
}

/** Instantiates every kernel under @p capability_ with @p node_id_type_ IDs, the judging one too;
 *  the unit compiling the fatbin names it once per node-ID type. Taking the kernels' addresses in
 *  device code makes NVCC emit them, while a host-only pass emits no launch stub. */
template <nk_capability_t capability_, typename node_id_type_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_rungs<capability_, node_id_type_>(std::make_index_sequence<std::size(basic::cuda_rungs)> {});
    static_cast<void>(&cuda_admit<capability_, node_id_type_>);
}

#pragma endregion Kernels

} // namespace unum::usearch::geospatial
