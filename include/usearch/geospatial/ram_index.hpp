/**
 *  @file include/usearch/geospatial/ram_index.hpp
 *  @author Ash Vardanian
 *  @date September 3, 2026
 *  @brief Similarity Search over observations on and above the WGS-84 ellipsoid.
 *
 *  A stored value is one @ref unum::usearch::geospatial::earth_observation_t of 32 bytes: a
 *  position as three Earth-centred, Earth-fixed integers in centimetres, altitude included, beside
 *  the rest of the observation quantized so that every rounding only widens a test. Every
 *  difference of positions is exact, and the squared chord between two of them is three
 *  subtractions and three wide multiply-adds. The chord orders identically to the great-circle arc,
 *  so the graph never evaluates a transcendental, and it is the distance every search reports.
 *
 *  One node is one observation, and a key may hold several through the index's key multimap.
 *  Every question is a walk around the query position followed by @c admits, which tests time,
 *  nearness widened by both positions' uncertainty, and view cones on what the walk found.
 *
 *  The surface is a two-dimensional manifold, and that is what the
 *  @ref unum::usearch::geospatial::surface_keep_octants_t selector exploits: the nearest neighbor
 *  in each of eight 45° tangent-plane cones contains the minimum spanning tree and routes to any
 *  node without evaluating the metric between candidates. Long-range structure comes from
 *  @ref unum::usearch::geospatial::draw_long_links, which wires four edges per node with lengths
 *  falling as the inverse square of distance, the exponent that makes greedy descent
 *  polylogarithmic on a surface.
 */
#pragma once

#include <cmath>   // `std::sin`, `std::cos`, `std::atan2`, `std::sqrt`, `std::exp`, `std::log`, `std::llround`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t`, `std::int64_t`
#include <cstring> // `std::memcpy` - a stored position read in place

#include <array>   // `std::array`
#include <bit>     // `std::bit_cast` - the codec's byte images
#include <limits>  // `std::numeric_limits`
#include <memory>  // `std::allocator`
#include <numbers> // `std::numbers::pi`
#include <span>    // `std::span`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include <numkong/geospatial.h> // `NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS` and kin

#include "usearch/utilities/types.hpp"
#include "usearch/basic/ram_index.hpp"

namespace unum::usearch::geospatial {
using namespace unum::usearch::utilities;

#pragma region Geometry

/** WGS-84 semi-minor axis, in metres. */
constexpr f64_t wgs84_semi_minor_axis_k = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS *
                                          (1.0 - 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING);

/** WGS-84 first eccentricity squared. */
constexpr f64_t wgs84_eccentricity_squared_k = (2.0 - 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING) /
                                               NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;

/** WGS-84 second eccentricity squared. */
constexpr f64_t wgs84_second_eccentricity_squared_k = (NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS *
                                                           NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS -
                                                       wgs84_semi_minor_axis_k * wgs84_semi_minor_axis_k) /
                                                      (wgs84_semi_minor_axis_k * wgs84_semi_minor_axis_k);

/** The storage unit: one hundred centimetres per metre. */
constexpr f64_t centimetres_per_metre_k = 100.0;

/** One position in Earth-centred, Earth-fixed centimetres: ±21,474 km along each axis,
 *  exact differences below 4,359 km of altitude, and a squared chord that fits 64 bits. */
struct earth_fixed_t {
    i32_t x_centimetres = 0;
    i32_t y_centimetres = 0;
    i32_t z_centimetres = 0;

    friend constexpr bool operator==(earth_fixed_t const&, earth_fixed_t const&) noexcept = default;
};

static_assert(sizeof(earth_fixed_t) == 12, "three int32 axes and nothing else");

/** One position as callers write it: geodetic degrees on WGS-84 and metres above it. */
struct geodetic_t {
    f64_t latitude_degrees = 0;
    f64_t longitude_degrees = 0;
    f64_t altitude_metres = 0;
};

/** The squared chord in square centimetres; differences widen to 64 bits before subtracting. */
constexpr i64_t chord_squared(earth_fixed_t first, earth_fixed_t second) noexcept {
    i64_t const delta_x = static_cast<i64_t>(first.x_centimetres) - static_cast<i64_t>(second.x_centimetres);
    i64_t const delta_y = static_cast<i64_t>(first.y_centimetres) - static_cast<i64_t>(second.y_centimetres);
    i64_t const delta_z = static_cast<i64_t>(first.z_centimetres) - static_cast<i64_t>(second.z_centimetres);
    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}

/** Rounds metres to centimetres, saturating at the storage range rather than wrapping. */
inline i32_t to_centimetres(f64_t metres) noexcept {
    constexpr f64_t lowest_k = static_cast<f64_t>(std::numeric_limits<i32_t>::lowest());
    constexpr f64_t highest_k = static_cast<f64_t>((std::numeric_limits<i32_t>::max)());
    return static_cast<i32_t>(std::llround(min_of(max_of(metres * centimetres_per_metre_k, lowest_k), highest_k)));
}

/** Converts a geodetic position to Earth-fixed centimetres on the WGS-84 ellipsoid. */
inline earth_fixed_t to_earth_fixed(geodetic_t geodetic) noexcept {
    constexpr f64_t degrees_to_radians_k = std::numbers::pi / 180.0;
    f64_t const latitude = geodetic.latitude_degrees * degrees_to_radians_k;
    f64_t const longitude = geodetic.longitude_degrees * degrees_to_radians_k;
    f64_t const sin_latitude = std::sin(latitude), cos_latitude = std::cos(latitude);
    f64_t const prime_vertical = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS /
                                 std::sqrt(1.0 - wgs84_eccentricity_squared_k * sin_latitude * sin_latitude);
    f64_t const horizontal = (prime_vertical + geodetic.altitude_metres) * cos_latitude;
    return {
        to_centimetres(horizontal * std::cos(longitude)),
        to_centimetres(horizontal * std::sin(longitude)),
        to_centimetres((prime_vertical * (1.0 - wgs84_eccentricity_squared_k) + geodetic.altitude_metres) *
                       sin_latitude),
    };
}

/** Converts Earth-fixed centimetres back to geodetic degrees and metres, through
 *  Bowring's closed form, accurate to well under a millimetre near the surface. */
inline geodetic_t to_geodetic(earth_fixed_t fixed) noexcept {
    constexpr f64_t radians_to_degrees_k = 180.0 / std::numbers::pi;
    f64_t const x = fixed.x_centimetres / centimetres_per_metre_k;
    f64_t const y = fixed.y_centimetres / centimetres_per_metre_k;
    f64_t const z = fixed.z_centimetres / centimetres_per_metre_k;
    f64_t const horizontal = std::sqrt(x * x + y * y);
    f64_t const parametric = std::atan2(z * NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS,
                                        horizontal * wgs84_semi_minor_axis_k);
    f64_t const sin_parametric = std::sin(parametric), cos_parametric = std::cos(parametric);
    f64_t const latitude = std::atan2(z + wgs84_second_eccentricity_squared_k * wgs84_semi_minor_axis_k *
                                              sin_parametric * sin_parametric * sin_parametric,
                                      horizontal - wgs84_eccentricity_squared_k *
                                                       NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS * cos_parametric *
                                                       cos_parametric * cos_parametric);
    f64_t const longitude = std::atan2(y, x);
    f64_t const sin_latitude = std::sin(latitude), cos_latitude = std::cos(latitude);
    f64_t const prime_vertical = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS /
                                 std::sqrt(1.0 - wgs84_eccentricity_squared_k * sin_latitude * sin_latitude);
    // Straight above a pole the horizontal component vanishes, and altitude reads off the polar
    // axis instead.
    f64_t const altitude = horizontal > 1.0 ? horizontal / cos_latitude - prime_vertical
                                            : std::abs(z) - wgs84_semi_minor_axis_k;
    return {latitude * radians_to_degrees_k, longitude * radians_to_degrees_k, altitude};
}

/** A unit vector in doubles, the scratch every frame computation passes through. */
struct unit_vector_t {
    f64_t x = 0, y = 0, z = 0;
};

/** The scale the integer tangent frame is quantized at: one unit is 2⁻³⁰. */
constexpr i64_t frame_scale_k = i64_t {1} << 30;

/**
 *  @brief An orthonormal basis of the tangent plane at one position, each axis quantized to
 *      int32 at @ref frame_scale_k, so projecting a displacement is one wide dot product.
 *
 *  The axes are named for reading and are not geographic: the basis is built against the
 *  coordinate axis least aligned with the position, which is well conditioned everywhere on
 *  the sphere, where a geographic east is undefined at the poles. Cone tests hold for any
 *  orthonormal basis, so every node may carry its own rotation.
 */
struct tangent_frame_t {
    std::array<i32_t, 3> east {};
    std::array<i32_t, 3> north {};
};

/** The unnormalized double-precision frame both @ref tangent_frame and @ref destination build
 *  on. */
struct unit_frame_t {
    unit_vector_t up, east, north;
};

/** Builds the unit frame at @p base; a position at the Earth's centre gets the identity axes. */
inline unit_frame_t unit_frame(earth_fixed_t base) noexcept {
    unit_vector_t const position {static_cast<f64_t>(base.x_centimetres), static_cast<f64_t>(base.y_centimetres),
                                  static_cast<f64_t>(base.z_centimetres)};
    f64_t const norm = std::sqrt(position.x * position.x + position.y * position.y + position.z * position.z);
    if (norm == 0) return {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}};
    unit_vector_t const up {position.x / norm, position.y / norm, position.z / norm};

    // The axis least aligned with the position keeps the cross product's length above 0.816.
    std::array<f64_t, 3> const magnitudes {std::abs(up.x), std::abs(up.y), std::abs(up.z)};
    std::size_t least_aligned = 0;
    if (magnitudes[1] < magnitudes[least_aligned]) least_aligned = 1;
    if (magnitudes[2] < magnitudes[least_aligned]) least_aligned = 2;
    unit_vector_t axis {};
    if (least_aligned == 0) axis.x = 1;
    else if (least_aligned == 1) axis.y = 1;
    else axis.z = 1;

    unit_vector_t east {axis.y * up.z - axis.z * up.y, axis.z * up.x - axis.x * up.z, axis.x * up.y - axis.y * up.x};
    f64_t const east_norm = std::sqrt(east.x * east.x + east.y * east.y + east.z * east.z);
    east = {east.x / east_norm, east.y / east_norm, east.z / east_norm};
    unit_vector_t const north {up.y * east.z - up.z * east.y, up.z * east.x - up.x * east.z,
                               up.x * east.y - up.y * east.x};
    return {up, east, north};
}

/** Quantizes one unit axis to the integer frame scale. */
inline std::array<i32_t, 3> quantize_axis(unit_vector_t axis) noexcept {
    return {static_cast<i32_t>(std::llround(axis.x * static_cast<f64_t>(frame_scale_k))),
            static_cast<i32_t>(std::llround(axis.y * static_cast<f64_t>(frame_scale_k))),
            static_cast<i32_t>(std::llround(axis.z * static_cast<f64_t>(frame_scale_k)))};
}

/** The integer tangent frame at @p base. */
inline tangent_frame_t tangent_frame(earth_fixed_t base) noexcept {
    unit_frame_t const frame = unit_frame(base);
    return {quantize_axis(frame.east), quantize_axis(frame.north)};
}

/**
 *  @brief Which of the eight 45° cones around @p base holds @p candidate: two sign bits and
 *      one magnitude compare over the integer projections of the displacement.
 *
 *  Bit 2 is the sign of the north projection, bit 1 the sign of the east projection, and
 *  bit 0 whether north dominates east. The projection of a position onto the tangent plane
 *  points along its initial great-circle bearing at any separation short of the antipode,
 *  so the cone is exact in bearing whether the candidate is a metre or a continent away.
 */
constexpr unsigned octant_of(tangent_frame_t const& frame, earth_fixed_t base, earth_fixed_t candidate) noexcept {
    i64_t const delta_x = static_cast<i64_t>(candidate.x_centimetres) - static_cast<i64_t>(base.x_centimetres);
    i64_t const delta_y = static_cast<i64_t>(candidate.y_centimetres) - static_cast<i64_t>(base.y_centimetres);
    i64_t const delta_z = static_cast<i64_t>(candidate.z_centimetres) - static_cast<i64_t>(base.z_centimetres);
    i64_t const east = delta_x * frame.east[0] + delta_y * frame.east[1] + delta_z * frame.east[2];
    i64_t const north = delta_x * frame.north[0] + delta_y * frame.north[1] + delta_z * frame.north[2];
    i64_t const east_magnitude = east < 0 ? -east : east;
    i64_t const north_magnitude = north < 0 ? -north : north;
    return (static_cast<unsigned>(north < 0) << 2u) | (static_cast<unsigned>(east < 0) << 1u) |
           static_cast<unsigned>(east_magnitude < north_magnitude);
}

/** The position reached from @p origin after @p range_centimetres along the great circle at
 *  @p bearing_radians, on the sphere of the origin's own radius - the target a long-link draw
 *  walks toward. */
inline earth_fixed_t destination(earth_fixed_t origin, f64_t range_centimetres, f64_t bearing_radians) noexcept {
    f64_t const radius = std::sqrt(static_cast<f64_t>(chord_squared(origin, earth_fixed_t {})));
    if (radius == 0) return origin;
    unit_frame_t const frame = unit_frame(origin);
    f64_t const angle = range_centimetres / radius;
    f64_t const along_up = std::cos(angle) * radius;
    f64_t const along_north = std::sin(angle) * std::cos(bearing_radians) * radius;
    f64_t const along_east = std::sin(angle) * std::sin(bearing_radians) * radius;
    auto const component = [&](f64_t up, f64_t north, f64_t east) noexcept {
        return to_centimetres((along_up * up + along_north * north + along_east * east) / centimetres_per_metre_k);
    };
    return {component(frame.up.x, frame.north.x, frame.east.x), component(frame.up.y, frame.north.y, frame.east.y),
            component(frame.up.z, frame.north.z, frame.east.z)};
}

#pragma endregion Geometry

#pragma region Observations

/** One observation in natural units, NaN wherever nothing was recorded: the C++ twin of the C
 *  interface's @c usearch_geospatial_observation_t, which documents every field. */
struct geodetic_observation_t {
    f64_t latitude_degrees = 0;
    f64_t longitude_degrees = 0;
    f64_t altitude_metres = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t accuracy_metres = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t heading_degrees = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t pitch_degrees = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t heading_accuracy_degrees = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t field_of_view_degrees = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t range_metres = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t begin_seconds = std::numeric_limits<f64_t>::quiet_NaN();
    f64_t end_seconds = std::numeric_limits<f64_t>::quiet_NaN();
};

/** What one entry stores: its position with altitude, and the rest of its observation quantized so
 *  that every rounding only widens a test. */
struct earth_observation_t {

    /** Earth-fixed centimetres, altitude included. */
    earth_fixed_t position;

    /** One standard deviation of the position, in whole metres rounded up and saturating; zero when
     *  unknown. */
    u16_t deviation_metres = 0;

    /** Where the device looked, clockwise from true north, in 2⁻¹⁶ turns. */
    u16_t azimuth = 0;

    /** The first instant covered, in Unix milliseconds; the minimum means no time was recorded. */
    i64_t begin_milliseconds = (std::numeric_limits<i64_t>::min)();

    /** Milliseconds from the first instant to the last; the maximum means open-ended. */
    u32_t duration_milliseconds = 0;

    /** How far the device saw, in whole metres rounded up; the maximum means without limit. */
    u16_t range_metres = (std::numeric_limits<u16_t>::max)();

    /** The pitch above the horizon in 2⁻⁸ turns; the minimum means unknown, which tests the azimuth
     *  alone. */
    i8_t elevation = (std::numeric_limits<i8_t>::min)();

    /** The full field of view in 2⁻⁸ turns, widened by three standard deviations of the heading on
     *  each side and by one step for the rounding of the angles; zero looks everywhere. */
    u8_t field = 0;

    friend constexpr bool operator==(earth_observation_t const&, earth_observation_t const&) noexcept = default;
};

static_assert(sizeof(earth_observation_t) == 32, "position, view and interval in a power of two");

/** Turns of the angles an observation stores, per full circle of its azimuth and of its pitch and
 *  field. */
constexpr f64_t azimuth_steps_k = 65536.0, elevation_steps_k = 256.0;

/** @p value rounded up into [0, @p limit], with NaN at zero. */
inline f64_t rounded_up_within(f64_t value, f64_t limit) noexcept {
    if (!(value > 0)) return 0;
    f64_t const rounded = std::ceil(value);
    return rounded < limit ? rounded : limit;
}

/** Encodes @p observation, placing it @p ground_metres above the ellipsoid when it records no
 *  altitude. */
inline earth_observation_t to_earth_observation(geodetic_observation_t const& observation,
                                                f64_t ground_metres = 0) noexcept {
    earth_observation_t stored;
    f64_t const altitude = std::isnan(observation.altitude_metres) ? ground_metres : observation.altitude_metres;
    stored.position = to_earth_fixed({observation.latitude_degrees, observation.longitude_degrees, altitude});
    stored.deviation_metres = static_cast<u16_t>(rounded_up_within(observation.accuracy_metres, 65535));
    if (!std::isnan(observation.range_metres))
        stored.range_metres = static_cast<u16_t>(rounded_up_within(observation.range_metres, 65535));

    // A field rounds up by a step beyond the heading's three deviations, covering the rounding of
    // every angle.
    if (!std::isnan(observation.heading_degrees) && !std::isnan(observation.field_of_view_degrees)) {
        f64_t const heading_widening = std::isnan(observation.heading_accuracy_degrees)
                                           ? 0
                                           : 6 * observation.heading_accuracy_degrees;
        f64_t const field = observation.field_of_view_degrees + heading_widening + 360.0 / elevation_steps_k;
        if (field < 360.0) {
            stored.field = static_cast<u8_t>(rounded_up_within(field / 360.0 * elevation_steps_k, 255));
            f64_t const turns = observation.heading_degrees / 360.0;
            stored.azimuth = static_cast<u16_t>(std::llround((turns - std::floor(turns)) * azimuth_steps_k) & 0xFFFF);
        }
    }
    if (!std::isnan(observation.pitch_degrees)) {
        f64_t const pitch = min_of(max_of(observation.pitch_degrees, -90.0), 90.0);
        stored.elevation = static_cast<i8_t>(std::llround(pitch / 360.0 * elevation_steps_k));
    }

    bool const has_begin = !std::isnan(observation.begin_seconds), has_end = !std::isnan(observation.end_seconds);
    if (has_begin || has_end) {
        f64_t const begin = has_begin ? observation.begin_seconds : observation.end_seconds;
        f64_t const end = has_end ? observation.end_seconds : observation.begin_seconds;
        stored.begin_milliseconds = static_cast<i64_t>(std::floor(begin * 1000.0));
        f64_t const duration = std::ceil(end * 1000.0) - static_cast<f64_t>(stored.begin_milliseconds);
        constexpr f64_t open_ended_k = static_cast<f64_t>((std::numeric_limits<u32_t>::max)());
        stored.duration_milliseconds = duration < open_ended_k ? static_cast<u32_t>(duration)
                                                               : (std::numeric_limits<u32_t>::max)();
    }
    return stored;
}

/** Decodes @p stored back into natural units, within the rounding @ref to_earth_observation
 *  applied; the heading's deviation comes back folded into the field of view. */
inline geodetic_observation_t to_geodetic_observation(earth_observation_t const& stored) noexcept {
    geodetic_observation_t observation;
    geodetic_t const geodetic = to_geodetic(stored.position);
    observation.latitude_degrees = geodetic.latitude_degrees;
    observation.longitude_degrees = geodetic.longitude_degrees;
    observation.altitude_metres = geodetic.altitude_metres;
    observation.accuracy_metres = stored.deviation_metres;
    if (stored.range_metres != (std::numeric_limits<u16_t>::max)()) observation.range_metres = stored.range_metres;
    if (stored.field) {
        observation.heading_degrees = stored.azimuth * 360.0 / azimuth_steps_k;
        observation.field_of_view_degrees = stored.field * 360.0 / elevation_steps_k;
    }
    if (stored.elevation != (std::numeric_limits<i8_t>::min)())
        observation.pitch_degrees = stored.elevation * 360.0 / elevation_steps_k;
    if (stored.begin_milliseconds != (std::numeric_limits<i64_t>::min)()) {
        observation.begin_seconds = static_cast<f64_t>(stored.begin_milliseconds) / 1000.0;
        observation.end_seconds = stored.duration_milliseconds == (std::numeric_limits<u32_t>::max)()
                                      ? std::numeric_limits<f64_t>::infinity()
                                      : static_cast<f64_t>(stored.begin_milliseconds + stored.duration_milliseconds) /
                                            1000.0;
    }
    return observation;
}

/** Which entries count as matches of a query; time and the caller's predicate narrow every relation
 *  alike. */
enum class relation_t : u32_t {

    /** Within the radius of each other, widened by three standard deviations of both positions
     *  combined. */
    near_k = 1,

    /** Near, with a clear line of sight over the terrain. */
    visible_k = 2,

    /** Near, and the entry's view cone holds the query. */
    sees_k = 3,

    /** Near, and the query's view cone holds the entry. */
    seen_k = 4,
};

/** The east, north and up projections of @p offset, in centimetres, at the geodetic position of
 *  @p base. */
inline std::array<f64_t, 3> local_offset(earth_fixed_t base, std::array<f64_t, 3> const& offset) noexcept {
    constexpr f64_t radians_k = std::numbers::pi / 180.0;
    geodetic_t const geodetic = to_geodetic(base);
    f64_t const latitude = geodetic.latitude_degrees * radians_k, longitude = geodetic.longitude_degrees * radians_k;
    f64_t const sin_latitude = std::sin(latitude), cos_latitude = std::cos(latitude);
    f64_t const sin_longitude = std::sin(longitude), cos_longitude = std::cos(longitude);
    return {
        -sin_longitude * offset[0] + cos_longitude * offset[1],
        -sin_latitude * cos_longitude * offset[0] - sin_latitude * sin_longitude * offset[1] + cos_latitude * offset[2],
        cos_latitude * cos_longitude * offset[0] + cos_latitude * sin_longitude * offset[1] + sin_latitude * offset[2]};
}

/** Whether @p target lies inside the view cone of @p viewer: within its range, and within half its
 *  field of where it looked, or of its azimuth alone when its pitch is unknown. */
inline bool within_view(earth_observation_t const& viewer, earth_observation_t const& target) noexcept {
    std::array<f64_t, 3> const offset {
        static_cast<f64_t>(target.position.x_centimetres) - static_cast<f64_t>(viewer.position.x_centimetres),
        static_cast<f64_t>(target.position.y_centimetres) - static_cast<f64_t>(viewer.position.y_centimetres),
        static_cast<f64_t>(target.position.z_centimetres) - static_cast<f64_t>(viewer.position.z_centimetres)};
    f64_t const slant = std::sqrt(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]);
    if (viewer.range_metres != (std::numeric_limits<u16_t>::max)() &&
        slant > viewer.range_metres * centimetres_per_metre_k)
        return false;
    if (!viewer.field || slant == 0) return true;
    std::array<f64_t, 3> const local = local_offset(viewer.position, offset);
    f64_t const half_field = viewer.field * std::numbers::pi / elevation_steps_k;
    f64_t const azimuth = viewer.azimuth * 2 * std::numbers::pi / azimuth_steps_k;
    if (viewer.elevation == (std::numeric_limits<i8_t>::min)()) {
        f64_t const ground = std::sqrt(local[0] * local[0] + local[1] * local[1]);
        if (ground == 0) return true;
        return (local[0] * std::sin(azimuth) + local[1] * std::cos(azimuth)) / ground >= std::cos(half_field);
    }
    f64_t const elevation = viewer.elevation * 2 * std::numbers::pi / elevation_steps_k;
    f64_t const along = (local[0] * std::sin(azimuth) + local[1] * std::cos(azimuth)) * std::cos(elevation) +
                        local[2] * std::sin(elevation);
    return along / slant >= std::cos(half_field);
}

/** Whether @p entry matches @p query under @p relation, short of any line of sight, which the
 *  terrain's owner tests after: the time intervals overlap within @p slack_milliseconds when the
 *  query has one, the chord stays within @p radius_metres widened by three standard deviations of
 *  both positions combined unless the radius is zero, and for @c sees and @c seen the viewer's cone
 *  holds the other. */
inline bool admits(relation_t relation, f64_t radius_metres, f64_t slack_milliseconds, earth_observation_t const& query,
                   earth_observation_t const& entry) noexcept {
    constexpr i64_t timeless_k = (std::numeric_limits<i64_t>::min)();
    constexpr u32_t open_ended_k = (std::numeric_limits<u32_t>::max)();
    if (query.begin_milliseconds != timeless_k && entry.begin_milliseconds != timeless_k) {
        f64_t const infinity = std::numeric_limits<f64_t>::infinity();
        f64_t const query_begin = static_cast<f64_t>(query.begin_milliseconds);
        f64_t const query_end = query.duration_milliseconds == open_ended_k ? infinity
                                                                            : query_begin + query.duration_milliseconds;
        f64_t const entry_begin = static_cast<f64_t>(entry.begin_milliseconds);
        f64_t const entry_end = entry.duration_milliseconds == open_ended_k ? infinity
                                                                            : entry_begin + entry.duration_milliseconds;
        if (entry_begin > query_end + slack_milliseconds || entry_end < query_begin - slack_milliseconds) return false;
    }
    if (radius_metres > 0) {
        f64_t const deviations = std::sqrt(static_cast<f64_t>(query.deviation_metres) * query.deviation_metres +
                                           static_cast<f64_t>(entry.deviation_metres) * entry.deviation_metres);
        f64_t const reach = (radius_metres + 3 * deviations) * centimetres_per_metre_k;
        if (static_cast<f64_t>(chord_squared(query.position, entry.position)) > reach * reach) return false;
    }
    if (relation == relation_t::sees_k) return within_view(entry, query);
    if (relation == relation_t::seen_k) return within_view(query, entry);
    return true;
}

#pragma endregion Observations

#pragma region Terrain

/** The share of the Earth's curvature light bends with, which lowers the bulge under every sight
 *  line: a seventh. */
inline constexpr f64_t terrain_refraction_k = 1.0 / 7;

/** Fills @p rows × @p columns heights from cell (@p first_row, @p first_column), NaN where the
 *  source has none. */
using terrain_callback_t = void (*)(std::size_t first_row, std::size_t first_column, std::size_t rows,
                                    std::size_t columns, f32_t* heights, std::size_t heights_stride, void* state);

/** The elevation lines of sight are tested over: a north-up geographic grid of heights above the
 *  ellipsoid, held by the caller as one buffer or filled on demand by @c callback. */
struct terrain_t {
    std::size_t rows = 0;
    std::size_t columns = 0;
    f64_t north_latitude_degrees = 0;
    f64_t west_longitude_degrees = 0;
    f64_t row_step_degrees = 0;
    f64_t column_step_degrees = 0;
    f32_t const* heights = nullptr;
    std::size_t heights_stride = 0;
    terrain_callback_t callback = nullptr;
    void* state = nullptr;

    /** Whether a grid is described at all. */
    bool empty() const noexcept { return !rows || !columns || (!heights && !callback); }

    /** Ground metres of one row step and of one column step, at the grid's middle row on the mean
     *  sphere. */
    std::array<f64_t, 2> step_metres() const noexcept {
        constexpr f64_t radians_k = std::numbers::pi / 180.0;
        f64_t const middle_latitude = north_latitude_degrees - row_step_degrees * static_cast<f64_t>(rows) / 2;
        return {row_step_degrees * radians_k * NUMKONG_EARTH_MEAN_RADIUS,
                column_step_degrees * radians_k * NUMKONG_EARTH_MEAN_RADIUS * std::cos(middle_latitude * radians_k)};
    }

    /** Where @p position falls: its fractional column and row, and its height above the
     *  ellipsoid. */
    std::array<f64_t, 3> grid_point(earth_fixed_t position) const noexcept {
        geodetic_t const geodetic = to_geodetic(position);
        return {(geodetic.longitude_degrees - west_longitude_degrees) / column_step_degrees,
                (north_latitude_degrees - geodetic.latitude_degrees) / row_step_degrees, geodetic.altitude_metres};
    }

    /** The fractional column and row of a geodetic latitude and longitude. */
    std::array<f64_t, 2> grid_point(f64_t latitude_degrees, f64_t longitude_degrees) const noexcept {
        return {(longitude_degrees - west_longitude_degrees) / column_step_degrees,
                (north_latitude_degrees - latitude_degrees) / row_step_degrees};
    }
};

#pragma endregion Terrain

#pragma region Selector

/**
 *  @brief Keeps the nearest candidate in each of eight tangent-plane cones around the base, then
 *      the longest of the remaining candidates up to the budget, so the neighbor list's long links
 *      outlive the reverse pass that re-selects a full neighbor list.
 *
 *  Never evaluates the metric between candidates: each candidate costs two integer dot
 *  products against the base's frame. Because the candidates arrive ascending, the first
 *  candidate seen in a cone is its nearest. The base is read from the metric's open round.
 *
 *  The candidates decide exactness. A cone facing an ocean or a desert has its nearest neighbor far
 *  beyond the nearest few dozen candidates, so insert with an expansion of at least 128; the
 *  searcher's default of twice the budget leaves a quarter of the cone edges inexact on skewed
 *  data. And any candidates that re-select a published node's neighbors must contain them:
 *  selecting from fresh search results alone keeps their far tail as the long links and drops the
 *  real ones, and never converges.
 */
struct surface_keep_octants_t {

    /** The candidates an insertion needs for exact cone winners on skewed data, as the type
     *  documents. */
    inline static constexpr std::size_t exact_candidates_k = 128;

    /** The neighbor list budget: eight cone winners and four long links. */
    std::size_t connectivity = 12;

    /** Cones around every node; strictly under 60° each, which the routing bound needs. */
    static constexpr std::size_t cones_k = 8;

    template <typename index_type_, typename metric_type_, typename context_type_>
    void operator()(index_type_ const& index, metric_type_& metric, context_type_& context) const noexcept {
        constexpr std::size_t missing_k = (std::numeric_limits<std::size_t>::max)();
        auto const& candidates = context.results;
        auto& selected = context.selected_neighbors;
        selected.clear();
        std::size_t const candidates_count = candidates.size();
        std::size_t const budget = connectivity < candidates_count ? connectivity : candidates_count;
        if (!budget) return;

        earth_fixed_t const base = metric.query();
        tangent_frame_t const frame = tangent_frame(base);
        std::array<std::size_t, cones_k> first_in_cone;
        first_in_cone.fill(missing_k);
        std::size_t winners = 0;
        for (std::size_t position = 0; position != candidates_count && winners != cones_k; ++position) {
            unsigned const cone = octant_of(frame, base,
                                            index.published_value_of(candidates[position].node_id).position);
            if (first_in_cone[cone] != missing_k) continue;
            first_in_cone[cone] = position;
            ++winners;
        }
        auto const is_winner = [&](std::size_t position) noexcept {
            for (std::size_t cone = 0; cone != cones_k; ++cone)
                if (first_in_cone[cone] == position) return true;
            return false;
        };

        // The longest non-winners fill the rest of the budget; `cut` is where they begin in the
        // ascending candidates.
        std::size_t const retained = budget > winners ? budget - winners : 0;
        std::size_t cut = candidates_count;
        for (std::size_t seen = 0; seen != retained && cut != 0;) {
            --cut;
            if (!is_winner(cut)) ++seen;
        }

        for (std::size_t position = 0; position != candidates_count; ++position)
            if (is_winner(position) || position >= cut) selected.push_back_reserved(candidates[position]);
    }
};

#pragma endregion Selector

#pragma region Serialization and Index

/** Serializes an observation as its 32 bytes, in native byte order: a SafeGraphs value column of
 *  @c u8 bytes, written under the @c chord metric token. */
struct earth_observation_codec_t {
    using bytes_image_t = std::array<byte_t, sizeof(earth_observation_t)>;

    inline static constexpr dtype_t value_dtype_k = dtype_t::u8_k;

    std::size_t storage_requirements(earth_observation_t const&) const noexcept { return sizeof(earth_observation_t); }

    /**
     *  Writes @p value as its 32 bytes at the front of @p output.
     *
     *  @par Statuses:
     *  - @c overflow_k @p output is shorter than one observation.
     */
    status_t save(earth_observation_t const& value, bytes_span_t output) const noexcept {
        if (output.size() < sizeof(earth_observation_t)) return status_t::overflow_k;
        bytes_image_t const image = std::bit_cast<bytes_image_t>(value);
        return copy_bytes(output.first(image.size()), image);
    }

    /**
     *  Reads @p value back from the 32 bytes of @p input.
     *
     *  @par Statuses:
     *  - @c malformed_k @p input is not exactly one observation long.
     */
    status_t load(bytes_view_t input, earth_observation_t& value) const noexcept {
        if (input.size() != sizeof(earth_observation_t)) return status_t::malformed_k;
        bytes_image_t image;
        if (status_t const copied = copy_bytes(image, input); failed(copied)) return copied;
        value = std::bit_cast<earth_observation_t>(image);
        return status_t::success_k;
    }

    /** What @ref view hands out for a stored position. */
    using view_t = earth_observation_t;

    /** The observation a stored @p position holds, its 32 bytes loaded straight into the value,
     *  since an I/O buffer promises no alignment to point into. */
    view_t view(bytes_view_t position) const noexcept {
        earth_observation_t value;
        std::memcpy(&value, position.data(), sizeof(value));
        return value;
    }
};

static_assert(basic::value_codec<earth_observation_codec_t, earth_observation_t>);

/** Similarity Search over observations: 32-byte values, squared chords in square centimetres as
 *  distances, and 32-bit node IDs. */
template <typename key_type_ = uuid128_t, typename wait_type_ = dummy_waiter_t,
          template <typename> class atomic_ref_template_ = fu::standard_atomic_ref,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = dummy_cache_hints_t>
using ram_index = basic::ram_index<earth_observation_t, key_type_, u32_t, nk_u64_t, wait_type_, atomic_ref_template_,
                                   allocator_type_, cache_hints_type_>;

using ram_index_t = ram_index<>;

#pragma endregion Serialization and Index

#pragma region Walks

/** One uniform draw in [0, 1) from 53 fresh bits of @p generator. */
template <typename generator_type_>
f64_t uniform_unit(generator_type_& generator) noexcept {
    return static_cast<f64_t>(static_cast<u64_t>(generator()) >> 11u) * 0x1.0p-53;
}

/**
 *  @brief Wires @p draws long links out of one published node, each to the node nearest a
 *      target drawn at a log-uniform range above the node's own spacing and a uniform
 *      bearing - the inverse-square law in distance, Kleinberg's exponent for a surface.
 *
 *  Costs one walk to measure the spacing and one per draw. A draw that falls back on the
 *  node, or on a neighbor it already links to, wires nothing and is not retried.
 *
 *  @param[in,out] index The index holding @p node, published.
 *  @param[in] node The node whose neighbor list receives the links.
 *  @param[in] draws How many links to draw; four is the design's budget.
 *  @param[in,out] metric This thread's scoring instance.
 *  @param[in,out] context The calling thread's scratch, over any allocator.
 *  @param[in,out] generator The thread's random generator; seed distinctly per thread.
 *
 *  @par Statuses:
 *  - @c overflow_k A scratch capacity of @p context, or the grown neighbor list, wraps when
 *    rounded.
 *  - @c malformed_k @p node is unpublished.
 */
template <typename index_type_, typename metric_type_, typename context_type_, typename generator_type_>
status_t draw_long_links(index_type_& index, typename index_type_::node_id_t node, std::size_t draws,
                         metric_type_& metric, context_type_& context, generator_type_& generator) noexcept {
    using node_id_t = typename index_type_::node_id_t;
    using search_config_t = typename index_type_::search_config_t;
    earth_fixed_t const origin = index.published_value_of(node).position;

    // The spacing is the chord to the nearest other node; a lone node draws nothing.
    auto const around = index.search(origin, search_config_t {4}, metric, context);
    if (failed(around.status)) return around.status;
    f64_t spacing = 0;
    for (std::size_t position = 0; position != around.value.count && spacing == 0; ++position)
        if (context.results[position].node_id != node && context.results[position].distance > 0)
            spacing = std::sqrt(static_cast<f64_t>(context.results[position].distance));
    if (spacing == 0) return status_t::success_k;
    // Half the equatorial circumference is the longest range a long link draws.
    f64_t const log_reach = std::log(std::numbers::pi * NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS *
                                     centimetres_per_metre_k / spacing);
    if (log_reach <= 0) return status_t::success_k;

    for (std::size_t draw = 0; draw != draws; ++draw) {
        f64_t const range = spacing * std::exp(uniform_unit(generator) * log_reach);
        f64_t const bearing = uniform_unit(generator) * 2.0 * std::numbers::pi;
        auto const found = index.search(destination(origin, range, bearing), search_config_t {}, metric, context);
        if (failed(found.status)) return found.status;
        node_id_t target = node;
        for (std::size_t position = 0; position != found.value.count && target == node; ++position)
            target = context.results[position].node_id;
        if (target == node) continue;
        auto const linked = index.link(node, target);
        if (failed(linked.status)) return linked.status;
    }
    return status_t::success_k;
}

/** How far a radius walk widens its candidates before trusting them complete. */
enum class radius_margin_t : u32_t {

    /** The spanner bound: complete on the exact cone graph, at about thirty times the ball. */
    spanner_k = 0,

    /** No widening: the candidates stop at the radius, complete only where the beam happened to
     *  look. */
    none_k = 1,
};

/** The spanning ratio of the eight-cone graph the completeness bound rests on. */
constexpr f64_t cone_graph_stretch_k = 2.4;

/** Outcome of one radius walk; the matches stay ascending in @c context.results. */
struct [[nodiscard]] radius_result_t {

    /** @c success_k, or why the walk could not run. */
    status_t status = status_t::success_k;

    /** Leading entries of @c context.results within the radius. */
    std::size_t count = 0;

    /** Entries the widened candidates hold in total, the cost of the margin. */
    std::size_t candidates_count = 0;

    explicit operator bool() const noexcept { return succeeded(status); }
};

/**
 *  @brief Finds every published node within @p radius_centimetres of @p query by doubling the
 *      candidates until the full candidates' tail passes the widened radius, then counting the
 *      leading matches.
 *
 *  Under @c spanner_k the candidates widen to the radius times one plus twice the stretch, which
 *  the cone graph's spanner property makes complete; under @c none_k they stop at the radius
 *  itself. The candidates double from the searcher's default breadth to the index size, and a
 *  doubling that finds nothing new ends it. Each round scores the nodes whose neighbors fill a
 *  walk's candidates, and a staging that fills ends a round early; a staging under one neighbor
 *  list reports @c config_mismatch_k.
 *
 *  @param[in,out] index The index to search.
 *  @param[in] query The position to search around.
 *  @param[in] radius_centimetres The radius the matches lie within.
 *  @param[in] margin Whether the candidates widen by the cone graph's stretch, @c spanner_k, or
 *      stop at the radius.
 *  @param[in] widest_list The most neighbors any node holds: the selector's budget plus the long
 *      links drawn past it.
 *  @param[in,out] metric The metric the candidates are scored with.
 *  @param[in,out] context The traversal context the walks draw their buffers from.
 */
template <typename index_type_, typename metric_type_, typename query_type_, typename context_type_>
radius_result_t within(index_type_& index, query_type_ const& query, i64_t radius_centimetres, radius_margin_t margin,
                       std::size_t widest_list, metric_type_& metric, context_type_& context) noexcept {
    using search_config_t = typename index_type_::search_config_t;
    using distance_t = typename index_type_::distance_t;
    if (metric.staging_limit() < widest_list) return {status_t::config_mismatch_k, 0, 0};
    f64_t const radius = static_cast<f64_t>(radius_centimetres);
    f64_t const widened = margin == radius_margin_t::spanner_k ? radius * (1.0 + 2.0 * cone_graph_stretch_k) : radius;
    f64_t const widened_squared = widened * widened;
    distance_t const radius_squared = static_cast<distance_t>(radius_centimetres * radius_centimetres);
    std::size_t const published = index.size();

    std::size_t expansion = index.non_zero_expansion_factor(search_config_t {});
    std::size_t candidates_count = 0;
    std::size_t previous_candidates_count = 0;
    for (;;) {
        auto const searched = index.search(query, search_config_t {expansion}, metric, context);
        if (failed(searched.status)) return {searched.status, 0, 0};
        candidates_count = searched.value.count;
        bool const covered = candidates_count == expansion &&
                             static_cast<f64_t>(context.results[candidates_count - 1].distance) > widened_squared;
        bool const exhausted = expansion >= published || candidates_count == previous_candidates_count;
        if (covered || exhausted) break;
        previous_candidates_count = candidates_count;
        expansion *= 2;
    }
    std::size_t count = 0;
    while (count != candidates_count && context.results[count].distance <= radius_squared) ++count;
    return {status_t::success_k, count, candidates_count};
}

#pragma endregion Walks

} // namespace unum::usearch::geospatial
