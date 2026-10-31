/**
 *  @file src_stub/geospatial.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_geospatial_* C ABI, testing every stored
 *      observation against every query so a binding can be exercised without the engine.
 *
 *  The thing here most worth a binding's attention is the NaN convention: every field of an
 *  observation but the latitude and the longitude may be NaN, and each NaN only widens the tests
 *  that read it, so a binding that maps a missing value to zero instead of NaN narrows them.
 *
 *  @section geospatial_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  Observations are stored as the doubles inserted, so a @c get returns them exactly rather than
 *  within the engine's quantization. A handle refuses every terrain, so the @c visible relation is
 *  always refused and the view relations test the cones alone. Refinement keeps no edges to
 *  re-select, so its @c changed is a constant zero rather than a settling signal, and a caller must
 *  not read it as one. A view copies its checkpoint rather than mapping it, and every VRAM
 *  lifecycle verb refuses. Traversal statistics are plausible fiction: shaped correctly, invented
 *  numerically. A checkpoint is a binary blob of this file's own memory image, readable by nothing
 *  else.
 *
 *  Keeping no edges, the stand-in links every one of its entries to its exact nearest at the frozen
 *  connectivity, the lists a perfect graph would hold where the engine's are the ones it stored. A
 *  cursor walks those lists breadth-first from its start keys, a split grows its parts over them,
 *  each receiver keeping the source's storage order, and a sample walks them as the library's plan
 *  walks its graph, placing its draws as the library does.
 */
#include <cmath> // `std::sqrt`, `std::sin`, `std::cos`, `std::isnan`

#include <algorithm> // `std::min`, `std::max`
#include <array>     // `std::array`
#include <limits>    // `std::numeric_limits`
#include <memory>    // `std::unique_ptr` - a finished refinement's handle
#include <numbers>   // `std::numbers::pi`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <utility>   // `std::exchange` - a consumed donor's slot
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/geospatial_index.h"

#include "lib.hpp"

using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace unum {
namespace usearch {
namespace stub {

#pragma region Storage

using geospatial_payload_t = usearch_geospatial_observation_t;
using geospatial_entry_t = entry<geospatial_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<geospatial_entry_t>,
              "reallocation must move entries, never copy them");

/** The north-up grid of heights lines of sight are tested over, copied from what the caller gave
 *  at creation; no rows means none, and a checkpoint never carries it. */
struct terrain_grid_t {
    std::size_t rows = 0, columns = 0;
    double north_latitude = 0, west_longitude = 0, row_step = 0, column_step = 0;
    std::vector<float> heights;

    bool present() const noexcept { return rows != 0; }

    /** The height of the cell nearest @p latitude, @p longitude; NaN off the grid or where the
     *  grid holds none, which never blocks a sight line. */
    double height_at(double latitude, double longitude) const noexcept {
        if (!present()) return std::numeric_limits<double>::quiet_NaN();
        double const row = std::round((north_latitude - latitude) / row_step);
        double const column = std::round((longitude - west_longitude) / column_step);
        if (row < 0 || column < 0 || row >= static_cast<double>(rows) || column >= static_cast<double>(columns))
            return std::numeric_limits<double>::quiet_NaN();
        return heights[static_cast<std::size_t>(row) * columns + static_cast<std::size_t>(column)];
    }
};

/** The connections-per-node bound a zero derives to, as the header promises: eight tangent-plane
 *  cones and four long-range links. */
inline constexpr std::size_t geospatial_connectivity_k = 12;

/** What a handle froze at construction, or adopted from a checkpoint. */
struct geospatial_index_options_t {

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Upper bound on entries sharing one key. */
    std::size_t values_per_key;

    /** The terrain lines of sight are tested over, named again at every load and view. */
    terrain_grid_t terrain;
};

using geospatial_index_t = handle<geospatial_payload_t, geospatial_index_options_t>;

using geospatial_cursor_t = graph_cursor<geospatial_index_t>;

/** Whether a receiver frozen at @p receiver takes a part of an index frozen at @p index: the same
 *  connectivity, since every observation is scored alike. */
static bool same_kind(geospatial_index_options_t const& index, geospatial_index_options_t const& receiver) {
    return index.connectivity == receiver.connectivity;
}

/** Copies the terrain @p given into @p terrain, or fills it once through the caller's handler;
 *  false once @p outcome records a dataset path, which no residency reads yet, a terrain both
 *  borrowed and filled, or a grid without rows, columns or positive steps. */
static bool adopt_terrain(terrain_grid_t& terrain, usearch_geospatial_terrain_t const& given,
                          usearch_outcome_t* outcome) {
    if (reject_if(given.path.data != nullptr, usearch_status_unsupported_k, "No residency reads a terrain path yet",
                  outcome) ||
        reject_if(given.heights && given.callback, usearch_status_invalid_argument_k,
                  "A terrain is borrowed or filled, never both", outcome))
        return false;
    if (!given.heights && !given.callback) return true;
    if (reject_if(!given.rows || !given.columns || !(given.row_step_degrees > 0) || !(given.column_step_degrees > 0),
                  usearch_status_config_mismatch_k, "A terrain needs rows, columns and positive steps", outcome))
        return false;
    terrain.rows = given.rows, terrain.columns = given.columns;
    terrain.north_latitude = given.north_latitude_degrees, terrain.west_longitude = given.west_longitude_degrees;
    terrain.row_step = given.row_step_degrees, terrain.column_step = given.column_step_degrees;
    terrain.heights.assign(given.rows * given.columns, std::numeric_limits<float>::quiet_NaN());
    std::size_t const row_bytes = given.columns * sizeof(float);
    if (given.heights) {
        std::size_t const stride = given.heights_stride ? given.heights_stride : row_bytes;
        for (std::size_t row = 0; row != given.rows; ++row)
            std::memcpy(terrain.heights.data() + row * given.columns,
                        reinterpret_cast<std::byte const*>(given.heights) + row * stride, row_bytes);
    }
    else given.callback(0, 0, given.rows, given.columns, terrain.heights.data(), row_bytes, given.state);
    return true;
}

/** @p observation with a NaN altitude put on the ground: @p terrain's height where it has one,
 *  which the entry then stores and @c get hands back. */
static usearch_geospatial_observation_t grounded(terrain_grid_t const& terrain,
                                                 usearch_geospatial_observation_t observation) {
    if (std::isnan(observation.altitude_metres)) {
        double const ground = terrain.height_at(observation.latitude_degrees, observation.longitude_degrees);
        if (!std::isnan(ground)) observation.altitude_metres = ground;
    }
    return observation;
}

/** Refuses observations without a position or ending before they begin. */
static bool reject_observations(usearch_geospatial_observation_t const* observations, std::size_t count,
                                usearch_outcome_t* outcome) {
    if (reject_if(!observations && count, usearch_status_invalid_argument_k, "The batch must name storage", outcome))
        return true;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_geospatial_observation_t const& observation = observations[offset];
        if (reject_if(std::isnan(observation.latitude_degrees) || std::isnan(observation.longitude_degrees),
                      usearch_status_invalid_argument_k, "An observation needs a latitude and a longitude", outcome))
            return true;
        if (reject_if(observation.end_seconds < observation.begin_seconds, usearch_status_invalid_argument_k,
                      "An observation ends before it begins", outcome))
            return true;
    }
    return false;
}

/** Refuses relations @p index cannot answer: unknown ones, and @c visible without terrain. */
static bool reject_relation(geospatial_index_t const& index, usearch_geospatial_relation_t relation,
                            usearch_outcome_t* outcome) {
    if (reject_if(relation == usearch_geospatial_relation_visible_k && !index.options.terrain.present(),
                  usearch_status_unsupported_k, "Lines of sight need a handle with terrain", outcome))
        return true;
    return reject_if(relation != usearch_geospatial_relation_near_k && relation != usearch_geospatial_relation_sees_k &&
                         relation != usearch_geospatial_relation_seen_k &&
                         relation != usearch_geospatial_relation_visible_k,
                     usearch_status_unknown_kind_k, "The relation is not one this ABI names", outcome);
}

#pragma endregion Storage

#pragma region Geometry

/** Earth-fixed metres and the local east, north and up axes at one observation, on WGS-84. */
struct placed_t {
    std::array<double, 3> position;
    std::array<double, 3> east, north, up;
};

static placed_t placed(usearch_geospatial_observation_t const& observation) {
    constexpr double semi_major_k = 6378137.0, flattening_k = 1.0 / 298.257223563;
    constexpr double eccentricity_squared_k = flattening_k * (2.0 - flattening_k);
    constexpr double radians_k = std::numbers::pi / 180.0;
    double const latitude = observation.latitude_degrees * radians_k;
    double const longitude = observation.longitude_degrees * radians_k;
    double const altitude = std::isnan(observation.altitude_metres) ? 0.0 : observation.altitude_metres;
    double const sin_latitude = std::sin(latitude), cos_latitude = std::cos(latitude);
    double const sin_longitude = std::sin(longitude), cos_longitude = std::cos(longitude);
    double const normal = semi_major_k / std::sqrt(1.0 - eccentricity_squared_k * sin_latitude * sin_latitude);
    placed_t result;
    result.position = {(normal + altitude) * cos_latitude * cos_longitude,
                       (normal + altitude) * cos_latitude * sin_longitude,
                       (normal * (1.0 - eccentricity_squared_k) + altitude) * sin_latitude};
    result.east = {-sin_longitude, cos_longitude, 0.0};
    result.north = {-sin_latitude * cos_longitude, -sin_latitude * sin_longitude, cos_latitude};
    result.up = {cos_latitude * cos_longitude, cos_latitude * sin_longitude, sin_latitude};
    return result;
}

static double dot(std::array<double, 3> const& first, std::array<double, 3> const& second) {
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

/** Whether @p target lies inside the view cone of @p viewer: within its range, and within half its
 *  field, widened by three standard deviations of its heading, of where it looked. */
static bool inside_view(usearch_geospatial_observation_t const& viewer, placed_t const& from, placed_t const& target) {
    constexpr double radians_k = std::numbers::pi / 180.0;
    std::array<double, 3> const offset {target.position[0] - from.position[0], target.position[1] - from.position[1],
                                        target.position[2] - from.position[2]};
    double const east = dot(offset, from.east), north = dot(offset, from.north), up = dot(offset, from.up);
    double const slant = std::sqrt(east * east + north * north + up * up);
    if (!std::isnan(viewer.range_metres) && slant > viewer.range_metres) return false;
    if (std::isnan(viewer.heading_degrees) || std::isnan(viewer.field_of_view_degrees)) return true;
    double const widening = std::isnan(viewer.heading_accuracy_degrees) ? 0 : 3 * viewer.heading_accuracy_degrees;
    double const half_field = (viewer.field_of_view_degrees / 2 + widening) * radians_k;
    if (half_field >= std::numbers::pi) return true;
    double const azimuth = viewer.heading_degrees * radians_k;
    if (std::isnan(viewer.pitch_degrees)) {
        double const ground = std::sqrt(east * east + north * north);
        if (ground == 0) return true;
        return (east * std::sin(azimuth) + north * std::cos(azimuth)) / ground >= std::cos(half_field);
    }
    if (slant == 0) return true;
    double const elevation = viewer.pitch_degrees * radians_k;
    double const along = east * std::sin(azimuth) * std::cos(elevation) +
                         north * std::cos(azimuth) * std::cos(elevation) + up * std::sin(elevation);
    return along / slant >= std::cos(half_field);
}

/** Geodetic latitude and longitude in degrees and height in metres of an Earth-fixed
 *  @p position, by the fixed-point iteration every geodesy text gives. */
static std::array<double, 3> geodetic(std::array<double, 3> const& position) {
    constexpr double semi_major_k = 6378137.0, flattening_k = 1.0 / 298.257223563;
    constexpr double eccentricity_squared_k = flattening_k * (2.0 - flattening_k);
    constexpr double degrees_k = 180.0 / std::numbers::pi;
    double const longitude = std::atan2(position[1], position[0]);
    double const radial = std::hypot(position[0], position[1]);
    double latitude = std::atan2(position[2], radial * (1.0 - eccentricity_squared_k)), height = 0;
    for (int iteration = 0; iteration != 8; ++iteration) {
        double const sin_latitude = std::sin(latitude);
        double const normal = semi_major_k / std::sqrt(1.0 - eccentricity_squared_k * sin_latitude * sin_latitude);
        height = radial / std::cos(latitude) - normal;
        latitude = std::atan2(position[2], radial * (1.0 - eccentricity_squared_k * normal / (normal + height)));
    }
    return {latitude * degrees_k, longitude * degrees_k, height};
}

/** Whether the straight line from @p from to @p to clears @p terrain: at every sample between
 *  them the line, lifted by the refraction of light, stands above the ground beneath it. The
 *  Earth-fixed chord carries the curvature on its own. */
static bool clear_sight(terrain_grid_t const& terrain, placed_t const& from, placed_t const& to) {
    if (!terrain.present()) return true;
    std::array<double, 3> const offset {to.position[0] - from.position[0], to.position[1] - from.position[1],
                                        to.position[2] - from.position[2]};
    double const chord = std::sqrt(dot(offset, offset));
    if (chord == 0) return true;
    // Two samples per cell, so a ridge one cell wide is never stepped over.
    constexpr double metres_per_degree_k = 111320.0, refraction_k = 0.13, earth_radius_k = 6371000.0;
    double const cell = std::min(terrain.row_step, terrain.column_step) * metres_per_degree_k;
    std::size_t const samples = std::min<std::size_t>(1u << 16, static_cast<std::size_t>(2 * chord / cell) + 2);
    for (std::size_t sample = 1; sample != samples; ++sample) {
        double const share = static_cast<double>(sample) / static_cast<double>(samples);
        std::array<double, 3> const point {from.position[0] + share * offset[0], from.position[1] + share * offset[1],
                                           from.position[2] + share * offset[2]};
        std::array<double, 3> const above = geodetic(point);
        double const lift = refraction_k * (share * chord) * ((1.0 - share) * chord) / (2.0 * earth_radius_k);
        double const ground = terrain.height_at(above[0], above[1]);
        if (!std::isnan(ground) && above[2] + lift <= ground) return false;
    }
    return true;
}

/** The closed interval an observation covers, unbounded when it records no time. */
static std::array<double, 2> interval_of(usearch_geospatial_observation_t const& observation) {
    double const infinity = std::numeric_limits<double>::infinity();
    bool const has_begin = !std::isnan(observation.begin_seconds), has_end = !std::isnan(observation.end_seconds);
    if (!has_begin && !has_end) return {-infinity, infinity};
    double const begin = has_begin ? observation.begin_seconds : observation.end_seconds;
    double const end = has_end ? observation.end_seconds : observation.begin_seconds;
    return {begin, end};
}

/** The chord in metres between two observations' positions, whatever their times and views. */
static double chord_between(usearch_geospatial_observation_t const& first,
                            usearch_geospatial_observation_t const& second) {
    placed_t const from = placed(first), to = placed(second);
    std::array<double, 3> const offset {to.position[0] - from.position[0], to.position[1] - from.position[1],
                                        to.position[2] - from.position[2]};
    return std::sqrt(dot(offset, offset));
}

/** The chord in metres when @p entry matches @p query under @p relation and @p tolerance, or
 *  infinity, which sorts last where a NaN would break the sort's ordering. */
static double matched(terrain_grid_t const& terrain, usearch_geospatial_observation_t const& query,
                      placed_t const& query_placed, usearch_geospatial_observation_t const& entry,
                      usearch_geospatial_relation_t relation, usearch_geospatial_tolerance_t const& tolerance) {
    double const unmatched = std::numeric_limits<double>::infinity();
    if (!std::isnan(query.begin_seconds) || !std::isnan(query.end_seconds)) {
        std::array<double, 2> const query_interval = interval_of(query), entry_interval = interval_of(entry);
        if (entry_interval[0] > query_interval[1] + tolerance.seconds ||
            entry_interval[1] < query_interval[0] - tolerance.seconds)
            return unmatched;
    }
    placed_t const entry_placed = placed(entry);
    std::array<double, 3> const offset {entry_placed.position[0] - query_placed.position[0],
                                        entry_placed.position[1] - query_placed.position[1],
                                        entry_placed.position[2] - query_placed.position[2]};
    double const chord = std::sqrt(dot(offset, offset));
    if (tolerance.metres > 0) {
        double const query_deviation = std::isnan(query.accuracy_metres) ? 0 : query.accuracy_metres;
        double const entry_deviation = std::isnan(entry.accuracy_metres) ? 0 : entry.accuracy_metres;
        double const widening = 3 * std::sqrt(query_deviation * query_deviation + entry_deviation * entry_deviation);
        if (chord > tolerance.metres + widening) return unmatched;
    }
    if (relation == usearch_geospatial_relation_sees_k && !inside_view(entry, entry_placed, query_placed))
        return unmatched;
    if (relation == usearch_geospatial_relation_seen_k && !inside_view(query, query_placed, entry_placed))
        return unmatched;
    if (relation != usearch_geospatial_relation_near_k && !clear_sight(terrain, query_placed, entry_placed))
        return unmatched;
    return chord;
}

/** One query's whole scan: every admitted entry that matches, nearest first. */
static void rank_matches(geospatial_index_t const& index, usearch_geospatial_observation_t const& asked,
                         usearch_geospatial_relation_t relation, usearch_geospatial_tolerance_t const& tolerance,
                         std::size_t entries, usearch_predicate_t predicate, std::vector<match_t>& ranked) {
    terrain_grid_t const& terrain = index.options.terrain;
    usearch_geospatial_observation_t const query = grounded(terrain, asked);
    placed_t const query_placed = placed(query);
    us::stub::rank<geospatial_payload_t>(
        std::span<geospatial_entry_t const>(index.entries).first(entries), predicate,
        [&](geospatial_payload_t const& entry) {
            return matched(terrain, query, query_placed, entry, relation, tolerance);
        },
        ranked);
    // Unmatched entries scored infinity and sorted last, so the matches are the finite prefix.
    std::size_t matches = 0;
    while (matches != ranked.size() && ranked[matches].distance != std::numeric_limits<double>::infinity()) ++matches;
    ranked.resize(matches);
}

#pragma endregion Geometry

#pragma region Serialization

/** What @c load needs before it can read an entry. The blob is this file's own memory
 *  image rather than a format: no magic, no version, and nothing else reads it back. */
struct geospatial_checkpoint_prefix_t {

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many entries follow the prefix. */
    std::size_t entries;
};

constexpr std::size_t record_bytes_k = sizeof(usearch_key_t) + sizeof(usearch_geospatial_observation_t);

static std::optional<std::size_t> serialized_bytes(std::size_t entries) {
    return blob_bytes(sizeof(geospatial_checkpoint_prefix_t), entries, record_bytes_k);
}

static void write_into(geospatial_index_t const& index, std::size_t entries, bytes_span_t target) {
    write_pod(target, geospatial_checkpoint_prefix_t {index.options.connectivity, entries});
    bytes_span_t cursor = target.subspan(sizeof(geospatial_checkpoint_prefix_t));
    for (std::size_t position = 0; position != entries; ++position) {
        write_pod(cursor, index.entries[position].key);
        write_pod(cursor.subspan(sizeof(usearch_key_t)), index.entries[position].payload);
        cursor = cursor.subspan(record_bytes_k);
    }
}

/** Reads the prefix and checks the blob is long enough for what it claims. */
static bool parse_prefix(geospatial_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(geospatial_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<geospatial_checkpoint_prefix_t>(blob);
    return fits_blob(blob, sizeof(geospatial_checkpoint_prefix_t), prefix.entries, record_bytes_k, outcome);
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. The bound on
 *  entries per key is the loader's to choose, since no checkpoint records it. */
static bool read_from(geospatial_index_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    geospatial_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return false;
    index.reset();
    if (!index.accommodate(prefix.entries)) {
        propagate_error(usearch_status_out_of_memory_k, outcome, "The checkpoint's entries could not be allocated");
        return false;
    }
    index.options.connectivity = prefix.connectivity;
    bytes_view_t cursor = blob.subspan(sizeof(geospatial_checkpoint_prefix_t));
    for (std::size_t position = 0; position != prefix.entries; ++position) {
        usearch_key_t const key = read_pod<usearch_key_t>(cursor);
        index.append(key, read_pod<usearch_geospatial_observation_t>(cursor.subspan(sizeof(usearch_key_t))));
        cursor = cursor.subspan(record_bytes_k);
    }
    return true;
}

/** Writes @p shard's whole image into @p blob; false once @p outcome records why not. */
static bool serialize_shard(geospatial_index_t const& shard, std::vector<std::byte>& blob, usearch_outcome_t* outcome) {
    std::size_t const entries = shard.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(entries);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return false;
    blob.resize(*length);
    write_into(shard, entries, blob);
    return true;
}

/** A new handle holding the checkpoint in @p blob, or null once the outcome records why not. */
template <typename index_type_>
static index_type_* open_checkpoint(bytes_view_t blob, usearch_outcome_t* outcome) {
    index_type_* const index = new (std::nothrow) index_type_();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return nullptr;
    if (read_from(*index, blob, outcome)) return index;
    delete index;
    return nullptr;
}

#pragma endregion Serialization

} // namespace stub
} // namespace usearch
} // namespace unum

struct usearch_geospatial_index_handle_t : geospatial_index_t {};
struct usearch_geospatial_cursor_handle_t : geospatial_cursor_t {};

extern "C" {

void usearch_geospatial_init_ram(usearch_geospatial_init_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    terrain_grid_t terrain;
    if (!adopt_terrain(terrain, arguments->terrain, outcome)) return;
    usearch_geospatial_index_handle_t* const index = new (std::nothrow) usearch_geospatial_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->connectivity ? arguments->connectivity : geospatial_connectivity_k,
                      arguments->values_per_key};
    index->shard_capacity = arguments->ram.shard_capacity;
    index->options.terrain = std::move(terrain);
    *arguments->index = index;
}

void usearch_geospatial_init_vram(usearch_geospatial_init_vram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_geospatial_load_ram(usearch_geospatial_load_ram_t* arguments) {
    if (through_executor<usearch_geospatial_load_ram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    terrain_grid_t terrain;
    if (!adopt_terrain(terrain, arguments->terrain, outcome)) return;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_geospatial_index_handle_t* const index = open_parts<usearch_geospatial_index_handle_t>(
        blob, arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_geospatial_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, 1, 1);
    index->options.terrain = std::move(terrain);
    *arguments->index = index;
}

void usearch_geospatial_load_vram(usearch_geospatial_load_vram_t* arguments) {
    if (through_executor<usearch_geospatial_load_vram>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "This build carries no GPU backends");
}

void usearch_geospatial_view_ram(usearch_geospatial_view_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    terrain_grid_t terrain;
    if (!adopt_terrain(terrain, arguments->terrain, outcome)) return;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Entries are still copied out of the mapping, so only the view's refusals are mirrored, not
    // its zero cost.
    usearch_geospatial_index_handle_t* const index = open_parts<usearch_geospatial_index_handle_t>(
        served.view(), arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_geospatial_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->served = std::move(served);
    index->options.terrain = std::move(terrain);
    *arguments->index = index;
}

void usearch_geospatial_view_nvme(usearch_geospatial_view_nvme_t* arguments) {
    if (through_executor<usearch_geospatial_view_nvme>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    terrain_grid_t terrain;
    if (!adopt_terrain(terrain, arguments->terrain, outcome)) return;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Serves the drives' checkpoint as the RAM view does, mirroring only the residency's refusals.
    usearch_geospatial_index_handle_t* const index = open_parts<usearch_geospatial_index_handle_t>(
        served.view(), arguments->path,
        [&](bytes_view_t part) { return open_checkpoint<usearch_geospatial_index_handle_t>(part, outcome); }, outcome);
    if (!index) return;
    index->served = std::move(served);
    index->residency = usearch_residency_nvme_k;
    index->options.terrain = std::move(terrain);
    *arguments->index = index;
}

void usearch_geospatial_free(usearch_geospatial_index_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_geospatial_reserve(usearch_geospatial_reserve_t* arguments) {
    if (through_executor<usearch_geospatial_reserve>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    reject_if(!index.accommodate(arguments->entries), usearch_status_out_of_memory_k,
              "The capacity could not be reserved", outcome);
}

void usearch_geospatial_clear(usearch_geospatial_clear_t* arguments) {
    if (through_executor<usearch_geospatial_clear>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    index.reset();
}

void usearch_geospatial_describe(usearch_geospatial_describe_t* arguments) {
    if (through_executor<usearch_geospatial_describe>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->description = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    // Held shared like every other reader, so a size taken beside an insertion is not a data race.
    std::shared_lock readers(index.mutex);
    *arguments->description = {
        {index.options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        index.entries.size() * record_bytes_k,
        usearch_cpu_class_serial_k,
        usearch_gpu_class_unknown_k,
        index.shards(),
    };
}

void usearch_geospatial_insert(usearch_geospatial_insert_t* arguments) {
    if (through_executor<usearch_geospatial_insert>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_read_only(index, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;
    if (reject_observations(arguments->observations, count, outcome)) return;

    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    std::size_t published = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        index.append(key_at(arguments->keys, offset), grounded(index.options.terrain, arguments->observations[offset]));
        ++published;
        if (!keep_going(arguments->progress, published, count)) break;
    }
    // This modality's batch is all-or-nothing, a cancellation included, so a stopped batch unwinds
    // to the watermark rather than leaving a prefix published.
    if (published != count) {
        index.unwind(watermark);
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the batch");
    }
    if (stats) {
        stats->visited_members = published;
        stats->computed_distances = published * index.entries.size();
        stats->restarts = 1;
    }
}

void usearch_geospatial_refine(usearch_geospatial_refine_t* arguments) {
    if (through_executor<usearch_geospatial_refine>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    std::size_t const entries = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    bool cancelled = false;
    for (std::size_t entry = 0; entry != entries && !cancelled; ++entry)
        cancelled = !keep_going(arguments->progress, entry + 1, entries);
    if (arguments->stats) arguments->stats->visited_members = entries;
    // An exact scan reaches every entry it holds in one level.
    if (arguments->report) arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

void usearch_geospatial_health(usearch_geospatial_health_t* arguments) {
    if (through_executor<usearch_geospatial_health>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->reachability = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    // An exact scan reaches every entry it holds in one level.
    std::size_t const entries = index.entries.size();
    *arguments->reachability = {entries, entries, entries ? 1u : 0u};
}

void usearch_geospatial_search(usearch_geospatial_search_t* arguments) {
    if (through_executor<usearch_geospatial_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    *arguments->matches = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_relation(*arguments->index, arguments->relation, outcome)) return;
    if (reject_traversal(arguments->traversal, outcome)) return;
    if (reject_observations(arguments->queries, count, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries},
                                                      outcome);
    matches_outputs_t outputs {};
    if (!arena || !carve_ranked(*arena, count, arguments->wanted, outputs, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const entries = index.entries.size();
    if (stats && arguments->expansion_factor) stats->computed_distances += arguments->expansion_factor * count;
    std::vector<match_t> ranked;
    std::size_t answered = 0;
    for (; answered != count; ++answered) {
        rank_matches(index, arguments->queries[answered], arguments->relation, arguments->tolerance, entries,
                     arguments->predicate, ranked);
        outputs.write(answered, ranked);
        if (stats) {
            stats->visited_members += entries;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(arguments->progress, answered + 1, count)) {
            ++answered;
            break;
        }
    }
    if (reject_cancelled(answered, count, outcome)) return;
    *arguments->matches = outputs.view();
}

void usearch_geospatial_within(usearch_geospatial_within_t* arguments) {
    if (through_executor<usearch_geospatial_within>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_geospatial_tolerance_t const& tolerance = arguments->tolerance;
    if (reject_relation(*arguments->index, arguments->relation, outcome)) return;
    if (reject_traversal(arguments->traversal, outcome)) return;
    if (reject_observations(arguments->queries, count, outcome)) return;
    // Containment is measured from a radius, so a zero-metre tolerance names no ball to report -
    // the one place this verb parts with its ranked sibling.
    if (reject_if(!(tolerance.metres > 0), usearch_status_invalid_argument_k,
                  "A containment walk needs a positive radius in metres", outcome))
        return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries},
                                                      outcome);
    std::size_t* const offsets = arena ? carve<std::size_t>(*arena, count + 1, outcome) : nullptr;
    if (!offsets) return;
    std::shared_lock readers(index.mutex);

    std::size_t const entries = index.entries.size();
    std::vector<match_t> ranked, balls;
    offsets[0] = 0;
    for (std::size_t query = 0; query != count; ++query) {
        rank_matches(index, arguments->queries[query], arguments->relation, tolerance, entries, arguments->predicate,
                     ranked);
        balls.insert(balls.end(), ranked.begin(), ranked.end());
        offsets[query + 1] = balls.size();
        if (stats) {
            stats->visited_members += entries;
            stats->computed_distances += entries;
            stats->restarts += 1;
        }
        if (!keep_going(arguments->progress, query + 1, count))
            return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the walks");
    }
    std::size_t const total = balls.size();
    usearch_key_t* const keys = carve<usearch_key_t>(*arena, total, outcome);
    usearch_distance_t* const scores = keys ? carve<usearch_distance_t>(*arena, total, outcome) : nullptr;
    if (!scores) return;
    for (std::size_t match = 0; match != total; ++match)
        keys[match] = balls[match].key, scores[match] = balls[match].distance;
    *arguments->matches = {keys, scores, offsets};
    if (arguments->written) *arguments->written = total;
}

void usearch_geospatial_count(usearch_geospatial_count_t* arguments) {
    if (through_executor<usearch_geospatial_count>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    *arguments->key_stats = {index.by_key.count(uuid_of(arguments->key))};
}

void usearch_geospatial_get(usearch_geospatial_get_t* arguments) {
    if (through_executor<usearch_geospatial_get>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->observations = nullptr;
    if (arguments->written) *arguments->written = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> const entries = nodes_of(index, arguments->key);
    if (entries.empty()) return;
    usearch_geospatial_observation_t* const observations = carve<usearch_geospatial_observation_t>(
        *arena, entries.size(), outcome);
    if (!observations) return;
    for (std::size_t found = 0; found != entries.size(); ++found)
        observations[found] = index.entries[entries[found]].payload;
    *arguments->observations = observations;
    if (arguments->written) *arguments->written = entries.size();
}

void usearch_geospatial_absorb(usearch_geospatial_absorb_t* arguments) {
    if (through_executor<usearch_geospatial_absorb>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_geospatial_index_t* const donors = arguments->donors;
    for (std::size_t position = 0; position != donors_count; ++position)
        if (reject_if(!donors[position], usearch_status_invalid_argument_k, "A donor must not be null", outcome) ||
            reject_if(!same_kind(index.options, donors[position]->options), usearch_status_config_mismatch_k,
                      "A donor of another kind or geometry is refused", outcome))
            return;
    bool accommodated = true;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    // Each donor moves in whole and is consumed at once, its slot nulled, so a failure leaves it
    // and every later one live handles the caller still owns and must free.
    for (std::size_t position = 0; accommodated && position != donors_count; ++position) {
        usearch_geospatial_index_handle_t& donor = *donors[position];
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (geospatial_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        delete std::exchange(donors[position], nullptr);
    }
    if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The donor's entries could not be accommodated",
                  outcome))
        return;
    // An exact scan keeps no edges, so the closing refine re-selects nothing; a cancel stops only
    // it.
    std::size_t const entries = index.entries.size();
    bool cancelled = false;
    for (std::size_t entry = 0; entry != entries && !cancelled; ++entry)
        cancelled = !keep_going(arguments->progress, entry + 1, entries);
    if (arguments->stats) arguments->stats->visited_members = entries;
    if (arguments->report) arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

/** The stand-in's refinement: the donors it still absorbs, and how far its sweep got. Its index
 *  admits no insertion, save or other refinement until it finishes. */
struct usearch_geospatial_refinement_handle_t {

    /** The handle the refinement was opened on, only ever compared with the one each call
     *  passes. */
    usearch_geospatial_index_handle_t const* opened_on;

    /** The caller's donor slots, each nulled once consumed. */
    usearch_geospatial_index_t* donors;
    std::size_t count;

    /** Donors consumed so far, and entries the sweep visited so far. */
    std::size_t consumed;
    std::size_t swept;

    /** The receivers a split fills, the seed picking its roots, and those split off so far. */
    usearch_geospatial_index_t const* receivers;
    std::size_t receivers_count;
    us::utilities::u64_t seed;
    std::size_t split_receivers_count;

    /** Entries the absorbed donors moved in, each one a node whose links changed. */
    std::size_t moved = 0;
};

void usearch_geospatial_refinement_init(usearch_geospatial_refinement_init_t* arguments) {
    if (through_executor<usearch_geospatial_refinement_init>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->refinement = nullptr;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    if (reject_if(arguments->count && !arguments->donors, usearch_status_invalid_argument_k,
                  "The donors must name handles", outcome))
        return;
    for (std::size_t position = 0; position != arguments->count; ++position) {
        if (reject_if(!arguments->donors[position], usearch_status_invalid_argument_k, "A donor must not be null",
                      outcome) ||
            reject_if(arguments->donors[position] == arguments->index, usearch_status_config_mismatch_k,
                      "An index cannot absorb itself", outcome) ||
            reject_if(!same_kind(index.options, arguments->donors[position]->options), usearch_status_config_mismatch_k,
                      "A donor of another kind or geometry is refused", outcome))
            return;
    }
    if (reject_receivers(*arguments, outcome)) return;
    std::unique_lock writers(index.mutex);
    if (reject_refining(index, outcome)) return;
    usearch_geospatial_refinement_t const refinement = new (std::nothrow) usearch_geospatial_refinement_handle_t {
        &index, arguments->donors,    arguments->count,           0,
        0,      arguments->receivers, arguments->receivers_count, arguments->seed,
        0};
    if (reject_if(!refinement, usearch_status_out_of_memory_k, "The refinement could not be allocated", outcome))
        return;
    index.refinement = refinement;
    *arguments->refinement = refinement;
}

void usearch_geospatial_refinement_next(usearch_geospatial_refinement_next_t* arguments) {
    if (through_executor<usearch_geospatial_refinement_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->reselected) *arguments->reselected = 0;
    if (arguments->changed) *arguments->changed = 0;
    usearch_geospatial_refinement_handle_t& refinement = *arguments->refinement;
    if (reject_if(arguments->index != refinement.opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    for (; refinement.consumed != refinement.count; ++refinement.consumed) {
        usearch_geospatial_index_handle_t& donor = *refinement.donors[refinement.consumed];
        if (reject_if(!index.accommodate(index.entries.size() + donor.entries.size()), usearch_status_out_of_memory_k,
                      "The donor's entries could not be accommodated", outcome))
            return;
        for (geospatial_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        refinement.moved += donor.entries.size();
        delete std::exchange(refinement.donors[refinement.consumed], nullptr);
    }
    if (refinement.split_receivers_count != refinement.receivers_count) {
        writers.unlock();
        std::span<usearch_geospatial_index_handle_t* const> const receivers {refinement.receivers,
                                                                             refinement.receivers_count};
        if (!split_into(index, receivers, refinement.seed, chord_between, same_kind, outcome)) return;
        refinement.split_receivers_count = refinement.receivers_count;
        writers.lock();
    }
    // Each call continues the sweep where the last one stopped.
    std::size_t const entries = index.entries.size();
    bool cancelled = false;
    while (refinement.swept < entries && !cancelled)
        cancelled = !keep_going(arguments->progress, ++refinement.swept, entries);
    // The stand-in re-selects nothing, so its sweep visits every entry and changes none.
    if (arguments->reselected) *arguments->reselected = refinement.swept;
    if (arguments->changed) *arguments->changed = std::min(refinement.moved, refinement.swept);
    if (arguments->stats)
        arguments->stats->visited_members = refinement.swept, arguments->stats->computed_distances = refinement.swept;
    reject_if(cancelled && refinement.swept != entries, usearch_status_cancelled_k,
              "The caller's handler stopped the sweep", outcome);
}

void usearch_geospatial_refinement_finish(usearch_geospatial_refinement_finish_t* arguments) {
    if (through_executor<usearch_geospatial_refinement_finish>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    std::unique_ptr<usearch_geospatial_refinement_handle_t> const refinement {arguments->refinement};
    if (reject_if(arguments->index != refinement->opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    index.refinement = nullptr;
    std::size_t const entries = index.entries.size();
    if (arguments->stats)
        arguments->stats->visited_members = entries, arguments->stats->computed_distances = refinement->swept;
    if (arguments->report) {
        arguments->report->reselected = refinement->swept;
        arguments->report->changed = refinement->moved;
        arguments->report->reachability = {entries, entries, entries ? 1u : 0u, 0};
    }
}

void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (reject_if(arguments->starts_count && !arguments->starts, usearch_status_invalid_argument_k,
                  "The starts must name keys", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_geospatial_cursor_handle_t* const cursor = new (std::nothrow) usearch_geospatial_cursor_handle_t {};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    open_walk(*cursor, index, arguments->starts, arguments->starts_count);
    *arguments->cursor = cursor;
}

void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments) {
    if (through_executor<usearch_geospatial_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->written = 0;
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    usearch_geospatial_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    if (reject_if(fu::standard_atomic_ref<u32_t>(cursor.pulling).fetch_test_and_set_bit(0, std::memory_order_acquire),
                  usearch_status_contended_k, "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    pull_walk(*arguments, cursor, chord_between, *arena, outcome);
    fu::standard_atomic_ref<u32_t>(cursor.pulling).clear_bits(1, std::memory_order_release);
}

void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

/** Places a geospatial draw at its Earth-fixed position in Earth radii, the semi-major axis,
 *  from the whole centimetres the library stores. */
struct earth_fixed_coordinates_t {
    static void coordinates_of(geospatial_payload_t const& payload, f32_t* coordinates) {
        constexpr double semi_major_centimetres_k = 6378137.0 * 100.0;
        placed_t const at = placed(payload);
        for (std::size_t axis = 0; axis != 3; ++axis)
            coordinates[axis] = static_cast<f32_t>(static_cast<double>(std::llround(at.position[axis] * 100.0)) /
                                                   semi_major_centimetres_k);
    }
};

void usearch_geospatial_sample(usearch_geospatial_sample_t* arguments) {
    if (through_executor<usearch_geospatial_sample>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    draw_sample(*arguments, *arguments->index, chord_between, earth_fixed_coordinates_t {}, outcome);
}

void usearch_geospatial_split(usearch_geospatial_split_t* arguments) {
    if (through_executor<usearch_geospatial_split>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->report) *arguments->report = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    split_and_report(*arguments, index, chord_between, same_kind, outcome);
}

void usearch_geospatial_save(usearch_geospatial_save_t* arguments) {
    if (through_executor<usearch_geospatial_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_if(index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view already is its checkpoint", outcome))
        return;
    std::shared_lock readers(index.mutex);
    if (reject_refining(index, outcome)) return;
    save_parts(index, arguments->path, arguments->progress, serialize_shard, outcome);
}

void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments) {
    if (through_executor<usearch_geospatial_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome) || !read_first_part(blob, arguments->path, outcome)) return;
    geospatial_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    *arguments->metadata = {prefix.connectivity};
}

} // extern "C"
