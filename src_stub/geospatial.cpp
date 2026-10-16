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
 *  else. Keeping no edges, a cursor opened at a key steps to its entries' exact nearest by chord at
 *  the frozen connectivity, and a sample links each draw to its exact nearest earlier draws - the
 *  lists a perfect graph would hold, where the engine's are the ones it stored.
 */
#include <cmath> // `std::sqrt`, `std::sin`, `std::cos`, `std::isnan`

#include <algorithm> // `std::min`, `std::max`
#include <array>     // `std::array`
#include <limits>    // `std::numeric_limits`
#include <memory>    // `std::unique_ptr` - a finished refinement's handle
#include <numbers>   // `std::numbers::pi`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <utility>   // `std::exchange` - a landed donor's slot
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

static_assert(std::is_nothrow_move_constructible_v<geospatial_entry_t>, "reallocation must move rows, never copy them");

/** What a handle froze at construction, or adopted from a checkpoint. */
struct geospatial_index_options_t {

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** Upper bound on entries sharing one key. */
    std::size_t values_per_key;
};

using geospatial_index_t = handle<geospatial_payload_t, geospatial_index_options_t>;

struct geospatial_cursor_t {

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    geospatial_index_t const* opened_on;

    /** The next entry to hand out. */
    std::size_t position;

    /** The creation watermark, which later insertions cannot raise, or the @c linked count. */
    std::size_t limit;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag pulling;

    /** The entries a step from a key yields, in link order; empty for a scan, which yields them as
     *  stored. */
    std::vector<std::size_t> linked;
};

/** Refuses any terrain, which this stand-in cannot test lines of sight over. */
static bool reject_terrain(usearch_geospatial_terrain_t const& terrain, usearch_outcome_t* outcome) {
    return reject_if(terrain.heights || terrain.callback || terrain.path.data, usearch_status_unsupported_k,
                     "The stub tests no lines of sight, so it holds no terrain", outcome);
}

/** Refuses observations without a position or ending before they begin. */
static bool reject_observations(usearch_geospatial_observation_t const* observations, std::size_t count,
                                usearch_outcome_t* outcome) {
    if (reject_if(!observations && count, usearch_status_invalid_argument_k, "The batch must name storage", outcome))
        return true;
    for (std::size_t row = 0; row != count; ++row) {
        usearch_geospatial_observation_t const& observation = observations[row];
        if (reject_if(std::isnan(observation.latitude_degrees) || std::isnan(observation.longitude_degrees),
                      usearch_status_invalid_argument_k, "An observation needs a latitude and a longitude", outcome))
            return true;
        if (reject_if(observation.end_seconds < observation.begin_seconds, usearch_status_invalid_argument_k,
                      "An observation ends before it begins", outcome))
            return true;
    }
    return false;
}

/** Refuses relations this stand-in cannot answer: unknown ones, and @c visible without terrain. */
static bool reject_relation(usearch_geospatial_relation_t relation, usearch_outcome_t* outcome) {
    if (reject_if(relation == usearch_geospatial_relation_visible_k, usearch_status_unsupported_k,
                  "Lines of sight need a handle with terrain", outcome))
        return true;
    return reject_if(relation != usearch_geospatial_relation_near_k && relation != usearch_geospatial_relation_sees_k &&
                         relation != usearch_geospatial_relation_seen_k,
                     usearch_status_invalid_argument_k, "The relation is not one this ABI names", outcome);
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
static double matched(usearch_geospatial_observation_t const& query, placed_t const& query_placed,
                      usearch_geospatial_observation_t const& entry, usearch_geospatial_relation_t relation,
                      usearch_geospatial_tolerance_t const& tolerance) {
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
    return chord;
}

/** One query's whole scan: every admitted entry that matches, nearest first. */
static void rank_matches(geospatial_index_t const& index, usearch_geospatial_observation_t const& query,
                         usearch_geospatial_relation_t relation, usearch_geospatial_tolerance_t const& tolerance,
                         std::size_t rows, usearch_predicate_t predicate, std::vector<match_t>& ranked) {
    placed_t const query_placed = placed(query);
    us::stub::rank<geospatial_payload_t>(
        std::span<geospatial_entry_t const>(index.entries).first(rows), predicate,
        [&](geospatial_payload_t const& entry) { return matched(query, query_placed, entry, relation, tolerance); },
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

static std::optional<std::size_t> serialized_bytes(std::size_t rows) {
    return blob_bytes(sizeof(geospatial_checkpoint_prefix_t), rows, record_bytes_k);
}

static void write_into(geospatial_index_t const& index, std::size_t rows, bytes_span_t target) {
    write_pod(target, geospatial_checkpoint_prefix_t {index.options.connectivity, rows});
    bytes_span_t cursor = target.subspan(sizeof(geospatial_checkpoint_prefix_t));
    for (std::size_t row = 0; row != rows; ++row) {
        write_pod(cursor, index.entries[row].key);
        write_pod(cursor.subspan(sizeof(usearch_key_t)), index.entries[row].payload);
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
    for (std::size_t row = 0; row != prefix.entries; ++row) {
        usearch_key_t const key = read_pod<usearch_key_t>(cursor);
        index.append(key, read_pod<usearch_geospatial_observation_t>(cursor.subspan(sizeof(usearch_key_t))));
        cursor = cursor.subspan(record_bytes_k);
    }
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
    if (reject_terrain(arguments->terrain, outcome)) return;
    usearch_geospatial_index_handle_t* const index = new (std::nothrow) usearch_geospatial_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->connectivity ? arguments->connectivity : us::utilities::connectivity_16_k,
                      arguments->values_per_key};
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
    if (reject_terrain(arguments->terrain, outcome)) return;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_geospatial_index_handle_t* const index = open_checkpoint<usearch_geospatial_index_handle_t>(blob, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, 1, 1);
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
    if (reject_terrain(arguments->terrain, outcome)) return;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Rows are still copied out of the mapping, so only the view's refusals are mirrored, not its zero cost.
    usearch_geospatial_index_handle_t* const index = open_checkpoint<usearch_geospatial_index_handle_t>(served.view(),
                                                                                                        outcome);
    if (!index) return;
    index->served = std::move(served);
    *arguments->index = index;
}

void usearch_geospatial_view_nvme(usearch_geospatial_view_nvme_t* arguments) {
    if (through_executor<usearch_geospatial_view_nvme>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (reject_terrain(arguments->terrain, outcome)) return;
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Serves the drives' checkpoint as the RAM view does, mirroring only the residency's refusals.
    usearch_geospatial_index_handle_t* const index = open_checkpoint<usearch_geospatial_index_handle_t>(served.view(),
                                                                                                        outcome);
    if (!index) return;
    index->served = std::move(served);
    index->residency = usearch_residency_nvme_k;
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
        "serial",
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
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        index.append(key_at(arguments->keys, row), arguments->observations[row]);
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
    if (arguments->changed) *arguments->changed = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    bool cancelled = false;
    for (std::size_t row = 0; row != rows && !cancelled; ++row)
        cancelled = !keep_going(arguments->progress, row + 1, rows);
    if (arguments->stats) arguments->stats->visited_members = rows;
    // An exact scan reaches every entry it holds in one level.
    if (arguments->reachability) *arguments->reachability = {rows, rows, rows ? 1u : 0u};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

void usearch_geospatial_search(usearch_geospatial_search_t* arguments) {
    if (through_executor<usearch_geospatial_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    *arguments->matches = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_relation(arguments->relation, outcome)) return;
    if (reject_observations(arguments->queries, count, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries},
                                                      outcome);
    ranked_landing_t landing {};
    if (!arena || !carve_ranked(*arena, count, arguments->wanted, landing, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    if (stats && arguments->expansion_factor) stats->computed_distances += arguments->expansion_factor * count;
    std::vector<match_t> ranked;
    std::size_t answered = 0;
    for (; answered != count; ++answered) {
        rank_matches(index, arguments->queries[answered], arguments->relation, arguments->tolerance, rows,
                     arguments->predicate, ranked);
        landing.write(answered, ranked);
        if (stats) {
            stats->visited_members += rows;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(arguments->progress, answered + 1, count)) {
            ++answered;
            break;
        }
    }
    if (reject_cancelled(answered, count, outcome)) return;
    *arguments->matches = landing.view();
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
    if (reject_relation(arguments->relation, outcome)) return;
    if (reject_observations(arguments->queries, count, outcome)) return;
    // Containment is measured from a radius, so a zero-metre tolerance names no ball to report -
    // the one place this verb parts with its ranked sibling.
    if (reject_if(!(tolerance.metres > 0), usearch_status_invalid_argument_k,
                  "A containment walk needs a positive radius in metres", outcome))
        return;
    if (reject_if(count && index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view walks no radius yet", outcome))
        return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries},
                                                      outcome);
    std::size_t* const offsets = arena ? carve<std::size_t>(*arena, count + 1, outcome) : nullptr;
    if (!offsets) return;
    std::shared_lock readers(index.mutex);

    std::size_t const rows = index.entries.size();
    std::vector<match_t> ranked, balls;
    offsets[0] = 0;
    for (std::size_t query = 0; query != count; ++query) {
        rank_matches(index, arguments->queries[query], arguments->relation, tolerance, rows, arguments->predicate,
                     ranked);
        balls.insert(balls.end(), ranked.begin(), ranked.end());
        offsets[query + 1] = balls.size();
        if (stats) {
            stats->visited_members += rows;
            stats->computed_distances += rows;
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
    *arguments->key_stats = {index.by_key.count(keyed(arguments->key))};
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
    std::vector<std::size_t> const rows = rows_under(index, arguments->key);
    if (rows.empty()) return;
    usearch_geospatial_observation_t* const observations = carve<usearch_geospatial_observation_t>(*arena, rows.size(),
                                                                                                   outcome);
    if (!observations) return;
    for (std::size_t found = 0; found != rows.size(); ++found) observations[found] = index.entries[rows[found]].payload;
    *arguments->observations = observations;
    if (arguments->written) *arguments->written = rows.size();
}

void usearch_geospatial_absorb(usearch_geospatial_absorb_t* arguments) {
    if (through_executor<usearch_geospatial_absorb>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_geospatial_index_t* const donors = arguments->donors;
    bool accommodated = true;
    std::unique_lock writers(index.mutex);
    // Each donor lands whole and is consumed at once, its slot nulled, so a failure leaves it and
    // every later one live handles the caller still owns and must free.
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
    std::size_t const rows = index.entries.size();
    bool cancelled = false;
    for (std::size_t row = 0; row != rows && !cancelled; ++row)
        cancelled = !keep_going(arguments->progress, row + 1, rows);
    if (arguments->stats) arguments->stats->visited_members = rows;
    if (arguments->reachability) *arguments->reachability = {rows, rows, rows ? 1u : 0u};
    reject_if(cancelled, usearch_status_cancelled_k, "The caller's handler stopped the sweep", outcome);
}

/** The stand-in's refinement: the donors it still lands, and how far its sweep got. It closes
 *  nothing, so insertions and saves meanwhile proceed. */
struct usearch_geospatial_refinement_handle_t {

    /** The handle the refinement was opened on, only ever compared with the one each call
     *  passes. */
    usearch_geospatial_index_handle_t const* opened_on;

    /** The caller's donor slots, each nulled once landed. */
    usearch_geospatial_index_t* donors;
    std::size_t count;

    /** Donors landed so far, and entries the sweep visited so far. */
    std::size_t landed;
    std::size_t swept;
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
                      "An index cannot absorb itself", outcome))
            return;
    }
    usearch_geospatial_refinement_t const refinement = new (std::nothrow)
        usearch_geospatial_refinement_handle_t {&index, arguments->donors, arguments->count, 0, 0};
    if (reject_if(!refinement, usearch_status_out_of_memory_k, "The refinement could not be allocated", outcome))
        return;
    *arguments->refinement = refinement;
}

void usearch_geospatial_refinement_next(usearch_geospatial_refinement_next_t* arguments) {
    if (through_executor<usearch_geospatial_refinement_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->visited) *arguments->visited = 0;
    if (arguments->changed) *arguments->changed = 0;
    usearch_geospatial_refinement_handle_t& refinement = *arguments->refinement;
    if (reject_if(arguments->index != refinement.opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::unique_lock writers(index.mutex);
    for (; refinement.landed != refinement.count; ++refinement.landed) {
        usearch_geospatial_index_handle_t& donor = *refinement.donors[refinement.landed];
        if (reject_if(!index.accommodate(index.entries.size() + donor.entries.size()), usearch_status_out_of_memory_k,
                      "The donor's entries could not be accommodated", outcome))
            return;
        for (geospatial_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        delete std::exchange(refinement.donors[refinement.landed], nullptr);
    }
    // Each call continues the sweep where the last one stopped.
    std::size_t const rows = index.entries.size();
    bool cancelled = false;
    while (refinement.swept < rows && !cancelled)
        cancelled = !keep_going(arguments->progress, ++refinement.swept, rows);
    // The stand-in re-selects nothing, so its sweep visits every entry and changes none.
    if (arguments->visited) *arguments->visited = refinement.swept;
    if (arguments->stats) arguments->stats->visited_members = refinement.swept;
    reject_if(cancelled && refinement.swept != rows, usearch_status_cancelled_k,
              "The caller's handler stopped the sweep", outcome);
}

void usearch_geospatial_refinement_finish(usearch_geospatial_refinement_finish_t* arguments) {
    if (through_executor<usearch_geospatial_refinement_finish>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    reset(arguments->stats);
    if (arguments->changed) *arguments->changed = 0;
    std::unique_ptr<usearch_geospatial_refinement_handle_t> const refinement {arguments->refinement};
    if (reject_if(arguments->index != refinement->opened_on, usearch_status_invalid_argument_k,
                  "The refinement was opened on another index", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    if (arguments->stats) arguments->stats->visited_members = rows;
    if (arguments->reachability) *arguments->reachability = {rows, rows, rows ? 1u : 0u};
}

void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::vector<std::size_t> linked;
    if (arguments->start) linked = linked_rows(index, *arguments->start, chord_between);
    std::size_t const limit = arguments->start ? linked.size() : index.entries.size();
    usearch_geospatial_cursor_handle_t* const cursor = new (std::nothrow)
        usearch_geospatial_cursor_handle_t {&index, 0, limit, {}, std::move(linked)};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    *arguments->cursor = cursor;
}

void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments) {
    if (through_executor<usearch_geospatial_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->written = 0;
    *arguments->keys = {};
    usearch_geospatial_observation_t const** const observations_output = arguments->observations;
    if (observations_output) *observations_output = nullptr;
    usearch_geospatial_cursor_handle_t& cursor = *arguments->cursor;
    if (reject_if(arguments->index != cursor.opened_on, usearch_status_invalid_argument_k,
                  "The cursor was opened on another index", outcome))
        return;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    if (reject_if(cursor.pulling.test_and_set(std::memory_order_acquire), usearch_status_contended_k,
                  "Another pull is still running on the cursor", outcome))
        return;
    std::shared_lock readers(index.mutex);
    std::size_t const written = std::min(arguments->wanted, cursor.limit - cursor.position);
    usearch_key_t* const keys = carve<usearch_key_t>(*arena, written, outcome);
    usearch_geospatial_observation_t* const observations =
        keys && observations_output ? carve<usearch_geospatial_observation_t>(*arena, written, outcome) : nullptr;

    // A batch the arena could not hold leaves the cursor where it was, so the next pull retries it.
    if (keys && (observations || !observations_output)) {
        for (std::size_t row = 0; row != written; ++row) {
            std::size_t const position = cursor.position + row;
            geospatial_entry_t const& entry = index.entries[cursor.linked.empty() ? position : cursor.linked[position]];
            keys[row] = entry.key;
            if (observations) observations[row] = entry.payload;
        }
        cursor.position += written;
        *arguments->keys = {keys, sizeof(usearch_key_t)};
        if (observations_output) *observations_output = observations;
        *arguments->written = written;
    }
    cursor.pulling.clear(std::memory_order_release);
}

void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_geospatial_sample(usearch_geospatial_sample_t* arguments) {
    if (through_executor<usearch_geospatial_sample>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->keys = {};
    *arguments->links = {};
    *arguments->written = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const anchors_count = arguments->anchors_count;
    if (anchors_count && reject_observations(arguments->anchors, anchors_count, outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->anchors},
                                                      outcome);
    if (!arena) return;

    std::shared_lock readers(index.mutex);
    land_sample(*arguments, std::span<geospatial_entry_t const>(index.entries),
                std::span<usearch_geospatial_observation_t const>(arguments->anchors, anchors_count), chord_between,
                *arena, outcome);
}

void usearch_geospatial_save(usearch_geospatial_save_t* arguments) {
    if (through_executor<usearch_geospatial_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_if(index.residency == usearch_residency_nvme_k, usearch_status_unsupported_k,
                  "An NVMe view already is its checkpoint", outcome))
        return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(rows);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return;
    if (!keep_going(arguments->progress, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(arguments->path, blob, outcome)) return;
    keep_going(arguments->progress, 1, 1);
}

void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments) {
    if (through_executor<usearch_geospatial_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    geospatial_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    *arguments->metadata = {prefix.connectivity};
}

} // extern "C"
