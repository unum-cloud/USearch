/**
 *  @file src_stub/geospatial.cpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief A brute-force stand-in for the @c usearch_geospatial_* C ABI, scoring every stored point
 *      set against every query so a binding can be exercised without the engine.
 *
 *  Two things here are worth a binding's attention more than any kernel. A layout's enumerator
 *  is not its width - @c surface_k carries two axes, @c timed_k three, @c elevated_timed_k four
 *  - so a binding that reached for the enumerator when sizing a point fails on its first
 *  insertion. And a query is an object with its own lifetime, emplaced once and reused across
 *  searches of differing mode and axes, which is a handle to leak or to pass to
 *  @c usearch_geospatial_emplaced_free twice.
 *
 *  @section geospatial_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  Positions convert onto a sphere of one mean radius rather than the WGS-84 ellipsoid, so
 *  distances are close but never survey-grade. Coordinates are stored as @c float, because that
 *  is what a cursor borrows them back as, so a @c get after an @c add returns single-precision
 *  values. A zeroed characteristic speed derives a fixed constant instead of the corpus median.
 *  Position deviations are ignored, and overlap scoring is refused rather than approximated.
 *  Refinement keeps no edges to re-select, so its @c changed is a constant zero rather than a
 *  settling signal, and a caller must not read it as one. A view copies its checkpoint rather than
 *  mapping it, and every VRAM lifecycle verb refuses. Traversal statistics are plausible fiction:
 *  shaped correctly, invented numerically. A checkpoint is a binary blob of this file's own memory
 *  image, readable by nothing else.
 */
#include <cmath> // `std::sqrt`, `std::sin`, `std::cos`

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <limits>    // `std::numeric_limits`
#include <array>     // `std::array`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
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

/** The mean radius the sphere stands in for the ellipsoid with, in metres. */
constexpr double earth_radius_k = 6371000.0;

/** What a zeroed characteristic speed derives to, in metres per second. */
constexpr double derived_speed_k = 13.0;

/** The converted form every mode scores over: Earth-fixed metres, plus scaled time. */
using geospatial_position_t = std::array<double, 4>;

using geospatial_payload_t = std::vector<float>;
using geospatial_entry_t = entry<geospatial_payload_t>;

static_assert(std::is_nothrow_move_constructible_v<geospatial_entry_t>, "reallocation must move rows, never copy them");

/** What a handle froze at construction, or adopted from a checkpoint. */
struct geospatial_index_options_t {

    /** Which axes each stored position carries. */
    usearch_geospatial_layout_t layout;

    /** What the ordering of points within one set means. */
    usearch_geospatial_structure_t structure;

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

    /** The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag pulling;
};

/** A query converted once and reused: it keeps the caller's coordinates, since the axes
 *  a search selects are not known until the search itself. */
struct geospatial_emplaced_t {

    /** The index whose layout and speed govern conversion; it must outlive this. */
    geospatial_index_t* index;

    /** The query's positions, as many as its point count times the layout's width. */
    std::vector<double> points;
};

/** How many coordinates one position carries. Deliberately a switch rather than a cast
 *  of the enumerator, whose values are not the widths they name. */
static std::size_t layout_width(usearch_geospatial_layout_t layout) {
    switch (layout) {
    case usearch_geospatial_layout_surface_k: return 2;
    case usearch_geospatial_layout_elevated_k: return 3;
    case usearch_geospatial_layout_timed_k: return 3;
    case usearch_geospatial_layout_elevated_timed_k: return 4;
    default: return 0;
    }
}

static bool layout_carries_altitude(usearch_geospatial_layout_t layout) {
    return layout == usearch_geospatial_layout_elevated_k || layout == usearch_geospatial_layout_elevated_timed_k;
}

static bool layout_carries_time(usearch_geospatial_layout_t layout) {
    return layout == usearch_geospatial_layout_timed_k || layout == usearch_geospatial_layout_elevated_timed_k;
}

/** Whether the layout is frozen; a zeroed one makes every derived extent zero. */
static bool reject_unfrozen(geospatial_index_t const& index, usearch_outcome_t* outcome) {
    return reject_if(!layout_width(index.options.layout), usearch_status_uninitialized_k, "The layout was never frozen",
                     outcome);
}

/** Whether the mode is one this build scores at all. */
static bool known_mode(usearch_geospatial_mode_t mode) {
    return mode == usearch_geospatial_mode_approach_k || mode == usearch_geospatial_mode_shape_k ||
           mode == usearch_geospatial_mode_coverage_k || mode == usearch_geospatial_mode_route_k;
}

/** Route scoring reads the ordering of a set, which only a path structure carries. */
static bool reject_mode(geospatial_index_t const& index, usearch_geospatial_mode_t mode, usearch_outcome_t* outcome) {
    if (reject_if(!known_mode(mode), usearch_status_unknown_kind_k, "The scoring mode is not one this build knows",
                  outcome))
        return true;
    bool const needs_paths = mode == usearch_geospatial_mode_route_k &&
                             index.options.structure != usearch_geospatial_structure_path_k;
    return reject_if(needs_paths, usearch_status_config_mismatch_k, "Route scoring needs an index of ordered paths",
                     outcome);
}

/** The exchange rate between the axes, which a tolerance names as a radius over a window. */
static double speed_of(usearch_geospatial_tolerance_t const* tolerance) {
    if (!tolerance || tolerance->metres <= 0 || tolerance->seconds <= 0) return derived_speed_k;
    return tolerance->metres / tolerance->seconds;
}

/** Whether the query was emplaced against this very index. The conversion a query holds
 *  is only meaningful under the layout it was built for, so crossing handles is rejected
 *  rather than scored. */
static bool reject_foreign(geospatial_index_t const& index, geospatial_emplaced_t const* query,
                           usearch_outcome_t* outcome) {
    if (reject_if(!query, usearch_status_invalid_argument_k, "The query is null", outcome)) return true;
    return reject_if(query->index != &index, usearch_status_config_mismatch_k,
                     "The query was emplaced against another index", outcome);
}

#pragma endregion Storage

#pragma region Scoring

/** Converts one geodetic position into Earth-fixed metres under the selected axes, with
 *  the unselected components zeroed so they cancel on both sides of a comparison. */
static geospatial_position_t converted(std::span<double const> point, usearch_geospatial_layout_t layout,
                                       usearch_geospatial_axes_t axes, double speed) {
    double const latitude = point[0] * M_PI / 180.0, longitude = point[1] * M_PI / 180.0;
    bool const with_altitude = layout_carries_altitude(layout) && axes != usearch_geospatial_axes_horizontal_k;
    bool const with_time = layout_carries_time(layout) && axes == usearch_geospatial_axes_full_k;
    double const altitude = with_altitude ? point[2] : 0.0;
    double const seconds = with_time ? point[layout_width(layout) - 1] : 0.0;

    double const radius = earth_radius_k + altitude;
    geospatial_position_t converted_position {};
    converted_position[0] = radius * std::cos(latitude) * std::cos(longitude);
    converted_position[1] = radius * std::cos(latitude) * std::sin(longitude);
    converted_position[2] = radius * std::sin(latitude);
    converted_position[3] = seconds * speed;
    return converted_position;
}

static double position_distance(geospatial_position_t const& first, geospatial_position_t const& second) {
    double squared = 0;
    for (std::size_t axis = 0; axis != first.size(); ++axis) {
        double const delta = first[axis] - second[axis];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

/** The converted form of a whole point set, which every mode below scores over. */
static void convert_set(std::span<double const> points, usearch_geospatial_layout_t layout,
                        usearch_geospatial_axes_t axes, double speed, std::vector<geospatial_position_t>& positions) {
    std::size_t const width = layout_width(layout);
    positions.clear();
    for (std::size_t point = 0; point * width != points.size(); ++point)
        positions.push_back(converted(points.subspan(point * width, width), layout, axes, speed));
}

static void convert_stored(std::span<float const> stored, usearch_geospatial_layout_t layout,
                           usearch_geospatial_axes_t axes, double speed, std::vector<double>& widened,
                           std::vector<geospatial_position_t>& positions) {
    widened.assign(stored.begin(), stored.end());
    convert_set(widened, layout, axes, speed, positions);
}

/** The directed nearest-approach of every position in @p from to the whole of @p onto. */
static void directed(std::span<geospatial_position_t const> from, std::span<geospatial_position_t const> onto,
                     double& nearest, double& worst) {
    nearest = 0, worst = 0;
    if (from.empty() || onto.empty()) return;
    nearest = std::numeric_limits<double>::infinity();
    for (geospatial_position_t const& position : from) {
        double closest = std::numeric_limits<double>::infinity();
        for (geospatial_position_t const& partner : onto)
            closest = std::min(closest, position_distance(position, partner));
        nearest = std::min(nearest, closest);
        worst = std::max(worst, closest);
    }
}

/** Scores one query set against one stored set under the selected mode. Route scoring
 *  pairs positions in order, which is what an ordered structure buys. */
static double distance(std::span<geospatial_position_t const> query, std::span<geospatial_position_t const> stored,
                       usearch_geospatial_mode_t mode) {
    if (mode == usearch_geospatial_mode_route_k) {
        std::size_t const paired = std::min(query.size(), stored.size());
        double worst = 0;
        for (std::size_t step = 0; step != paired; ++step)
            worst = std::max(worst, position_distance(query[step], stored[step]));
        return worst;
    }

    double forward_nearest = 0, forward_worst = 0, reverse_nearest = 0, reverse_worst = 0;
    directed(query, stored, forward_nearest, forward_worst);
    directed(stored, query, reverse_nearest, reverse_worst);
    switch (mode) {
    case usearch_geospatial_mode_approach_k: return std::min(forward_nearest, reverse_nearest);
    case usearch_geospatial_mode_coverage_k: return forward_worst;
    default: return std::max(forward_worst, reverse_worst); // shape, the only kind reject_mode leaves
    }
}

/** Reusable buffers, so scoring a whole corpus allocates once rather than per pair. */
struct geospatial_search_scratch_t {
    std::vector<geospatial_position_t> query_positions;
    std::vector<geospatial_position_t> entry_positions;
    std::vector<double> widened;
};

/** One query's whole scan. The query converts once, then every entry scores against it. */
static void rank(geospatial_index_t const& index, geospatial_emplaced_t const& query, usearch_geospatial_mode_t mode,
                 usearch_geospatial_axes_t axes, usearch_geospatial_tolerance_t const* tolerance, std::size_t rows,
                 usearch_predicate_t predicate, geospatial_search_scratch_t& scratch, std::vector<match_t>& ranked) {
    double const speed = speed_of(tolerance);
    convert_set(query.points, index.options.layout, axes, speed, scratch.query_positions);
    us::stub::rank<geospatial_payload_t>(
        std::span<geospatial_entry_t const>(index.entries).first(rows), predicate,
        [&](geospatial_payload_t const& payload) {
            convert_stored(payload, index.options.layout, axes, speed, scratch.widened, scratch.entry_positions);
            return distance(scratch.query_positions, scratch.entry_positions, mode);
        },
        ranked);
}

/**
 *  @brief The body of a search, filtered when its @c predicate is set.
 *  @return Queries answered, short of @c count exactly when the handler cancelled.
 */
template <typename arguments_>
static std::size_t search_queries(geospatial_index_t& index, arguments_ const& arguments,
                                  ranked_landing_t const& landing) {
    std::size_t const rows = index.entries.size();
    std::size_t const count = arguments.count;
    usearch_geospatial_traversal_stats_t* const stats = arguments.stats;
    if (stats && arguments.expansion_factor) stats->computed_distances += arguments.expansion_factor * count;

    geospatial_search_scratch_t scratch;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != count; ++query) {
        rank(index, *arguments.queries[query], arguments.mode, arguments.axes, &arguments.tolerance, rows,
             arguments.predicate, scratch, ranked);
        landing.write(query, ranked);
        if (stats) {
            stats->visited_members += rows;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(arguments.progress, arguments.progress_state, query + 1, count)) return query + 1;
    }
    return count;
}

#pragma endregion Scoring

#pragma region Serialization

/** What @c load needs before it can read an entry. The blob is this file's own memory
 *  image rather than a format: no magic, no version, and nothing else reads it back. */
struct geospatial_checkpoint_prefix_t {

    /** The connections-per-node bound the handle was built with. */
    std::size_t connectivity;

    /** How many entries follow the prefix. */
    std::size_t entries;

    /** Bytes of entry payload that follow, keys and point counts included. */
    std::size_t payload_bytes;

    /** Which axes each stored position carries. */
    usearch_geospatial_layout_t layout;

    /** What the ordering of points within one set means. */
    usearch_geospatial_structure_t structure;
};

/** The per-entry overhead of the blob: the key, then the point count that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(us::utilities::u64_t);

static std::size_t payload_extent(geospatial_index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row)
        extent += record_overhead_k + index.entries[row].payload.size() * sizeof(float);
    return extent;
}

static std::optional<std::size_t> serialized_bytes(geospatial_index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(geospatial_checkpoint_prefix_t), payload_extent(index, rows), 1);
}

static void write_into(geospatial_index_t const& index, std::size_t rows, bytes_span_t target) {
    geospatial_checkpoint_prefix_t prefix {};
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.layout = index.options.layout;
    prefix.structure = index.options.structure;
    write_pod(target, prefix);

    std::size_t const width = layout_width(index.options.layout);
    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        geospatial_entry_t const& entry = index.entries[row];
        std::size_t const points = width ? entry.payload.size() / width : 0;
        std::size_t const payload_bytes = entry.payload.size() * sizeof(float);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, points);
        cursor = cursor.subspan(sizeof(points));
        [[maybe_unused]] status_t const copied = copy_bytes(cursor.first(payload_bytes),
                                                            bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** Reads the prefix and checks the blob is long enough for what it claims. */
static bool parse_prefix(geospatial_checkpoint_prefix_t& prefix, bytes_view_t blob, usearch_outcome_t* outcome) {
    if (reject_if(blob.size() < sizeof(geospatial_checkpoint_prefix_t), usearch_status_malformed_k,
                  "The checkpoint ends early", outcome))
        return false;
    prefix = read_pod<geospatial_checkpoint_prefix_t>(blob);
    return fits_blob(blob, sizeof(geospatial_checkpoint_prefix_t), prefix.payload_bytes, 1, outcome);
}

/** Adopts a checkpoint's configuration. The bound on entries per key is the loader's to choose,
 *  since no checkpoint records it. */
static void adopt(geospatial_index_options_t& options, geospatial_checkpoint_prefix_t const& prefix) {
    options.layout = prefix.layout;
    options.structure = prefix.structure;
    options.connectivity = prefix.connectivity;
}

/** Fills a fresh @p index from a checkpoint, which no other thread can see yet. */
static bool read_from(geospatial_index_t& index, bytes_view_t blob, usearch_outcome_t* outcome) {
    geospatial_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return false;
    std::size_t const entries = prefix.entries;
    index.reset();
    if (!index.accommodate(entries)) {
        propagate_error(usearch_status_out_of_memory_k, outcome, "The checkpoint's entries could not be allocated");
        return false;
    }
    adopt(index.options, prefix);
    std::size_t const width = layout_width(index.options.layout);

    bytes_view_t cursor = blob.subspan(sizeof(geospatial_checkpoint_prefix_t));
    geospatial_payload_t payload;
    for (std::size_t entry_index = 0; entry_index != entries; ++entry_index) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        usearch_key_t key;
        std::size_t points = 0;
        key = read_pod<usearch_key_t>(cursor);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        points = read_pod<std::size_t>(cursor);
        cursor = cursor.subspan(sizeof(points));
        std::size_t const scalars = points * width;
        if (cursor.size() < scalars * sizeof(float)) {
            index.reset();
            propagate_error(usearch_status_malformed_k, outcome, "The checkpoint ends inside an entry");
            return false;
        }
        payload.resize(scalars);
        [[maybe_unused]] status_t const copied = copy_bytes(bytes_of(payload.data(), payload.size()),
                                                            cursor.first(scalars * sizeof(float)));
        cursor = cursor.subspan(scalars * sizeof(float));
        index.append(key, payload);
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
struct usearch_geospatial_emplaced_handle_t : geospatial_emplaced_t {};

extern "C" {

void usearch_geospatial_init_ram(usearch_geospatial_init_ram_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    usearch_geospatial_index_handle_t* const index = new (std::nothrow) usearch_geospatial_index_handle_t();
    if (reject_if(!index, usearch_status_out_of_memory_k, "The handle could not be allocated", outcome)) return;
    index->options = {arguments->layout, arguments->structure,
                      arguments->connectivity ? arguments->connectivity : us::utilities::connectivity_16_k,
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
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    usearch_geospatial_index_handle_t* const index = open_checkpoint<usearch_geospatial_index_handle_t>(blob, outcome);
    if (!index) return;
    index->options.values_per_key = arguments->values_per_key;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
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
    mapped_file_t served = map_file(arguments->path, outcome);
    if (!served) return;
    // Rows are still copied out of the mapping, so only the view's refusals are mirrored, not its zero cost.
    usearch_geospatial_index_handle_t* const index = open_checkpoint<usearch_geospatial_index_handle_t>(served.view(),
                                                                                                        outcome);
    if (!index) return;
    index->served = std::move(served);
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
    geospatial_index_options_t const& options = index.options;
    *arguments->description = {
        {options.layout, options.structure, options.connectivity},
        index.entries.size(),
        index.entries.capacity(),
        payload_extent(index, index.entries.size()),
        "serial",
    };
}

void usearch_geospatial_insert(usearch_geospatial_insert_t* arguments) {
    if (through_executor<usearch_geospatial_insert>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_tape_view_t const& points = arguments->points;
    if (reject_read_only(index, outcome)) return;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_keys(arguments->keys, outcome)) return;
    if (reject_tape(points, count, outcome)) return;
    if (reject_if(points.scalar != usearch_scalar_f64_k, usearch_status_unknown_kind_k,
                  "A point tape holds f64 degrees, metres and seconds", outcome))
        return;
    std::size_t const width = layout_width(index.options.layout);

    std::unique_lock writers(index.mutex);
    if (reject_oversubscribed(index, arguments->keys, count, index.options.values_per_key, outcome)) return;
    if (reject_if(!index.accommodate(index.entries.size() + count), usearch_status_out_of_memory_k,
                  "The batch could not be accommodated", outcome))
        return;

    std::size_t const watermark = index.entries.size();
    double const* const coordinates = static_cast<double const*>(points.data);
    geospatial_payload_t payload;
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        payload.assign(coordinates + points.offsets[row] * width, coordinates + points.offsets[row + 1] * width);
        index.append(key_at(arguments->keys, row), payload);
        ++published;
        if (!keep_going(arguments->progress, arguments->progress_state, published, count)) break;
    }
    // This modality's batch is all-or-nothing, a cancellation included, so a stopped tape unwinds
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
    // Cancelling is not an error here, the header having made a stopped sweep merely less improved.
    for (std::size_t row = 0; row != rows; ++row)
        if (!keep_going(arguments->progress, arguments->progress_state, row + 1, rows)) break;
    if (arguments->stats) arguments->stats->visited_members = rows;
}

void usearch_geospatial_emplace(usearch_geospatial_emplace_t* arguments) {
    if (through_executor<usearch_geospatial_emplace>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->emplaced = nullptr;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_if(!arguments->points && arguments->point_count, usearch_status_invalid_argument_k,
                  "The points must name storage", outcome))
        return;
    usearch_geospatial_emplaced_handle_t* const emplaced = new (std::nothrow) usearch_geospatial_emplaced_handle_t();
    if (reject_if(!emplaced, usearch_status_out_of_memory_k, "The query could not be allocated", outcome)) return;
    std::size_t const width = layout_width(index.options.layout);
    emplaced->index = &index;
    emplaced->points.assign(arguments->points, arguments->points + arguments->point_count * width);
    *arguments->emplaced = emplaced;
}

void usearch_geospatial_emplaced_free(usearch_geospatial_emplaced_t handle) { delete handle; }

void usearch_geospatial_search(usearch_geospatial_search_t* arguments) {
    if (through_executor<usearch_geospatial_search>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    reset(arguments->stats);
    *arguments->matches = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_mode(index, arguments->mode, outcome)) return;
    if (reject_if(!arguments->queries && count, usearch_status_invalid_argument_k, "The batch must name storage",
                  outcome))
        return;
    for (std::size_t query = 0; query != count; ++query)
        if (reject_foreign(index, arguments->queries[query], outcome)) return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries},
                                                      outcome);
    ranked_landing_t landing {};
    if (!arena || !carve_ranked(*arena, count, arguments->wanted, landing, outcome)) return;

    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_queries(index, *arguments, landing);
    if (reject_cancelled(answered, count, outcome)) return;
    *arguments->matches = landing.view();
}

void usearch_geospatial_within(usearch_geospatial_within_t* arguments) {
    if (through_executor<usearch_geospatial_within>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    usearch_geospatial_traversal_stats_t* const stats = arguments->stats;
    reset(stats);
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::size_t const count = arguments->count;
    usearch_geospatial_tolerance_t const& tolerance = arguments->tolerance;
    if (reject_unfrozen(index, outcome)) return;
    if (reject_mode(index, arguments->mode, outcome)) return;
    if (reject_if(!arguments->queries && count, usearch_status_invalid_argument_k, "The batch must name storage",
                  outcome))
        return;
    for (std::size_t query = 0; query != count; ++query)
        if (reject_foreign(index, arguments->queries[query], outcome)) return;
    // Containment is measured from a radius, so a zero-metre tolerance names no ball to report -
    // the one place this verb parts with its ranked sibling, which derives it.
    if (reject_if(!(tolerance.metres > 0), usearch_status_invalid_argument_k,
                  "A containment walk needs a positive radius in metres", outcome))
        return;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {arguments->queries},
                                                      outcome);
    std::size_t* const offsets = arena ? carve<std::size_t>(*arena, count + 1, outcome) : nullptr;
    if (!offsets) return;
    std::shared_lock readers(index.mutex);

    std::size_t const rows = index.entries.size();
    geospatial_search_scratch_t scratch;
    std::vector<match_t> ranked, balls;
    offsets[0] = 0;
    for (std::size_t query = 0; query != count; ++query) {
        rank(index, *arguments->queries[query], arguments->mode, arguments->axes, &tolerance, rows,
             arguments->predicate, scratch, ranked);
        // Ranked nearest first, so each ball is a prefix, and only the walk learns how long it runs.
        std::size_t inside = 0;
        while (inside != ranked.size() && ranked[inside].distance <= tolerance.metres) ++inside;
        balls.insert(balls.end(), ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(inside));
        offsets[query + 1] = balls.size();
        if (stats) {
            stats->visited_members += rows;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(arguments->progress, arguments->progress_state, query + 1, count))
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
    *arguments->key_stats = {};
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::size_t const width = layout_width(index.options.layout);
    auto const range = index.by_key.equal_range(keyed(arguments->key));
    std::size_t entries = 0, total_points = 0, largest_points = 0;
    for (auto position = range.first; position != range.second; ++position, ++entries) {
        std::size_t const points = width ? index.entries[position->second].payload.size() / width : 0;
        total_points += points;
        largest_points = std::max(largest_points, points);
    }
    *arguments->key_stats = {entries, total_points, largest_points};
}

void usearch_geospatial_get(usearch_geospatial_get_t* arguments) {
    if (through_executor<usearch_geospatial_get>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->points = {};
    if (arguments->written) *arguments->written = 0;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, {}, outcome);
    if (!arena) return;
    std::shared_lock readers(index.mutex);
    std::size_t const width = layout_width(index.options.layout);
    std::vector<std::size_t> const rows = rows_under(index, arguments->key);
    if (rows.empty() || !width) return;
    std::size_t total_scalars = 0;
    for (std::size_t row : rows) total_scalars += index.entries[row].payload.size();
    std::size_t* const offsets = carve<std::size_t>(*arena, rows.size() + 1, outcome);
    double* const coordinates = offsets ? carve<double>(*arena, total_scalars, outcome) : nullptr;
    if (!coordinates) return;
    offsets[0] = 0;
    for (std::size_t found = 0; found != rows.size(); ++found) {
        geospatial_payload_t const& payload = index.entries[rows[found]].payload;
        std::copy(payload.begin(), payload.end(), coordinates + offsets[found] * width);
        offsets[found + 1] = offsets[found] + payload.size() / width;
    }
    *arguments->points = {coordinates, offsets, usearch_scalar_f64_k};
    if (arguments->written) *arguments->written = rows.size();
}

void usearch_geospatial_merge(usearch_geospatial_merge_t* arguments) {
    if (through_executor<usearch_geospatial_merge>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    if (reject_read_only(index, outcome)) return;

    std::size_t const donors_count = arguments->count;
    usearch_geospatial_index_t* const donors = arguments->donors;
    usearch_progress_callback_t const progress = arguments->progress;
    void* const progress_state = arguments->progress_state;
    std::size_t merged = 0;
    bool accommodated = true;
    while (accommodated && merged != donors_count) {
        usearch_geospatial_index_handle_t& donor = *donors[merged];
        std::unique_lock writers(index.mutex);
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (geospatial_entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        ++merged;
        if (!keep_going(progress, progress_state, merged, donors_count)) break;
    }
    // Only an absorbed donor is consumed, and its slot is nulled as it goes, so a stopped merge
    // leaves the rest live handles the caller still owns and must free.
    for (std::size_t position = 0; position != merged; ++position) {
        delete donors[position];
        donors[position] = nullptr;
    }
    if (reject_if(!accommodated, usearch_status_out_of_memory_k, "The donor's entries could not be accommodated",
                  outcome))
        return;
    reject_cancelled(merged, donors_count, outcome);
}

void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments) {
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    usearch_geospatial_cursor_handle_t* const cursor = new (std::nothrow)
        usearch_geospatial_cursor_handle_t {&index, 0, index.entries.size(), {}};
    if (reject_if(!cursor, usearch_status_out_of_memory_k, "The cursor could not be allocated", outcome)) return;
    *arguments->cursor = cursor;
}

void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments) {
    if (through_executor<usearch_geospatial_cursor_next>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->written = 0;
    *arguments->keys = {};
    usearch_tape_view_t* const points = arguments->points;
    if (points) *points = {};
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
    std::size_t const width = layout_width(index.options.layout);
    std::size_t const written = std::min(arguments->wanted, cursor.limit - cursor.position);

    // Entries hold the geodetic degrees inserted; the header hands back Earth-fixed centimetres.
    std::size_t points_total = 0;
    for (std::size_t row = 0; row != written; ++row)
        points_total += width ? index.entries[cursor.position + row].payload.size() / width : 0;
    usearch_key_t* const keys = carve<usearch_key_t>(*arena, written, outcome);
    std::size_t* const offsets = keys && points ? carve<std::size_t>(*arena, written + 1, outcome) : nullptr;
    us::utilities::i32_t* const centimetres = offsets ? carve<us::utilities::i32_t>(*arena, points_total * 3, outcome)
                                                      : nullptr;

    // A batch the arena could not hold leaves the cursor where it was, so the next pull retries it.
    if (keys && (centimetres || !points)) {
        std::vector<double> widened;
        std::size_t filled = 0;
        if (points) offsets[0] = 0;
        for (std::size_t row = 0; row != written; ++row) {
            geospatial_entry_t const& entry = index.entries[cursor.position + row];
            keys[row] = entry.key;
            if (!points) continue;
            std::size_t const points = width ? entry.payload.size() / width : 0;
            widened.assign(entry.payload.begin(), entry.payload.end());
            for (std::size_t point = 0; point != points; ++point) {
                geospatial_position_t const fixed = converted(
                    std::span<double const>(widened).subspan(point * width, width), index.options.layout,
                    usearch_geospatial_axes_full_k, 0.0);
                for (std::size_t axis = 0; axis != 3; ++axis)
                    centimetres[(filled + point) * 3 + axis] = static_cast<us::utilities::i32_t>(fixed[axis] * 100.0);
            }
            filled += points;
            offsets[row + 1] = filled;
        }
        cursor.position += written;
        *arguments->keys = {keys, sizeof(usearch_key_t)};
        if (points) *points = {centimetres, offsets, usearch_scalar_i32_k};
        *arguments->written = written;
    }
    cursor.pulling.clear(std::memory_order_release);
}

void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t handle) {
    wait_for_calls_using(handle);
    delete handle;
}

void usearch_geospatial_save(usearch_geospatial_save_t* arguments) {
    if (through_executor<usearch_geospatial_save>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    usearch_geospatial_index_handle_t& index = *arguments->index;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (reject_if(!length, usearch_status_overflow_k, "The checkpoint would not fit an addressable buffer", outcome))
        return;
    if (!keep_going(arguments->progress, arguments->progress_state, 0, 1))
        return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(arguments->path, blob, outcome)) return;
    keep_going(arguments->progress, arguments->progress_state, 1, 1);
}

void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments) {
    if (through_executor<usearch_geospatial_metadata_read>(arguments)) return;
    deferred_outcome_t outcome(arguments->outcome);
    *arguments->metadata = {};
    std::vector<std::byte> blob;
    if (!read_file(arguments->path, blob, outcome)) return;
    geospatial_checkpoint_prefix_t prefix {};
    if (!parse_prefix(prefix, blob, outcome)) return;
    geospatial_index_options_t options {};
    adopt(options, prefix);
    *arguments->metadata = {options.layout, options.structure, options.connectivity};
}

} // extern "C"
