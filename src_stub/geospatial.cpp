/**
 *  @brief A brute-force stand-in for the @c usearch_geospatial_* C ABI, scoring every stored
 *      point set against every query so a binding can be exercised without the engine.
 *
 *  @file src_stub/geospatial.cpp
 *
 *  Two things here are worth a binding's attention more than any kernel. A layout's enumerator
 *  is not its width - @c surface_k carries two axes, @c timed_k three, @c elevated_timed_k four
 *  - so a binding that reached for the enumerator when sizing a point fails on its first
 *  insertion. And a query is an object with its own lifetime, emplaced once and reused across
 *  searches of differing mode and axes, which is a handle to leak or to free twice.
 *
 *  @section geospatial_stub_blind_spots What It Cannot Tell You
 *
 *  Search is an exact scan, so recall is trivially perfect and ties fall in insertion order.
 *  Positions convert onto a sphere of one mean radius rather than the WGS-84 ellipsoid, so
 *  distances are close but never survey-grade. Coordinates are stored as @c float, because that
 *  is what a cursor borrows them back as, so a @c get after an @c add returns single-precision
 *  values. A zeroed characteristic speed derives a fixed constant instead of the corpus median.
 *  Traversal statistics are plausible fiction: shaped correctly, invented numerically. A
 *  checkpoint is a binary blob of this file's own memory image, readable by nothing else.
 */
#include <cmath> // `std::sqrt`, `std::sin`, `std::cos`

#include <algorithm> // `std::stable_sort`, `std::min`, `std::max`
#include <array>     // `std::array`
#include <optional>  // `std::optional`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include "usearch/types.h"
#include "usearch/geospatial_index.h"

#include "lib.hpp"

namespace us = unum::usearch;
using namespace unum::usearch;
using namespace unum::usearch::stub;

namespace {

#pragma region Storage

/** @brief The mean radius the sphere stands in for the ellipsoid with, in metres. */
constexpr double earth_radius_k = 6371000.0;

/** @brief What a zeroed characteristic speed derives to, in metres per second. */
constexpr double derived_speed_k = 13.0;

/** @brief The converted form every mode scores over: Earth-fixed metres, plus scaled time. */
using position_t = std::array<double, 4>;

using payload_t = std::vector<float>;
using entry_t = entry<payload_t>;

static_assert(std::is_nothrow_move_constructible_v<entry_t>, "reallocation must move rows, never copy them");

using index_t = handle<payload_t, usearch_geospatial_options_t>;

struct cursor_t {
    /** @brief The index being walked; it must outlive the cursor. */
    index_t* index;
    /** @brief The next entry to hand out. */
    std::size_t position;
    /** @brief The watermark captured at creation, which later insertions cannot raise. */
    std::size_t limit;
    /** @brief Earth-fixed centimetres of the batch just handed out, borrowed until the next call. */
    std::vector<us::i32_t> centimetres;
};

/**
 *  @brief A query converted once and reused: it keeps the caller's coordinates, since the axes
 *      a search selects are not known until the search itself.
 */
struct emplaced_t {
    /** @brief The index whose layout and speed govern conversion; it must outlive this. */
    index_t* index;
    /** @brief The query's positions, as many as its point count times the layout's width. */
    std::vector<double> points;
};

/**
 *  @brief How many coordinates one position carries. Deliberately a switch rather than a cast
 *      of the enumerator, whose values are not the widths they name.
 */
std::size_t layout_width(usearch_geospatial_layout_t layout) {
    switch (layout) {
    case usearch_geospatial_layout_surface_k: return 2;
    case usearch_geospatial_layout_elevated_k: return 3;
    case usearch_geospatial_layout_timed_k: return 3;
    case usearch_geospatial_layout_elevated_timed_k: return 4;
    default: return 0;
    }
}

bool layout_carries_altitude(usearch_geospatial_layout_t layout) {
    return layout == usearch_geospatial_layout_elevated_k || layout == usearch_geospatial_layout_elevated_timed_k;
}

bool layout_carries_time(usearch_geospatial_layout_t layout) {
    return layout == usearch_geospatial_layout_timed_k || layout == usearch_geospatial_layout_elevated_timed_k;
}

/** @brief Whether the layout is frozen; a zeroed one makes every derived extent zero. */
bool reject_unfrozen(index_t const& index, usearch_error_t* error) {
    return reject_if(!layout_width(index.options.layout), usearch_error_uninitialized_k, "The layout was never frozen",
                     error);
}

/** @brief Whether the mode is one this build scores at all. */
bool known_mode(usearch_geospatial_mode_t mode) {
    return mode == usearch_geospatial_mode_approach_k || mode == usearch_geospatial_mode_shape_k ||
           mode == usearch_geospatial_mode_coverage_k || mode == usearch_geospatial_mode_route_k;
}

/** @brief Route scoring reads the ordering of a set, which only a path structure carries. */
bool reject_mode(index_t const& index, usearch_geospatial_mode_t mode, usearch_error_t* error) {
    if (reject_if(!known_mode(mode), usearch_error_unknown_kind_k, "The scoring mode is not one this build knows",
                  error))
        return true;
    bool const needs_paths = mode == usearch_geospatial_mode_route_k &&
                             index.options.structure != usearch_geospatial_structure_path_k;
    return reject_if(needs_paths, usearch_error_config_mismatch_k, "Route scoring needs an index of ordered paths",
                     error);
}

/** @brief The exchange rate between the axes, which a tolerance names as a radius over a window. */
double speed_of(usearch_geospatial_tolerance_t const* tolerance) {
    if (!tolerance || tolerance->metres <= 0 || tolerance->seconds <= 0) return derived_speed_k;
    return tolerance->metres / tolerance->seconds;
}

/**
 *  @brief Whether the query was emplaced against this very index. The conversion a query holds
 *      is only meaningful under the layout it was built for, so crossing handles is rejected
 *      rather than scored.
 */
bool reject_foreign(index_t const& index, usearch_geospatial_emplaced_t query, usearch_error_t* error) {
    if (reject_if(!query, usearch_error_malformed_input_k, "The query is null", error)) return true;
    return reject_if(borrowed<emplaced_t const>(query).index != &index, usearch_error_config_mismatch_k,
                     "The query was emplaced against another index", error);
}

#pragma endregion Storage

#pragma region Scoring

/**
 *  @brief Converts one geodetic position into Earth-fixed metres under the selected axes, with
 *      the unselected components zeroed so they cancel on both sides of a comparison.
 */
position_t converted(std::span<double const> point, usearch_geospatial_layout_t layout, usearch_geospatial_axes_t axes,
                     double speed) {
    double const latitude = point[0] * M_PI / 180.0, longitude = point[1] * M_PI / 180.0;
    bool const with_altitude = layout_carries_altitude(layout) && axes != usearch_geospatial_axes_horizontal_k;
    bool const with_time = layout_carries_time(layout) && axes == usearch_geospatial_axes_full_k;
    double const altitude = with_altitude ? point[2] : 0.0;
    double const seconds = with_time ? point[layout_width(layout) - 1] : 0.0;

    double const radius = earth_radius_k + altitude;
    position_t converted_position {};
    converted_position[0] = radius * std::cos(latitude) * std::cos(longitude);
    converted_position[1] = radius * std::cos(latitude) * std::sin(longitude);
    converted_position[2] = radius * std::sin(latitude);
    converted_position[3] = seconds * speed;
    return converted_position;
}

double position_distance(position_t const& first, position_t const& second) {
    double squared = 0;
    for (std::size_t axis = 0; axis != first.size(); ++axis) {
        double const delta = first[axis] - second[axis];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

/** @brief The converted form of a whole point set, which every mode below scores over. */
void convert_set(std::span<double const> points, usearch_geospatial_layout_t layout, usearch_geospatial_axes_t axes,
                 double speed, std::vector<position_t>& positions) {
    std::size_t const width = layout_width(layout);
    positions.clear();
    for (std::size_t point = 0; point * width != points.size(); ++point)
        positions.push_back(converted(points.subspan(point * width, width), layout, axes, speed));
}

void convert_stored(std::span<float const> stored, usearch_geospatial_layout_t layout, usearch_geospatial_axes_t axes,
                    double speed, std::vector<double>& widened, std::vector<position_t>& positions) {
    widened.assign(stored.begin(), stored.end());
    convert_set(widened, layout, axes, speed, positions);
}

/** @brief The directed nearest-approach of every position in @p from to the whole of @p onto. */
void directed(std::span<position_t const> from, std::span<position_t const> onto, double& nearest, double& worst) {
    nearest = INFINITY, worst = 0;
    for (position_t const& position : from) {
        double closest = INFINITY;
        for (position_t const& partner : onto) closest = std::min(closest, position_distance(position, partner));
        if (closest == INFINITY) closest = 0;
        nearest = std::min(nearest, closest);
        worst = std::max(worst, closest);
    }
    if (nearest == INFINITY) nearest = 0;
}

/**
 *  @brief Scores one query set against one stored set under the selected mode. Route scoring
 *      pairs positions in order, which is what an ordered structure buys.
 */
double distance(std::span<position_t const> query, std::span<position_t const> stored, usearch_geospatial_mode_t mode) {
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

/** @brief Reusable buffers, so scoring a whole corpus allocates once rather than per pair. */
struct scratch_t {
    std::vector<position_t> query_positions;
    std::vector<position_t> entry_positions;
    std::vector<double> widened;
};

/** @brief One query's whole scan. The query converts once, then every entry scores against it. */
void rank(index_t const& index, emplaced_t const& query, usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,
          usearch_geospatial_tolerance_t const* tolerance, std::size_t rows, usearch_predicate_t predicate,
          void* predicate_state, scratch_t& scratch, std::vector<match_t>& ranked) {
    double const speed = speed_of(tolerance);
    convert_set(query.points, index.options.layout, axes, speed, scratch.query_positions);
    us::stub::rank<payload_t>(
        std::span<entry_t const>(index.entries).first(rows), admitting(predicate, predicate_state),
        [&](payload_t const& payload) {
            convert_stored(payload, index.options.layout, axes, speed, scratch.widened, scratch.entry_positions);
            return distance(scratch.query_positions, scratch.entry_positions, mode);
        },
        ranked);
}

/**
 *  @brief The shared body of both search forms; the filtered one passes a predicate.
 *  @return Queries answered, short of @p counts.size() exactly when the handler cancelled.
 */
std::size_t search_queries(index_t& index, usearch_geospatial_emplaced_t const* queries,
                           usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,
                           usearch_geospatial_tolerance_t const* tolerance, std::size_t wanted,
                           usearch_geospatial_expansion_config_t const* config,
                           usearch_geospatial_traversal_stats_t* stats, usearch_predicate_t predicate,
                           void* predicate_state, usearch_progress_callback_t progress, void* progress_state,
                           usearch_key_t* keys, std::size_t keys_stride, usearch_distance_t* distances,
                           std::size_t distances_stride, std::span<std::size_t> counts) {
    std::size_t const rows = index.entries.size();
    if (stats && config && config->expansion_factor)
        stats->computed_distances += config->expansion_factor * counts.size();

    std::fill(counts.begin(), counts.end(), 0);
    scratch_t scratch;
    std::vector<match_t> ranked;
    for (std::size_t query = 0; query != counts.size(); ++query) {
        rank(index, borrowed<emplaced_t const>(queries[query]), mode, axes, tolerance, rows, predicate,
             predicate_state, scratch, ranked);
        std::span<usearch_key_t> const keys_row = row_at<usearch_key_t>(keys, query, keys_stride, wanted);
        std::span<usearch_distance_t> const distances_row = row_at<usearch_distance_t>(distances, query,
                                                                                       distances_stride, wanted);
        std::size_t const found = std::min(wanted, ranked.size());
        for (std::size_t match = 0; match != found; ++match)
            keys_row[match] = ranked[match].key, distances_row[match] = ranked[match].distance;
        counts[query] = found;
        if (stats) {
            stats->visited_members += rows;
            stats->computed_distances += ranked.size();
            stats->restarts += 1;
        }
        if (!keep_going(progress, progress_state, query + 1, counts.size())) return query + 1;
    }
    return counts.size();
}

#pragma endregion Scoring

#pragma region Serialization

/**
 *  @brief What @c load needs before it can read an entry. The blob is this file's own memory
 *      image rather than a format: no magic, no version, and nothing else reads it back.
 */
struct prefix_t {
    /** @brief The connections-per-node bound the handle was built with. */
    std::size_t connectivity;
    /** @brief How many entries follow the prefix. */
    std::size_t entries;
    /** @brief Bytes of entry payload that follow, keys and point counts included. */
    std::size_t payload_bytes;
    /** @brief Which axes each stored position carries. */
    usearch_geospatial_layout_t layout;
    /** @brief What the ordering of points within one set means. */
    usearch_geospatial_structure_t structure;
};

/** @brief The per-entry overhead of the blob: the key, then the point count that follows it. */
constexpr std::size_t record_overhead_k = sizeof(usearch_key_t) + sizeof(us::u64_t);

std::size_t payload_extent(index_t const& index, std::size_t rows) {
    std::size_t extent = 0;
    for (std::size_t row = 0; row != rows; ++row)
        extent += record_overhead_k + index.entries[row].payload.size() * sizeof(float);
    return extent;
}

std::optional<std::size_t> serialized_bytes(index_t const& index, std::size_t rows) {
    return blob_bytes(sizeof(prefix_t), payload_extent(index, rows), 1);
}

void write_into(index_t const& index, std::size_t rows, bytes_span_t target) {
    prefix_t prefix {};
    prefix.connectivity = index.options.connectivity;
    prefix.entries = rows;
    prefix.payload_bytes = payload_extent(index, rows);
    prefix.layout = index.options.layout;
    prefix.structure = index.options.structure;
    write_pod(target, prefix);

    std::size_t const width = layout_width(index.options.layout);
    bytes_span_t cursor = target.subspan(sizeof(prefix));
    for (std::size_t row = 0; row != rows; ++row) {
        entry_t const& entry = index.entries[row];
        std::size_t const points = width ? entry.payload.size() / width : 0;
        std::size_t const payload_bytes = entry.payload.size() * sizeof(float);
        write_pod(cursor, entry.key);
        cursor = cursor.subspan(sizeof(usearch_key_t));
        write_pod(cursor, points);
        cursor = cursor.subspan(sizeof(points));
        [[maybe_unused]] status_t const copied =
            copy_bytes(cursor.first(payload_bytes), bytes_of(entry.payload.data(), entry.payload.size()));
        cursor = cursor.subspan(payload_bytes);
    }
}

/** @brief Reads the prefix and checks the blob is long enough for what it claims. */
bool parse_prefix(prefix_t& prefix, bytes_view_t blob, usearch_error_t* error) {
    if (reject_if(blob.size() < sizeof(prefix_t), usearch_error_malformed_input_k, "The checkpoint ends early", error))
        return false;
    prefix = read_pod<prefix_t>(blob);
    return fits_blob(blob, sizeof(prefix_t), prefix.payload_bytes, 1, error);
}

void adopt(usearch_geospatial_options_t& options, prefix_t const& prefix) {
    options.layout = prefix.layout;
    options.structure = prefix.structure;
    options.connectivity = prefix.connectivity;
}

/** @brief Replaces the contents of @p index; the caller holds both locks exclusively. */
bool read_from(index_t& index, bytes_view_t blob, usearch_error_t* error) {
    prefix_t prefix {};
    if (!parse_prefix(prefix, blob, error)) return false;
    std::size_t const entries = prefix.entries;
    index.reset();
    if (!index.accommodate(entries)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The checkpoint's entries could not be allocated");
        return false;
    }
    adopt(index.options, prefix);
    std::size_t const width = layout_width(index.options.layout);

    bytes_view_t cursor = blob.subspan(sizeof(prefix_t));
    payload_t payload;
    for (std::size_t entry_index = 0; entry_index != entries; ++entry_index) {
        if (cursor.size() < record_overhead_k) {
            index.reset();
            propagate_error(usearch_error_malformed_input_k, error, "The checkpoint ends inside an entry");
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
            propagate_error(usearch_error_malformed_input_k, error, "The checkpoint ends inside an entry");
            return false;
        }
        payload.resize(scalars);
        [[maybe_unused]] status_t const copied =
            copy_bytes(bytes_of(payload.data(), payload.size()), cursor.first(scalars * sizeof(float)));
        cursor = cursor.subspan(scalars * sizeof(float));
        index.append(key, payload);
    }
    return true;
}

#pragma endregion Serialization

} // namespace

extern "C" {

usearch_geospatial_index_t usearch_geospatial_init(usearch_geospatial_options_t const* options,
                                                   usearch_error_t* error) {
    propagate_success(error);
    if (!options) {
        propagate_error(usearch_error_malformed_input_k, error, "The options are null");
        return nullptr;
    }
    index_t* index = new (std::nothrow) index_t();
    if (!index) {
        propagate_error(usearch_error_out_of_memory_k, error, "The handle could not be allocated");
        return nullptr;
    }
    index->options = *options;
    if (!index->options.connectivity) index->options.connectivity = us::connectivity_16_k;
    return index;
}

void usearch_geospatial_free(usearch_geospatial_index_t handle) { delete static_cast<index_t*>(handle); }

void usearch_geospatial_reserve(usearch_geospatial_index_t handle, std::size_t entries,
                                [[maybe_unused]] std::size_t points, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    if (!index.accommodate(entries))
        propagate_error(usearch_error_out_of_memory_k, error, "The capacity could not be reserved");
}

void usearch_geospatial_clear(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::unique_lock writers(index.mutex);
    index.reset();
}

std::size_t usearch_geospatial_size(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).entries.size();
}
std::size_t usearch_geospatial_capacity(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return index.entries.capacity();
}
std::size_t usearch_geospatial_connectivity(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    return borrowed<index_t>(handle).options.connectivity;
}
std::size_t usearch_geospatial_memory_usage(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    return payload_extent(index, index.entries.size());
}
char const* usearch_geospatial_capabilities([[maybe_unused]] usearch_geospatial_index_t handle,
                                            usearch_error_t* error) {
    propagate_success(error);
    return "serial";
}

std::size_t usearch_geospatial_add(usearch_geospatial_index_t handle, usearch_executor_t executor,
                                   usearch_key_t const* keys, double const* const* point_sets,
                                   std::size_t const* point_counts,
                                   [[maybe_unused]] double const* const* deviations, std::size_t count,
                                   [[maybe_unused]] usearch_geospatial_expansion_config_t const* config,
                                   usearch_progress_callback_t progress, void* progress_state,
                                   usearch_geospatial_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    if (reject_unfrozen(index, error)) return 0;
    std::size_t const width = layout_width(index.options.layout);

    std::unique_lock writers(index.mutex);
    if (!index.accommodate(index.entries.size() + count)) {
        propagate_error(usearch_error_out_of_memory_k, error, "The batch could not be accommodated");
        return 0;
    }

    payload_t payload;
    std::size_t published = 0;
    for (std::size_t row = 0; row != count; ++row) {
        std::span<double const> const points {point_sets[row], point_counts[row] * width};
        payload.assign(points.begin(), points.end());
        index.append(keys[row], payload);
        ++published;
        if (!keep_going(progress, progress_state, published, count)) break;
    }
    if (stats) {
        stats->visited_members = published;
        stats->computed_distances = published * index.entries.size();
        stats->restarts = 1;
    }
    reject_cancelled(published, count, error);
    return published;
}

std::size_t usearch_geospatial_refine(usearch_geospatial_index_t handle, usearch_executor_t executor,
                                      [[maybe_unused]] usearch_geospatial_expansion_config_t const* config,
                                      usearch_progress_callback_t progress, void* progress_state,
                                      usearch_geospatial_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    // An exact scan keeps no edges, so a sweep re-selects nothing and the graph is always settled.
    // Cancelling is not an error here, the header having made a stopped sweep merely less improved.
    for (std::size_t row = 0; row != rows; ++row)
        if (!keep_going(progress, progress_state, row + 1, rows)) break;
    if (stats) stats->visited_members = rows;
    return 0;
}

usearch_geospatial_emplaced_t usearch_geospatial_emplace(usearch_geospatial_index_t handle, double const* points,
                                                         std::size_t point_count, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_unfrozen(index, error)) return nullptr;
    emplaced_t* emplaced = new (std::nothrow) emplaced_t();
    if (!emplaced) {
        propagate_error(usearch_error_out_of_memory_k, error, "The query could not be allocated");
        return nullptr;
    }
    std::size_t const width = layout_width(index.options.layout);
    emplaced->index = &index;
    emplaced->points.assign(points, points + point_count * width);
    return emplaced;
}

void usearch_geospatial_release(usearch_geospatial_emplaced_t handle, usearch_error_t* error) {
    propagate_success(error);
    delete static_cast<emplaced_t*>(handle);
}

void usearch_geospatial_search(usearch_geospatial_index_t handle, usearch_executor_t executor,
                               usearch_geospatial_emplaced_t const* queries, std::size_t queries_count,
                               usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,
                               usearch_geospatial_tolerance_t const* tolerance, std::size_t wanted,
                               usearch_geospatial_expansion_config_t const* config,
                               usearch_progress_callback_t progress, void* progress_state, usearch_key_t* keys,
                               std::size_t keys_stride, usearch_distance_t* distances, std::size_t distances_stride,
                               std::size_t* counts, usearch_geospatial_traversal_stats_t* stats,
                               usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    if (reject_mode(index, mode, error)) return;
    for (std::size_t query = 0; query != queries_count; ++query)
        if (reject_foreign(index, queries[query], error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_queries(index, queries, mode, axes, tolerance, wanted, config, stats, nullptr,
                                                nullptr, progress, progress_state, keys, keys_stride, distances,
                                                distances_stride, {counts, queries_count});
    reject_cancelled(answered, queries_count, error);
}

void usearch_geospatial_search_if(usearch_geospatial_index_t handle, usearch_executor_t executor,
                                  usearch_geospatial_emplaced_t const* queries, std::size_t queries_count,
                                  usearch_geospatial_mode_t mode, usearch_geospatial_axes_t axes,
                                  usearch_geospatial_tolerance_t const* tolerance, std::size_t wanted,
                                  usearch_geospatial_expansion_config_t const* config, usearch_predicate_t predicate,
                                  void* predicate_state, usearch_progress_callback_t progress, void* progress_state,
                                  usearch_key_t* keys, std::size_t keys_stride, usearch_distance_t* distances,
                                  std::size_t distances_stride, std::size_t* counts,
                                  usearch_geospatial_traversal_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    reset(stats);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (reject_unfrozen(index, error)) return;
    if (reject_mode(index, mode, error)) return;
    for (std::size_t query = 0; query != queries_count; ++query)
        if (reject_foreign(index, queries[query], error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const answered = search_queries(index, queries, mode, axes, tolerance, wanted, config, stats,
                                                predicate, predicate_state, progress, progress_state, keys,
                                                keys_stride, distances, distances_stride, {counts, queries_count});
    reject_cancelled(answered, queries_count, error);
}

std::size_t usearch_geospatial_within(usearch_geospatial_index_t handle, usearch_executor_t executor,
                                      usearch_geospatial_emplaced_t query, usearch_geospatial_mode_t mode,
                                      usearch_geospatial_axes_t axes,
                                      usearch_geospatial_tolerance_t const* tolerance,
                                      usearch_progress_callback_t progress, void* progress_state,
                                      usearch_key_t* keys, usearch_distance_t* distances, std::size_t capacity,
                                      usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return 0;
    if (reject_unfrozen(index, error)) return 0;
    if (reject_mode(index, mode, error)) return 0;
    if (reject_foreign(index, query, error)) return 0;
    std::shared_lock readers(index.mutex);

    scratch_t scratch;
    std::vector<match_t> ranked;
    rank(index, borrowed<emplaced_t const>(query), mode, axes, tolerance, index.entries.size(), nullptr, nullptr,
         scratch, ranked);
    // Zero metres names no radius at all, so every match is inside it.
    double const radius = tolerance && tolerance->metres > 0 ? tolerance->metres : INFINITY;
    // The header has this one saturate at capacity rather than report the true total, unlike
    // every truncating verb of the dense modality.
    std::size_t written = 0;
    for (match_t const& match : ranked) {
        if (written == capacity || match.distance > radius) break;
        if (keys) keys[written] = match.key;
        if (distances) distances[written] = match.distance;
        ++written;
        if (!keep_going(progress, progress_state, written, capacity)) break;
    }
    return written;
}

std::size_t usearch_geospatial_count(usearch_geospatial_index_t handle, usearch_key_t key,
                                     usearch_geospatial_key_stats_t* stats, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::size_t const width = layout_width(index.options.layout);
    std::vector<std::size_t> const rows = rows_under(index, key);
    std::size_t total_points = 0, largest_points = 0;
    for (std::size_t row : rows) {
        std::size_t const points = width ? index.entries[row].payload.size() / width : 0;
        total_points += points;
        largest_points = std::max(largest_points, points);
    }
    if (stats) {
        stats->entries = rows.size();
        stats->total_points = total_points;
        stats->largest_points = largest_points;
    }
    return rows.size();
}

std::size_t usearch_geospatial_get(usearch_geospatial_index_t handle, usearch_key_t key, double* point_sets,
                                   std::size_t wanted, std::size_t stride, std::size_t* point_counts,
                                   usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::size_t const width = layout_width(index.options.layout);
    std::vector<std::size_t> const rows = rows_under(index, key);
    for (std::size_t found = 0; found != rows.size() && found < wanted; ++found) {
        entry_t const& entry = index.entries[rows[found]];
        if (point_sets) {
            std::span<double> const target = row_at<double>(point_sets, found, stride, entry.payload.size());
            for (std::size_t scalar = 0; scalar != entry.payload.size(); ++scalar)
                target[scalar] = entry.payload[scalar];
        }
        if (point_counts) point_counts[found] = width ? entry.payload.size() / width : 0;
    }
    return rows.size();
}

void usearch_geospatial_merge(usearch_geospatial_index_t handle, usearch_executor_t executor,
                              usearch_geospatial_index_t const* donors, std::size_t donors_count,
                              usearch_progress_callback_t progress, void* progress_state, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;

    std::size_t merged = 0;
    bool accommodated = true;
    while (accommodated && merged != donors_count) {
        index_t& donor = borrowed<index_t>(donors[merged]);
        std::unique_lock writers(index.mutex);
        accommodated = index.accommodate(index.entries.size() + donor.entries.size());
        if (!accommodated) break;
        for (entry_t const& entry : donor.entries) index.append(entry.key, entry.payload);
        ++merged;
        if (!keep_going(progress, progress_state, merged, donors_count)) break;
    }
    // Consumed whether merged or not; the ABI forbids the caller freeing them.
    for (std::size_t position = 0; position != donors_count; ++position)
        delete static_cast<index_t*>(donors[position]);
    if (!accommodated)
        return propagate_error(usearch_error_out_of_memory_k, error, "The donor's entries could not be accommodated");
    reject_cancelled(merged, donors_count, error);
}

usearch_geospatial_cursor_t usearch_geospatial_cursor(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    cursor_t* cursor = new (std::nothrow) cursor_t {&index, 0, index.entries.size(), {}};
    if (!cursor) propagate_error(usearch_error_out_of_memory_k, error, "The cursor could not be allocated");
    return cursor;
}

std::size_t usearch_geospatial_cursor_next(usearch_geospatial_cursor_t handle, usearch_key_t* keys,
                                           us::i32_t const** point_sets, std::size_t* point_counts,
                                           std::size_t wanted, usearch_error_t* error) {
    propagate_success(error);
    cursor_t& cursor = borrowed<cursor_t>(handle);
    index_t& index = *cursor.index;
    std::shared_lock readers(index.mutex);
    std::size_t const width = layout_width(index.options.layout);
    std::size_t const written = std::min(wanted, cursor.limit - cursor.position);

    // Entries hold the geodetic degrees inserted; the header hands back Earth-fixed centimetres,
    // so the batch converts into the cursor's own buffer, sized before any view points into it.
    std::size_t points_total = 0;
    for (std::size_t row = 0; row != written; ++row)
        points_total += width ? index.entries[cursor.position + row].payload.size() / width : 0;
    cursor.centimetres.assign(points_total * 3, 0);

    std::span<usearch_key_t> const keys_row {keys, written};
    std::vector<double> widened;
    std::size_t filled = 0;
    for (std::size_t row = 0; row != written; ++row) {
        entry_t const& entry = index.entries[cursor.position + row];
        std::size_t const points = width ? entry.payload.size() / width : 0;
        keys_row[row] = entry.key;
        if (point_sets) point_sets[row] = cursor.centimetres.data() + filled * 3;
        if (point_counts) point_counts[row] = points;
        widened.assign(entry.payload.begin(), entry.payload.end());
        for (std::size_t point = 0; point != points; ++point) {
            position_t const fixed = converted(std::span<double const>(widened).subspan(point * width, width),
                                               index.options.layout, usearch_geospatial_axes_full_k, 0.0);
            for (std::size_t axis = 0; axis != 3; ++axis)
                cursor.centimetres[(filled + point) * 3 + axis] = static_cast<us::i32_t>(fixed[axis] * 100.0);
        }
        filled += points;
    }
    cursor.position += written;
    return written;
}

void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t handle) { delete static_cast<cursor_t*>(handle); }

std::size_t usearch_geospatial_storage_requirements(usearch_geospatial_index_t handle, usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    std::shared_lock readers(index.mutex);
    std::optional<std::size_t> const length = serialized_bytes(index, index.entries.size());
    if (!length) {
        propagate_error(usearch_error_overflow_k, error, "The checkpoint would not fit an addressable buffer");
        return 0;
    }
    return *length;
}

void usearch_geospatial_save(usearch_geospatial_index_t handle, usearch_executor_t executor,
                             usearch_bytes_view_t path, usearch_progress_callback_t progress, void* progress_state,
                             usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    std::shared_lock readers(index.mutex);
    std::size_t const rows = index.entries.size();
    std::optional<std::size_t> const length = serialized_bytes(index, rows);
    if (!length)
        return propagate_error(usearch_error_overflow_k, error, "The checkpoint would not fit an addressable buffer");
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob(*length);
    write_into(index, rows, blob);
    if (!write_file(path, blob, error)) return;
    keep_going(progress, progress_state, 1, 1);
}

void usearch_geospatial_load(usearch_geospatial_index_t handle, usearch_executor_t executor,
                             usearch_bytes_view_t path, usearch_progress_callback_t progress, void* progress_state,
                             usearch_error_t* error) {
    propagate_success(error);
    index_t& index = borrowed<index_t>(handle);
    if (reject_executor(executor, error)) return;
    if (!keep_going(progress, progress_state, 0, 1))
        return propagate_error(usearch_error_cancelled_k, error, "The caller's handler stopped the checkpoint");
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    std::unique_lock writers(index.mutex);
    if (!read_from(index, blob, error)) return;
    keep_going(progress, progress_state, 1, 1);
}

void usearch_geospatial_metadata(usearch_bytes_view_t path, usearch_geospatial_options_t* options, usearch_error_t* error) {
    propagate_success(error);
    std::vector<std::byte> blob;
    if (!read_file(path, blob, error)) return;
    prefix_t prefix {};
    if (parse_prefix(prefix, blob, error)) adopt(*options, prefix);
}

} // extern "C"
