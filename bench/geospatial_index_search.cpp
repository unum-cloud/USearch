/**
 *  @file bench/geospatial_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The geospatial search benchmark: observations from a file, or synthetic GeoNames-like
 *      places, built into one index of any residency through the C interface and judged against an
 *      exact scan under one relation.
 *
 *  Places cluster into towns spread over the globe, a town's share of places falling as the inverse
 *  of its rank, so a few cities hold most places and most towns a handful, scattered over a few
 *  kilometres. Queries are the last observations, kept out of the index, and match entries recorded
 *  at any time, so a relation judges positions and view cones alone; the exact scan tests every
 *  sight line over the terrain the index borrows. A VRAM index places its shards over the devices
 *  `USEARCH_DEVICES` names, and an NVMe run builds in RAM, saves a checkpoint and searches a view.
 *
 *  @verbatim
 *  Flag                 Default     Meaning
 *  --base-observations  none        Eleven columns per observation, `.f64bin`, in place of places
 *  --places             10,001,000  Synthetic places, the queries among them, without a file
 *  --relation           near        What a match is: near, visible, sees or seen
 *  --radius             unbounded   Metres a match lies within, widened by both positions' deviations
 *  --terrain            none        Heights above the ellipsoid, `.fbin`, rows running south
 *  --terrain-north      none        Latitude of the terrain's first row, in degrees
 *  --terrain-west       none        Longitude of the terrain's first column, in degrees
 *  --terrain-step       none        Degrees between the terrain's rows and between its columns
 *  @endverbatim
 */
#include <cmath>   // `std::asin`, `std::pow`, `std::isnan`
#include <cstdlib> // `std::strtod` - signed terrain degrees

#include <array>       // `std::array` - a town's centre
#include <limits>      // `std::numeric_limits` - the unbounded time slack
#include <span>        // `std::span` - the observations, from a file or generated
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - places

#include "usearch/geospatial_index.h"

#include "abi.hpp"

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** A uniform real in @b [0,1) from the top 53 random bits. */
template <typename generator_type_>
f64_t uniform_unit(generator_type_& generator) noexcept {
    return static_cast<f64_t>(generator() >> 11) * 0x1p-53;
}

/** Draws @p count GeoNames-like places from @p seed, as the file's docblock describes. */
inline std::vector<usearch_geospatial_observation_t> generate_places(std::size_t count, u64_t seed) {
    constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
    constexpr f64_t degrees_k = 180.0 / 3.14159265358979323846;
    constexpr std::size_t places_per_town_k = 64;
    constexpr f64_t spread_degrees_k = 0.02;
    splitmix64_t random_generator {seed};
    std::size_t const towns = max_of<std::size_t>(1, count / places_per_town_k);
    std::vector<std::array<f64_t, 2>> centres(towns);
    for (std::array<f64_t, 2>& centre : centres)
        centre = {std::asin(2 * uniform_unit(random_generator) - 1) * degrees_k,
                  360 * uniform_unit(random_generator) - 180};
    std::vector<usearch_geospatial_observation_t> places(count);
    for (usearch_geospatial_observation_t& place : places) {
        // A log-uniform rank draws a town in inverse proportion to its rank.
        std::size_t const town = min_of(
            towns - 1,
            static_cast<std::size_t>(std::pow(static_cast<f64_t>(towns), uniform_unit(random_generator))) - 1);
        // The sum of three uniforms approximates a normal offset.
        f64_t offsets[2];
        for (f64_t& offset : offsets)
            offset = spread_degrees_k * (uniform_unit(random_generator) + uniform_unit(random_generator) +
                                         uniform_unit(random_generator) - 1.5);
        f64_t const latitude = min_of(90.0, max_of(-90.0, centres[town][0] + offsets[0]));
        f64_t const longitude = centres[town][1] + offsets[1];
        place = {latitude, longitude > 180 ? longitude - 360 : longitude,
                 nan_k,    nan_k,
                 nan_k,    nan_k,
                 nan_k,    nan_k,
                 nan_k,    nan_k,
                 nan_k};
    }
    return places;
}

/** The signed degrees @p text spells, or NaN when it spells none. */
inline f64_t degrees_of(std::string_view text) noexcept {
    // Text values view the tail of their `argv` entry, so `strtod` stops at its terminator.
    char* end = nullptr;
    f64_t const degrees = std::strtod(text.data(), &end);
    return !text.empty() && end == text.data() + text.size() ? degrees : std::numeric_limits<f64_t>::quiet_NaN();
}

/** Runs the geospatial search benchmark over the flags in @p argv. */
int bench_geospatial_index_search(int argc, char const* const* argv) {
    search_bench_settings_t search;
    std::string_view relation_text = "near", observations_path, terrain_path, terrain_north_text, terrain_west_text;
    count_t places_count {.value = 10'001'000};
    double radius_metres = 0, terrain_step = 0;
    std::array<flag_t, 9> const shared = search_bench_flags(search);
    std::vector<flag_t> flags(shared.begin(), shared.end());
    std::array<flag_t, 8> const own {{
        {"--base-observations", "path",
         "eleven columns per observation, `.f64bin`, in place of synthetic places; none by default",
         &observations_path},
        {"--places", "count", "synthetic places, the queries among them; 10,001,000 by default", &places_count},
        {"--relation", "name", "what a match is: near, visible, sees or seen; near by default", &relation_text},
        {"--radius", "metres", "how far a match lies at most, widened by deviations; unbounded by default",
         &radius_metres},
        {"--terrain", "path", "heights above the ellipsoid, `.fbin`, rows running south; none by default",
         &terrain_path},
        {"--terrain-north", "degrees", "latitude of the terrain's first row", &terrain_north_text},
        {"--terrain-west", "degrees", "longitude of the terrain's first column", &terrain_west_text},
        {"--terrain-step", "degrees", "degrees between the terrain's rows and between its columns", &terrain_step},
    }};
    flags.insert(flags.end(), own.begin(), own.end());
    settings_t const settings = read_settings(argc, argv, "usearch_bench_geospatial_index_search [flags]", flags);
    print(settings);
    usearch_geospatial_relation_t const relation = geospatial_relation_of(relation_text);
    if (relation == usearch_geospatial_relation_unknown_k)
        return fmt::println(stderr, "--relation=\"{}\" does not parse, expected near, visible, sees or seen",
                            relation_text),
               1;

    // Observations come from the file in place, or are generated.
    mapped_rows_t observations_file;
    std::vector<usearch_geospatial_observation_t> places;
    std::span<usearch_geospatial_observation_t const> observations;
    constexpr std::size_t columns_k = sizeof(usearch_geospatial_observation_t) / sizeof(f64_t);
    if (!observations_path.empty()) {
        auto [file, read_status] = mapped_rows_t::open(observations_path);
        if (failed(read_status) || file.scalar() != usearch_scalar_f64_k || file.columns() != columns_k)
            return fmt::println(stderr, "--base-observations=\"{}\" must be a .f64bin of {} columns per observation",
                                observations_path, columns_k),
                   1;
        observations_file = std::move(file);
        observations = {static_cast<usearch_geospatial_observation_t const*>(
                            observations_file.view(0, observations_file.rows()).data),
                        observations_file.rows()};
    }
    else {
        places = generate_places(places_count.value, settings.seed);
        observations = places;
    }

    // A terrain is borrowed in place from its file by the index.
    mapped_rows_t terrain_file;
    usearch_geospatial_terrain_t terrain {};
    if (!terrain_path.empty()) {
        f64_t const north = degrees_of(terrain_north_text), west = degrees_of(terrain_west_text);
        if (std::isnan(north) || std::isnan(west) || !(terrain_step > 0))
            return fmt::println(stderr,
                                "--terrain needs --terrain-north, --terrain-west and a positive --terrain-step"),
                   1;
        auto [file, read_status] = mapped_rows_t::open(terrain_path);
        if (failed(read_status) || file.scalar() != usearch_scalar_f32_k)
            return fmt::println(stderr, "--terrain=\"{}\" must be a .fbin", terrain_path), 1;
        terrain_file = std::move(file);
        usearch_rows_view_t const heights = terrain_file.view(0, terrain_file.rows());
        terrain = {.rows = terrain_file.rows(),
                   .columns = terrain_file.columns(),
                   .north_latitude_degrees = north,
                   .west_longitude_degrees = west,
                   .row_step_degrees = terrain_step,
                   .column_step_degrees = terrain_step,
                   .heights = static_cast<float const*>(heights.data),
                   .heights_stride = heights.stride};
    }

    if (observations.size() <= search.queries.value)
        return fmt::println(stderr, "The file holds {} observations, too few to hold out {} queries",
                            observations.size(), search.queries.value),
               1;
    std::size_t const query_first = observations.size() - search.queries.value;
    std::size_t const base_count = search.max_base_entries.value ? min_of(search.max_base_entries.value, query_first)
                                                                 : query_first;
    fmt::println("- Observations: {} · relation: {} · radius: {} m · terrain: {} × {} heights",
                 observations_path.empty() ? std::string_view {"GeoNames-like"} : observations_path, relation_text,
                 radius_metres, terrain.rows, terrain.columns);

    geospatial_geometry_t const geometry {{search.connectivity.value}, terrain};
    geospatial_queries_t const queries {observations.data() + query_first,
                                        relation,
                                        {radius_metres, std::numeric_limits<f64_t>::infinity()}};
    return run_search_bench<geospatial_verbs_t>(settings, search, geometry, geometry, observations.data(), base_count,
                                                queries, search.queries.value, settings.checkpoint);
}

} // namespace unum::usearch::bench

namespace us = unum::usearch;

int main(int argc, char** argv) { return us::bench::bench_geospatial_index_search(argc, argv); }
