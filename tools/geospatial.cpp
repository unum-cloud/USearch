/**
 *  @file tools/geospatial.cpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief The geospatial verbs: insert, absorb, refine, split, health and search, each over
 *      SafeGraphs checkpoints at any address.
 *
 *  Observations and queries come as the geospatial benchmark reads them: an <tt>.f64bin</tt> of
 *  eleven columns per observation, in the field order of @c usearch_geospatial_observation_t.
 *  Keys are the observations' numbers, so slices built apart keep distinct keys when merged.
 *  Indexes open without a terrain, so a search asks @c near, @c sees or @c seen.
 */
#include <cstdio> // `stderr`

#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits` - the unbounded time slack
#include <optional>    // `std::optional`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "usearch/geospatial_index.h"

#include "abi.hpp"   // `geospatial_verbs_t`, `mapped_rows_t`
#include "world.hpp" // `checkpoint_insert`, `checkpoint_search`

namespace unum::usearch::tools {

/** Columns of one observation in an <tt>.f64bin</tt>. */
static constexpr std::size_t geospatial_columns_k = sizeof(usearch_geospatial_observation_t) / sizeof(f64_t);

/** Maps the observations of @p path into @p file; none after printing why it cannot. */
static usearch_geospatial_observation_t const* geospatial_open_observations(std::string_view path,
                                                                            mapped_rows_t& file) {
    auto [mapped, status] = mapped_rows_t::open(path);
    file = std::move(mapped);
    void const* const observations = file.view(0, file.rows()).data;
    if (failed(status) || file.scalar() != usearch_scalar_f64_k || file.columns() != geospatial_columns_k ||
        !observations) {
        fmt::println(stderr, "{} must be an .f64bin of {} columns per observation", path, geospatial_columns_k);
        return nullptr;
    }
    return static_cast<usearch_geospatial_observation_t const*>(observations);
}

int modality_insert(geospatial_verbs_t, world_t& world, int argc, char const* const* argv) {
    mapped_rows_t file;
    return checkpoint_insert<geospatial_verbs_t>(
        world, argc, argv, "usearch geospatial insert <observations> --save <address> [flags]", {},
        [&](std::string_view path,
            checkpoint_settings_t const& settings) -> std::optional<checkpoint_entries<geospatial_verbs_t>> {
            usearch_geospatial_observation_t const* const observations = geospatial_open_observations(path, file);
            if (!observations) return std::nullopt;
            geospatial_geometry_t const geometry {{settings.connectivity.value}, {}};
            return checkpoint_entries<geospatial_verbs_t> {observations, file.rows(), geometry};
        });
}

int modality_split(geospatial_verbs_t, world_t& world, int argc, char const* const* argv) {
    return checkpoint_split<geospatial_verbs_t>(
        world, argc, argv, {},
        [](usearch_geospatial_metadata_t const& metadata) { return geospatial_geometry_t {metadata, {}}; });
}

int modality_search(geospatial_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view query_observations, relation_name = "near";
    double radius_metres = 0;
    std::array<flag_t, 3> const flags {{
        {"--query-observations", "path", "required: an .f64bin of one query observation per row", &query_observations},
        {"--relation", "name", "defaults to near: what a match is, near, visible, sees or seen", &relation_name},
        {"--radius", "metres", "defaults to unbounded: how far a match lies at most, widened by deviations",
         &radius_metres},
    }};
    mapped_rows_t file;
    return checkpoint_search<geospatial_verbs_t>(
        world, argc, argv, "usearch geospatial search <address> --query-observations <path> [flags]", flags,
        [&]() -> std::optional<checkpoint_queries<geospatial_verbs_t>> {
            if (query_observations.empty()) {
                fmt::println(stderr, "--query-observations is required");
                return std::nullopt;
            }
            usearch_geospatial_relation_t const relation = geospatial_relation_of(relation_name);
            if (relation == usearch_geospatial_relation_unknown_k) {
                fmt::println(stderr, "--relation=\"{}\" names none of near, visible, sees or seen", relation_name);
                return std::nullopt;
            }
            usearch_geospatial_observation_t const* const observations = geospatial_open_observations(
                query_observations, file);
            if (!observations) return std::nullopt;
            geospatial_queries_t const queries {observations,
                                                relation,
                                                {radius_metres, std::numeric_limits<f64_t>::infinity()}};
            return checkpoint_queries<geospatial_verbs_t> {queries, file.rows()};
        });
}

} // namespace unum::usearch::tools
