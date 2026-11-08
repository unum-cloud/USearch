/**
 *  @file tools/shapes.cpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief The shapes verbs: insert, absorb, refine, split, health and search, each over SafeGraphs
 *      checkpoints at any address.
 *
 *  Point clouds come as the shapes benchmark reads them: the points of every cloud end to end, 3
 *  f32 or f16 columns per point, and the points per cloud, 1 u32 column per cloud, both in
 *  RetriEval files. Points are stored in the scalar the file holds, and keys are the clouds'
 *  numbers, so slices built apart keep distinct keys when merged.
 */
#include <cstdio> // `stderr`

#include <array>       // `std::array`
#include <optional>    // `std::optional`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "usearch/shapes_index.h"

#include "abi.hpp"   // `shapes_verbs_t`, `open_point_clouds`
#include "world.hpp" // `checkpoint_insert`, `checkpoint_search`

namespace unum::usearch::tools {

int modality_insert(shapes_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view counts_path, metric_name = "chamfer";
    std::array<flag_t, 2> const flags {{
        {"--counts", "path", "required: points per cloud, an .ibin of 1 column", &counts_path},
        {"--metric", "name", "defaults to chamfer: chamfer, hausdorff, gromov_wasserstein or wasserstein_procrustes",
         &metric_name},
    }};
    point_clouds_t clouds;
    return checkpoint_insert<shapes_verbs_t>(
        world, argc, argv, "usearch shapes insert <points> --counts <path> --save <address> [flags]", flags,
        [&](std::string_view path,
            checkpoint_settings_t const& settings) -> std::optional<checkpoint_entries<shapes_verbs_t>> {
            usearch_shapes_metric_t const metric = usearch_shapes_metric_from_name(bytes_view(metric_name));
            if (metric == usearch_shapes_metric_unknown_k) {
                fmt::println(stderr, "--metric=\"{}\" names no metric", metric_name);
                return std::nullopt;
            }
            if (counts_path.empty()) {
                fmt::println(stderr, "--counts is required");
                return std::nullopt;
            }
            if (!open_point_clouds(path, counts_path, clouds)) return std::nullopt;
            usearch_shapes_metadata_t const geometry {metric, clouds.points.scalar(), 0, settings.connectivity.value};
            return checkpoint_entries<shapes_verbs_t> {clouds.view(), clouds.size(), geometry};
        });
}

int modality_split(shapes_verbs_t, world_t& world, int argc, char const* const* argv) {
    return checkpoint_split<shapes_verbs_t>(world, argc, argv, {},
                                            [](usearch_shapes_metadata_t const& metadata) { return metadata; });
}

int modality_search(shapes_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view points_path, counts_path;
    std::array<flag_t, 2> const flags {{
        {"--query-points", "path", "required: every query cloud's points end to end, as insert reads them",
         &points_path},
        {"--query-counts", "path", "required: points per query cloud, an .ibin of 1 column", &counts_path},
    }};
    point_clouds_t clouds;
    return checkpoint_search<shapes_verbs_t>(
        world, argc, argv, "usearch shapes search <address> --query-points <path> --query-counts <path> [flags]", flags,
        [&]() -> std::optional<checkpoint_queries<shapes_verbs_t>> {
            if (points_path.empty() || counts_path.empty()) {
                fmt::println(stderr, "--query-points and --query-counts are required");
                return std::nullopt;
            }
            if (!open_point_clouds(points_path, counts_path, clouds)) return std::nullopt;
            return checkpoint_queries<shapes_verbs_t> {clouds.view(), clouds.size()};
        });
}

} // namespace unum::usearch::tools
