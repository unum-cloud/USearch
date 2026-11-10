/**
 *  @file bench/shapes_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The shapes search benchmark: point clouds end to end and their point counts, built into
 *      one index of any residency through the C interface and judged against an exact scan.
 *
 *  Points are stored in the scalar the file holds. Queries are the file's last shapes, kept out of
 *  the index, so no query finds itself; a hit counts when its distance is within the exact k-th. A
 *  VRAM index places its shards over the devices `USEARCH_DEVICES` names, and an NVMe run builds in
 *  RAM, saves a checkpoint and searches a view of it on storage threads.
 *
 *  @verbatim
 *  Flag            Default   Meaning
 *  --base-points   required  Every shape's points end to end, `.fbin` or `.f16bin` with 3 columns
 *  --base-counts   required  Points per shape, `.ibin` with 1 column
 *  --metric        chamfer   chamfer, hausdorff, gromov_wasserstein or wasserstein_procrustes
 *  @endverbatim
 */
#include <array>       // `std::array` - the flags
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - the flags

#include "usearch/shapes_index.h"

#include "abi.hpp"

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** Runs the shapes search benchmark over the flags in @p argv. */
int bench_shapes_index_search(int argc, char const* const* argv) {
    search_bench_settings_t search;
    std::string_view base_points_path, base_counts_path;
    std::array<flag_t, 9> const shared = search_bench_flags(search);
    std::vector<flag_t> flags(shared.begin(), shared.end());
    flags.push_back({"--base-points", "path", "every shape's points end to end, `.fbin` or `.f16bin` with 3 columns",
                     &base_points_path});
    flags.push_back({"--base-counts", "path", "points per shape, `.ibin` with 1 column", &base_counts_path});
    settings_t const settings = read_settings(
        argc, argv, "usearch_bench_shapes_index_search --base-points <path> --base-counts <path> [flags]", flags);
    print(settings);

    std::string_view const metric_name = search.metric_name.empty() ? "chamfer" : search.metric_name;
    usearch_shapes_metric_t const metric = usearch_shapes_metric_from_name(bytes_view(metric_name));
    if (metric == usearch_shapes_metric_unknown_k)
        return fmt::println(stderr, "--metric=\"{}\" names no shapes kind", metric_name), 1;

    point_clouds_t clouds;
    if (!open_point_clouds(base_points_path, base_counts_path, clouds)) return 1;
    std::size_t const shapes = clouds.size();
    if (shapes <= search.queries.value)
        return fmt::println(stderr, "The file holds {} shapes, too few to hold out {} queries", shapes,
                            search.queries.value),
               1;
    std::size_t const query_first = shapes - search.queries.value;
    std::size_t const base_count = search.max_base_entries.value ? min_of(search.max_base_entries.value, query_first)
                                                                 : query_first;
    usearch_tape_view_t const tape = clouds.view();
    fmt::println("- Metric: {} · scalar: {} · {} shapes of {} points", metric_name,
                 usearch_scalar_name(clouds.points.scalar()), shapes, clouds.points.rows());

    usearch_shapes_metadata_t const geometry {metric, clouds.points.scalar(), 0, search.connectivity.value};
    return run_search_bench<shapes_verbs_t>(settings, search, geometry, geometry, tape, base_count,
                                            shapes_verbs_t::slice(tape, query_first), search.queries.value,
                                            settings.checkpoint);
}

} // namespace unum::usearch::bench

namespace us = unum::usearch;

int main(int argc, char** argv) { return us::bench::bench_shapes_index_search(argc, argv); }
