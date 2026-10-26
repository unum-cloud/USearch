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
 *  --metric        chamfer   chamfer, hausdorff, gromov-wasserstein or wasserstein-procrustes
 *  @endverbatim
 */
#include <array>       // `std::array` - the kinds
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - the offsets

#include "usearch/shapes_index.h"

#include "abi.hpp"

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** One shapes kind: its checkpoint token and its C enumerator. */
struct shapes_kind_t {
    std::string_view token;
    usearch_shapes_metric_t metric;
};

inline constexpr std::array<shapes_kind_t, 4> shapes_kinds_k {{
    {"chamfer", usearch_shapes_metric_chamfer_k},
    {"hausdorff", usearch_shapes_metric_hausdorff_k},
    {"gromov-wasserstein", usearch_shapes_metric_gromov_wasserstein_k},
    {"wasserstein-procrustes", usearch_shapes_metric_wasserstein_procrustes_k},
}};

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
    shapes_kind_t const* kind = nullptr;
    for (shapes_kind_t const& candidate : shapes_kinds_k)
        if (candidate.token == metric_name) kind = &candidate;
    if (!kind) return fmt::println(stderr, "--metric=\"{}\" names no shapes kind", metric_name), 1;

    auto [points, points_status] = mapped_rows_t::open(base_points_path);
    if (failed(points_status) || points.columns() != 3 ||
        (points.scalar() != usearch_scalar_f32_k && points.scalar() != usearch_scalar_f16_k))
        return fmt::println(stderr, "--base-points=\"{}\" must hold 3 f32 or f16 columns per point", base_points_path),
               1;
    auto [counts, counts_status] = mapped_rows_t::open(base_counts_path);
    if (failed(counts_status) || counts.columns() != 1)
        return fmt::println(stderr, "--base-counts=\"{}\" must hold 1 column per shape", base_counts_path), 1;
    std::uint32_t const* const point_counts = static_cast<std::uint32_t const*>(counts.view(0, counts.rows()).data);
    std::vector<std::size_t> offsets {0};
    for (std::size_t shape = 0; shape != counts.rows(); ++shape)
        offsets.push_back(offsets.back() + point_counts[shape]);
    if (offsets.back() != points.rows())
        return fmt::println(stderr, "--base-counts sum to {} points, --base-points holds {}", offsets.back(),
                            points.rows()),
               1;
    std::size_t const shapes = counts.rows();
    if (shapes <= search.queries.value)
        return fmt::println(stderr, "The file holds {} shapes, too few to hold out {} queries", shapes,
                            search.queries.value),
               1;
    std::size_t const query_first = shapes - search.queries.value;
    std::size_t const base_count = search.max_base_entries.value ? std::min(search.max_base_entries.value, query_first)
                                                                 : query_first;
    usearch_tape_view_t const tape {points.view(0, points.rows()).data, offsets.data(), points.scalar()};
    fmt::println("- Metric: {} · scalar: {} · {} shapes of {} points", kind->token,
                 usearch_scalar_name(points.scalar()), shapes, points.rows());

    usearch_shapes_metadata_t const geometry {kind->metric, points.scalar(), 0, search.connectivity.value};
    return run_search_bench<shapes_verbs_t>(settings, search, geometry, geometry, tape, base_count,
                                            shapes_verbs_t::slice(tape, query_first), search.queries.value,
                                            settings.checkpoint);
}

} // namespace unum::usearch::bench

namespace us = unum::usearch;

int main(int argc, char** argv) { return us::bench::bench_shapes_index_search(argc, argv); }
