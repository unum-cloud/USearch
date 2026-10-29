/**
 *  @file test/abi/shapes.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The shapes suite over the C interface, swept on the diagonal: every metric at the best
 *      class and Chamfer at every admitted class, over each compiled residency at one shard and at
 *      three.
 *
 *  Shapes are eight template clouds of 24 points, each entry keeping 16 to 24 of them under a
 *  hundredth of jitter, so neighbors cluster by template while point counts vary per entry; the
 *  alignment metrics, which iterate over every pair of points, take fewer and smaller clouds. Each
 *  cell checks the verbs a caller reaches: the shapes land under their keys across the shards and
 *  each counts once, a walk keeps most of the exact scan's top ten, a RAM index refines and absorbs
 *  while a GPU one refuses both, the index saves and reloads into every residency and as a view, an
 *  NVMe view answers and refuses every write, and a cleared index holds nothing. Stored poses round
 *  twice, packed at insert and unpacked at save, so a RAM index's poses come back within that band
 *  of each cloud's extent after each of three save and load cycles, and a reloaded index ranks
 *  within the distance band it implies.
 */
#include <cmath> // `std::sqrt` - a point's move from its coordinates' bands

#include <algorithm> // `std::minmax_element` - a cloud's extent
#include <array>     // `std::array` - the metrics and their names
#include <string>    // `std::string` - the cells' names
#include <vector>    // `std::vector` - the points, the offsets

#include "usearch/shapes_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Shapes a cell builds, the queries held out past them, the donor's shapes, the points per
 *  template cloud, the fewest an entry keeps, and the graph degree, zero deriving it. */
struct shapes_sizes_t {
    std::size_t shapes = 0, queries = 0, donor_shapes = 0, points = 0, fewest_points = 0, connectivity = 0;
};

/** The sizes of the point-set metrics' cells, and the smaller ones of the alignments, whose every
 *  distance iterates a transport plan, so a refinement pass costs minutes at the derived degree. */
static constexpr shapes_sizes_t shapes_sizes_k {512, 64, 128, 24, 16, 0}, alignment_sizes_k {64, 16, 16, 8, 6, 8};

/** The breadth every walk takes. */
static constexpr std::size_t shapes_expansion_k = 128;

/** Points end to end and the offsets around them, as the C interface takes a tape of shapes. */
struct shapes_tape_t {
    std::vector<f32_t> coordinates;
    std::vector<std::size_t> offsets {0};

    usearch_tape_view_t view() const noexcept { return {coordinates.data(), offsets.data(), usearch_scalar_f32_k}; }
};

/** @p count shapes from eight template clouds of @p sizes ' points, each keeping a prefix of
 *  @p sizes ' fewest points or more under a hundredth of jitter. */
static shapes_tape_t generate_shapes(std::size_t count, shapes_sizes_t const& sizes, u64_t seed) {
    constexpr std::size_t templates_k = 8;
    splitmix64_t random_generator {seed};
    uniform_between<f64_t> const unit {-1, 1}, jitter {-0.01, 0.01};
    uniform_below<u32_t> const template_distribution {templates_k};
    uniform_below<u32_t> const kept_distribution {static_cast<u32_t>(sizes.points - sizes.fewest_points + 1)};
    std::vector<f32_t> templates(templates_k * sizes.points * 3);
    for (f32_t& coordinate : templates) coordinate = static_cast<f32_t>(unit(random_generator));
    shapes_tape_t tape;
    for (std::size_t shape = 0; shape != count; ++shape) {
        f32_t const* const cloud = templates.data() + template_distribution(random_generator) * sizes.points * 3;
        std::size_t const kept = sizes.fewest_points + kept_distribution(random_generator);
        for (std::size_t coordinate = 0; coordinate != kept * 3; ++coordinate)
            tape.coordinates.push_back(cloud[coordinate] + static_cast<f32_t>(jitter(random_generator)));
        tape.offsets.push_back(tape.offsets.back() + kept);
    }
    return tape;
}

/** The unit roundoff of the coordinates a shapes index stores as @p quantization. */
static double unit_roundoff_of(usearch_scalar_t quantization) noexcept {
    return quantization == usearch_scalar_f32_k ? 0x1p-24 : 0x1p-11;
}

/**
 *  @brief How far a reloaded shape's distance may move: three roundings at @p unit_roundoff.
 *
 *  The pack at insert and the unpack at save each move a coordinate of the widest cloud of @p tape
 *  by its extent's rounding, so a point by √3 of that, and the score itself rounds to the storage
 *  dtype on either side of the comparison. Chamfer and Hausdorff move no farther than their points
 *  do, and no score of @p tape exceeds the diagonal of the box every coordinate falls in, which
 *  bounds the score's own rounding.
 */
static double shapes_score_band(shapes_tape_t const& tape, double unit_roundoff) {
    double widest = 0;
    for (std::size_t shape = 0; shape + 1 < tape.offsets.size(); ++shape) {
        auto const first = tape.coordinates.begin() + tape.offsets[shape] * 3;
        auto const extremes = std::minmax_element(first, tape.coordinates.begin() + tape.offsets[shape + 1] * 3);
        widest = std::max(widest, static_cast<double>(*extremes.second - *extremes.first));
    }
    auto const extremes = std::minmax_element(tape.coordinates.begin(), tape.coordinates.end());
    double const diagonal = std::sqrt(3.0) * (static_cast<double>(*extremes.second) - *extremes.first);
    return std::sqrt(3.0) * pose_rounding_band(widest, unit_roundoff) + 2 * unit_roundoff * diagonal;
}

/** The coordinates the first @p count keys of @p index read back as in f32, end to end, each key's
 *  cloud starting at its entry of @p offsets, which ends with the total. */
static std::vector<float> poses_of(handle<shapes_verbs_t>& index, std::size_t count, std::vector<std::size_t>& offsets,
                                   std::size_t& failures) {
    std::vector<float> poses;
    offsets.assign(1, 0);
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_tape_view_t shapes {};
        std::size_t written = 0;
        usearch_shapes_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.shapes = &shapes, get.written = &written, get.arena = index.session->arena;
        get.outputs_scalar = usearch_scalar_f32_k;
        usearch_shapes_get(&get);
        failures += check(outcome) + check(written == 1);
        if (outcome.status != usearch_status_success_k || written != 1) break;
        float const* const coordinates = static_cast<float const*>(shapes.data);
        poses.insert(poses.end(), coordinates + shapes.offsets[0] * 3, coordinates + shapes.offsets[1] * 3);
        offsets.push_back(poses.size());
    }
    return poses;
}

/** A RAM index's poses come back within their rounding band after each of three save and load
 *  cycles under @p stem. */
static std::size_t expect_poses_round_trip(handle<shapes_verbs_t>& index, std::string const& stem) {
    constexpr std::size_t keys_k = 64, cycles_k = 3;
    std::size_t failures = 0;
    double const unit_roundoff = unit_roundoff_of(index.geometry.quantization);
    std::vector<std::size_t> offsets, reloaded_offsets;
    std::vector<float> const original = poses_of(index, keys_k, offsets, failures);
    handle<shapes_verbs_t> cycled {nullptr, *index.session, usearch_residency_ram_k, index.geometry};
    for (std::size_t cycle = 0; cycle != cycles_k && !failures; ++cycle) {
        handle<shapes_verbs_t>& source = cycle ? cycled : index;
        std::string const path = fmt::format("{}-cycle{}.safegraphs", stem, cycle);
        usearch_outcome_t outcome {};
        shapes_verbs_t::save(source.get(), source.executor(), bytes_view(path), &outcome);
        failures += check(outcome);
        std::vector<std::int32_t> placement;
        shapes_verbs_t::index_t loaded = nullptr;
        shapes_verbs_t::load_ram(&loaded, index.session->executor, bytes_view(path),
                                 index.session->ram_options(0, placement), index.geometry, &outcome);
        failures += check(outcome);
        cycled = {loaded, *index.session, usearch_residency_ram_k, index.geometry};
        if (cycled) {
            std::vector<float> const reloaded = poses_of(cycled, keys_k, reloaded_offsets, failures);
            failures += check(reloaded_offsets == offsets) +
                        expect_poses_within_rounding(original, reloaded, offsets, unit_roundoff);
        }
        for (std::string const& file : files_starting_with(fmt::format("{}-cycle{}", stem, cycle)))
            std::remove(file.c_str());
    }
    return failures;
}

/** One metric of one cell, named @p name, as the file's docblock describes. */
static std::size_t check_shapes_metric(sweep_cell_t const& cell, session_t& session, usearch_shapes_metadata_t geometry,
                                       std::string const& name, u64_t seed) {
    bool const aligned = geometry.metric == usearch_shapes_metric_gromov_wasserstein_k ||
                         geometry.metric == usearch_shapes_metric_wasserstein_procrustes_k;
    shapes_sizes_t const sizes = aligned ? alignment_sizes_k : shapes_sizes_k;
    geometry.connectivity = sizes.connectivity;
    handle<shapes_verbs_t> index = open_index<shapes_verbs_t>(session, geometry, cell.shards);
    if (!index) return 1;
    shapes_tape_t const shapes = generate_shapes(sizes.shapes + sizes.queries, sizes, seed);
    shapes_tape_t const donated = generate_shapes(sizes.donor_shapes, sizes, ~seed);
    usearch_tape_view_t const queries = shapes_verbs_t::slice(shapes.view(), sizes.shapes);

    std::size_t failures = expect_built(cell, index, shapes.view(), sizes.shapes);
    if (failures) return failures;
    failures += expect_walk_recall(index, queries, sizes.queries, shapes_expansion_k, 0.8);
    failures += expect_refined(index, usearch_status_unsupported_k);
    std::string const stem = fmt::format("/tmp/usearch-abi-shapes-{:016x}", stream_key(seed, name));
    if (cell.residency == usearch_residency_ram_k) failures += expect_poses_round_trip(index, stem + "-poses");
    failures += expect_persistence(cell, index, shapes.view(), queries, sizes.queries, shapes_expansion_k, 0.8, stem,
                                   saved_bytes_t::differing_k,
                                   shapes_score_band(shapes, unit_roundoff_of(geometry.quantization)));
    if (cell.residency == usearch_residency_nvme_k) return failures;
    failures += expect_absorbed(index, donated.view(), sizes.donor_shapes, sizes.shapes, usearch_status_unsupported_k);
    return failures + expect_cleared(index);
}

std::size_t check_shapes_abi(settings_t const& settings) {
    constexpr std::array<usearch_shapes_metric_t, 4> metrics_k {
        usearch_shapes_metric_chamfer_k, usearch_shapes_metric_hausdorff_k, usearch_shapes_metric_gromov_wasserstein_k,
        usearch_shapes_metric_wasserstein_procrustes_k};
    constexpr std::array<std::string_view, 4> metric_names_k {"chamfer", "hausdorff", "gromov-wasserstein",
                                                              "wasserstein-procrustes"};
    constexpr std::size_t gpu_metrics_k = 2;
    std::size_t failures = 0;
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_unknown_k)) {
        // The best class sweeps every metric, and a pinned class the first.
        std::size_t const metrics_count = cell.cpu_class == usearch_cpu_class_unknown_k ? metrics_k.size() : 1;
        for (std::size_t metric = 0; metric != metrics_count; ++metric) {
            std::string const name = fmt::format("shapes {} {}", name_of(cell), metric_names_k[metric]);
            if (!settings.filter.selects(name)) continue;
            // GPUs serve the point-set metrics alone; the alignments refuse at open.
            if (cell.residency == usearch_residency_vram_k && metric >= gpu_metrics_k) {
                report(name.c_str(), "served on the CPU alone");
                continue;
            }
            session_t session = open_session(settings_of(settings, cell), cell.cpu_class);
            usearch_shapes_metadata_t const geometry {metrics_k[metric], usearch_scalar_f16_k, 0, 0};
            failures += report(
                name.c_str(),
                session ? check_shapes_metric(cell, session, geometry, name, stream_key(settings.seed, name)) : 1);
        }
    }
    return failures;
}

} // namespace unum::usearch::test::abi
