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
 *  each counts once, a walk keeps most of the exact scan's top ten, one RAM search's matches hint
 *  the next while other residencies refuse hints, a RAM index refines, through a plan too, and
 *  gathers a second source through a reshard plan while a GPU one refuses all three, an audit
 *  reaches every entry, cursors walk every entry and a sample draws distinct ones, the index saves
 *  into memory and back and into every residency and as a view, an NVMe view answers, audits, walks
 *  and samples and refuses every write, a RAM index reshards into two targets through a plan with
 *  every key in one, a cleared index holds nothing and ends a cursor opened before it, and every
 *  metric's name reads back. Stored poses round twice, packed at insert and unpacked at save, so a
 *  RAM index's poses come back within that band of each cloud's extent after each of three save and
 *  load cycles, and a reloaded index ranks within the distance band it implies.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath> // `std::sqrt` - a point's move from its coordinates' bands

#include <algorithm> // `std::minmax_element` - a cloud's extent
#include <array>     // `std::array` - the metrics and their names
#include <string>    // `std::string` - the cells' names
#include <vector>    // `std::vector` - the points, the offsets

#include "usearch/shapes_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Shapes a cell builds, the shapes a second source holds, the points per template cloud, the
 *  fewest an entry keeps, and the graph degree, zero deriving it; the queries held out past them
 *  number @c recall_samples_k. */
struct shapes_sizes_t {
    std::size_t shapes = 0, packed_shapes = 0, points = 0, fewest_points = 0, connectivity = 0;
};

/** The sizes of the point-set metrics' cells, and the smaller ones of the alignments, whose every
 *  distance iterates a transport plan, so a refinement pass costs minutes at the derived degree. */
static constexpr shapes_sizes_t shapes_sizes_k {512, 128, 24, 16, 0}, alignment_sizes_k {64, 16, 8, 6, 8};

/** The breadth every walk takes. */
static constexpr std::size_t shapes_expansion_k = 128;

/** Points end to end and the offsets around them, as the C interface takes a tape of shapes. */
struct shapes_tape_t {
    std::vector<util::f32_t> coordinates;
    std::vector<std::size_t> offsets {0};

    usearch_tape_view_t view() const noexcept {
        return tape_of(coordinates.data(), offsets.data(), usearch_scalar_f32_k);
    }
};

/** @p count shapes from eight template clouds of @p sizes ' points, each keeping a prefix of
 *  @p sizes ' fewest points or more under a hundredth of jitter. */
static shapes_tape_t generate_shapes(std::size_t count, shapes_sizes_t const& sizes, util::u64_t seed) {
    constexpr std::size_t templates_k = 8;
    split_mix_generator_t random_generator {seed};
    uniform_between<util::f64_t> const unit {-1, 1}, jitter {-0.01, 0.01};
    util::uniform_below<util::u32_t> const template_distribution {templates_k};
    util::uniform_below<util::u32_t> const kept_distribution {
        static_cast<util::u32_t>(sizes.points - sizes.fewest_points + 1)};
    nk::matrix<util::f32_t> templates = make_matrix<util::f32_t>(templates_k, sizes.points * 3);
    for (util::f32_t& coordinate : util::to_writable_span(templates))
        coordinate = static_cast<util::f32_t>(unit(random_generator));
    shapes_tape_t tape;
    for (std::size_t shape = 0; shape != count; ++shape) {
        nk::vector_view<util::f32_t> const cloud =
            templates.view().row(template_distribution(random_generator)).as_vector();
        std::size_t const kept = sizes.fewest_points + kept_distribution(random_generator);
        for (std::size_t coordinate = 0; coordinate != kept * 3; ++coordinate)
            tape.coordinates.push_back(cloud[coordinate] + static_cast<util::f32_t>(jitter(random_generator)));
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
        widest = util::max_of(widest, static_cast<double>(*extremes.second - *extremes.first));
    }
    auto const extremes = std::minmax_element(tape.coordinates.begin(), tape.coordinates.end());
    double const diagonal = std::sqrt(3.0) * (static_cast<double>(*extremes.second) - *extremes.first);
    return std::sqrt(3.0) * pose_rounding_band(widest, unit_roundoff) + 2 * unit_roundoff * diagonal;
}

/** The coordinates the first @p count keys of @p index read back as in f32 into @p poses, end to
 *  end, each key's cloud starting at its entry of @p offsets, which ends with the total. */
static void poses_of(handle<shapes_verbs_t>& index, std::size_t count, std::vector<std::size_t>& offsets,
                     std::vector<util::f32_t>& poses) {
    poses.clear();
    offsets.assign(1, 0);
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_tape_view_t shapes {};
        std::size_t written = 0;
        usearch_shapes_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.shapes = &shapes, get.written = &written, get.arena = index.arena();
        get.outputs_scalar = usearch_scalar_f32_k;
        usearch_shapes_get(&get);
        expect_success(outcome);
        expect_true(written == 1);
        if (outcome.status != usearch_status_success_k || written != 1) break;
        util::f32_t const* const coordinates = static_cast<util::f32_t const*>(shapes.data);
        poses.insert(poses.end(), coordinates + tape_offset(shapes, 0) * 3, coordinates + tape_offset(shapes, 1) * 3);
        offsets.push_back(poses.size());
    }
}

/** A RAM index's poses come back within their rounding band after each of three save and load
 *  cycles under @p stem. */
static void expect_poses_round_trip(handle<shapes_verbs_t>& index, std::string const& stem) {
    constexpr std::size_t keys_k = 64, cycles_k = 3;
    double const unit_roundoff = unit_roundoff_of(index.geometry.quantization);
    std::vector<std::size_t> offsets, reloaded_offsets;
    std::vector<util::f32_t> original, reloaded;
    poses_of(index, keys_k, offsets, original);
    if (!expect_equal(offsets.size(), keys_k + 1)) return;
    handle<shapes_verbs_t> cycled {nullptr, *index.session, usearch_residency_ram_k, index.geometry};
    for (std::size_t cycle = 0; cycle != cycles_k; ++cycle) {
        handle<shapes_verbs_t>& source = cycle ? cycled : index;
        std::string const path = fmt::format("{}-cycle{}.safegraphs", stem, cycle);
        usearch_outcome_t outcome {};
        shapes_verbs_t::save(source.get(), source.executor(), bytes_view(path), &outcome);
        if (!expect_success(outcome)) return;
        st::vector<std::int32_t> placement;
        shapes_verbs_t::index_t loaded = nullptr;
        shapes_verbs_t::load_ram(&loaded, index.session->executor, bytes_view(path),
                                 index.session->ram_options(0, placement), index.geometry, &outcome);
        if (!expect_success(outcome)) return;
        cycled = {loaded, *index.session, usearch_residency_ram_k, index.geometry};
        if (cycled) {
            poses_of(cycled, keys_k, reloaded_offsets, reloaded);
            if (!expect_true(reloaded_offsets == offsets)) return;
            expect_poses_within_rounding(original, reloaded, offsets, unit_roundoff);
        }
        for (sz::string_t const& file : files_starting_with(fmt::format("{}-cycle{}", stem, cycle)))
            std::remove(file.c_str());
    }
}

/** One metric of one cell, as the file's docblock describes; GPUs serve the point-set metrics
 *  alone, the alignments refusing at open. */
static case_result_t test_shapes_metric(sweep_cell_t const& cell, usearch_shapes_metadata_t geometry,
                                        util::u64_t seed) {
    bool const point_set = geometry.metric == usearch_shapes_metric_chamfer_k ||
                           geometry.metric == usearch_shapes_metric_hausdorff_k;
    if (cell.residency == usearch_residency_vram_k && !point_set)
        return case_result_t::skipped("needs usearch_residency_ram_k");
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return {};
    bool const aligned = geometry.metric == usearch_shapes_metric_gromov_wasserstein_k ||
                         geometry.metric == usearch_shapes_metric_wasserstein_procrustes_k;
    shapes_sizes_t const sizes = aligned ? alignment_sizes_k : shapes_sizes_k;
    geometry.connectivity = sizes.connectivity;
    handle<shapes_verbs_t> index = open_index<shapes_verbs_t>(session, geometry, cell.shards);
    if (!expect_true(bool(index))) return {};
    shapes_tape_t const shapes = generate_shapes(sizes.shapes + recall_samples_k, sizes, seed);
    shapes_tape_t const second_source = generate_shapes(sizes.packed_shapes, sizes, ~seed);
    usearch_tape_view_t const queries = shapes_verbs_t::slice(shapes.view(), sizes.shapes);

    if (!check_built(cell, index, shapes.view(), sizes.shapes)) return {};
    expect_walk_recall(index, queries, recall_samples_k, shapes_expansion_k, 0.8);
    // An NVMe cell builds in RAM and views its checkpoint only once saved, so it walks hinted.
    expect_hints(index, queries, recall_samples_k, shapes_expansion_k,
                 index.residency == usearch_residency_ram_k ? usearch_status_success_k : usearch_status_unsupported_k);
    expect_refined(index, queries, recall_samples_k, shapes_expansion_k, usearch_status_success_k);
    expect_health(index, sizes.shapes);
    expect_cursor(index, sizes.shapes);
    expect_sample(index, sizes.shapes);
    expect_buffer_round_trip(index, queries, recall_samples_k, shapes_expansion_k, 0.8, usearch_status_unsupported_k);
    scratch_directory_t const scratch;
    std::string const stem = scratch.path("checkpoint");
    if (cell.residency == usearch_residency_ram_k) expect_poses_round_trip(index, stem + "-poses");
    expect_widened(index, stem + "-widened");
    expect_persistence(cell, index, shapes.view(), queries, recall_samples_k, shapes_expansion_k, 0.8, stem,
                       saved_bytes_t::differing_k, shapes_score_band(shapes, unit_roundoff_of(geometry.quantization)));
    if (cell.residency == usearch_residency_nvme_k) return {};
    expect_refine_plan(index, shapes.view(), sizes.shapes, usearch_status_success_k);
    expect_reshard_gather(index, second_source.view(), sizes.packed_shapes, sizes.shapes, usearch_status_success_k);
    expect_true(describe(index).total_points == shapes.offsets[sizes.shapes] + second_source.offsets.back());
    expect_reshard_cut(index, usearch_status_success_k);
    expect_cleared(index, shapes.view());
    return {};
}

void run_shapes_abi_cases() {
    constexpr std::array<usearch_shapes_metric_t, 4> metrics_k {
        usearch_shapes_metric_chamfer_k, usearch_shapes_metric_hausdorff_k, usearch_shapes_metric_gromov_wasserstein_k,
        usearch_shapes_metric_wasserstein_procrustes_k};
    constexpr std::array<std::string_view, 4> metric_constants_k {
        "usearch_shapes_metric_chamfer_k", "usearch_shapes_metric_hausdorff_k",
        "usearch_shapes_metric_gromov_wasserstein_k", "usearch_shapes_metric_wasserstein_procrustes_k"};
    // The streams keep the names the cells had before their labels spelled the constants.
    constexpr std::array<std::string_view, 4> metric_names_k {"chamfer", "hausdorff", "gromov-wasserstein",
                                                              "wasserstein-procrustes"};
    run_test_case("test_metric_names<shapes_verbs_t>", test_metric_names<shapes_verbs_t>);
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_unknown_k)) {
        // The best class sweeps every metric, and a class set as the ceiling the first.
        std::size_t const metrics_count = cell.cpu_class == usearch_cpu_class_unknown_k ? metrics_k.size() : 1;
        for (std::size_t metric = 0; metric != metrics_count; ++metric) {
            std::string const name = fmt::format("shapes {} {}", name_of(cell), metric_names_k[metric]);
            usearch_shapes_metadata_t const geometry {metrics_k[metric], usearch_scalar_f16_k, 0, 0};
            run_test_case(fmt::format("test_shapes_metric({}, {}, {}, {})", constant_of(cell.residency),
                                      constant_of(cell.cpu_class), cell.shards, metric_constants_k[metric])
                              .c_str(),
                          test_shapes_metric, cell, geometry, stream_key(test_run.settings.seed, name));
        }
    }
}

} // namespace unum::usearch::test::abi
