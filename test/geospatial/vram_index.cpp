/**
 *  @file test/geospatial/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The geospatial VRAM cell: observations on an NVIDIA or AMD GPU beside the RAM index on
 *      the same observations, both through the C interface, judged by recall against the exact
 *      matches of each relation, and every reported chord against the exact one.
 *
 *  Collections are uniform over the globe, a dense square kilometre of one city, or that city's
 *  cameras with a heading, a field of view and a range each, which the @c sees and @c seen
 *  relations narrow by view cone on the device. Observations insert in three batches, so the later
 *  batches walk graphs whose reverse links earlier batches merged.
 *
 *  Over rolling hills, half the cameras record no altitude and stand on the ground, and every
 *  relation past @c near also needs a clear sight line. Those collections are no wider than the
 *  walk, so the device judges every entry and its matches equal the exact ones rank for rank: its
 *  clearances must decide as the CPU's do.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::asin`, `std::sqrt`, `std::sin`, `std::cos`, `std::isnan`
#include <cstring> // `std::memcpy` - the matches copied off the arena

#include <algorithm>   // `std::sort`
#include <array>       // `std::array` - the grid points and sight lines
#include <limits>      // `std::numeric_limits` - unrecorded fields
#include <span>        // `std::span` - a query's exact chords
#include <string>      // `std::string` - the cases' labels
#include <string_view> // `std::string_view` - a case label's arguments
#include <vector>      // `std::vector` - the packed terrain

#include "usearch/geospatial_index.h"
#include "usearch/geospatial/ram_metric.hpp" // `i32_serial_geospatial_kit_t` - the CPU's sight lines
#include "usearch/utilities/vram_device.hpp" // `gpu_device_t` - the listing the C interface indexes

#include "harness.hpp"

namespace unum::usearch::test::geospatial {
using namespace unum::usearch::geospatial;

#if USEARCH_ARCH_CUDA_

/** The NVIDIA cells, whose shards the C interface places on CUDA devices. */
using cell_executor_t = cuda_executor_t;
static constexpr gpu_vendor_t cell_vendor_k = gpu_vendor_t::nvidia_k;

#elif USEARCH_ARCH_ROCM_

/** The AMD cells, whose shards the C interface places on ROCm devices. */
using cell_executor_t = hip_executor_t;
static constexpr gpu_vendor_t cell_vendor_k = gpu_vendor_t::amd_k;

#endif

#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

/** How a collection draws its observations, as the file's docblock describes. */
enum class observations_kind_t { global_k, city_k, cameras_k };

/** Draws @p count observations of @p kind. */
static nk::vector<usearch_geospatial_observation_t> generate_observations(observations_kind_t kind, std::size_t count,
                                                                          u64_t seed) {
    constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
    constexpr f64_t degrees_k = 180.0 / 3.14159265358979323846;
    split_mix_generator_t random_generator {seed};
    nk::vector<usearch_geospatial_observation_t> observations = make_vector<usearch_geospatial_observation_t>(count);
    for (usearch_geospatial_observation_t& observation : to_writable_span(observations)) {
        f64_t const across = 2 * uniform_unit(random_generator) - 1, along = 2 * uniform_unit(random_generator) - 1;
        bool const global = kind == observations_kind_t::global_k;
        // Uniform over the sphere: the sine of the latitude is uniform.
        observation = {global ? std::asin(across) * degrees_k : 51.5 + across * 0.0045,
                       global ? along * 180 : -0.1 + along * 0.0072,
                       nan_k,
                       nan_k,
                       nan_k,
                       nan_k,
                       nan_k,
                       nan_k,
                       nan_k,
                       nan_k,
                       nan_k};
        if (kind != observations_kind_t::cameras_k) continue;
        observation.accuracy_metres = 5 * uniform_unit(random_generator);
        observation.heading_degrees = 360 * uniform_unit(random_generator);
        observation.field_of_view_degrees = 30 + 60 * uniform_unit(random_generator);
        observation.range_metres = 100 + 400 * uniform_unit(random_generator);
    }
    return observations;
}

/** Rolling hills under the city's square kilometre and past it, a cell every ten-thousandth of a
 *  degree, whose sight lines the CPU judges with the serial kit every CPU kit shares. */
struct hills_t {
    using kit_t = i32_serial_geospatial_kit_t;
    static constexpr std::size_t rows_k = 101, columns_k = 161;

    nk::matrix<f32_t> heights = make_matrix<f32_t>(rows_k, columns_k);
    std::vector<byte_t> pack;
    terrain_t grid {rows_k, columns_k, 51.505, -0.108, 0.0001, 0.0001, nullptr, columns_k * sizeof(f32_t)};

    /** Raises hills of 25 metres around 30 over about 600 metres either way, and packs them. */
    hills_t() {
        for (std::size_t row = 0; row != rows_k; ++row)
            for (std::size_t column = 0; column != columns_k; ++column)
                heights(row, column) = static_cast<f32_t>(30 + 25 * std::sin(row * 0.11) * std::cos(column * 0.07));
        grid.heights = heights.data();
        nk_size_t bytes = 0;
        [[maybe_unused]] nk_status_t const sized = kit_t::terrain_pack_size(rows_k, columns_k, &bytes);
        pack.resize(bytes);
        std::array<f64_t, 2> const steps = grid.step_metres();
        [[maybe_unused]] nk_status_t const packed = kit_t::terrain_pack(
            heights.data(), rows_k, columns_k, grid.heights_stride, steps[0], steps[1], pack.data(), nullptr);
    }

    /** A handle pointing at the grid, valid while the grid lives. */
    usearch_geospatial_terrain_t terrain() const noexcept {
        return {.rows = rows_k,
                .columns = columns_k,
                .north_latitude_degrees = grid.north_latitude_degrees,
                .west_longitude_degrees = grid.west_longitude_degrees,
                .row_step_degrees = grid.row_step_degrees,
                .column_step_degrees = grid.column_step_degrees,
                .heights = heights.data(),
                .heights_stride = grid.heights_stride};
    }

    /** Ground height under @p latitude_degrees and @p longitude_degrees, zero off the grid. */
    f64_t ground_metres(f64_t latitude_degrees, f64_t longitude_degrees) const noexcept {
        std::array<f64_t, 2> const cell = grid.grid_point(latitude_degrees, longitude_degrees);
        if (!(cell[0] >= 0 && cell[0] <= columns_k - 1.0 && cell[1] >= 0 && cell[1] <= rows_k - 1.0)) return 0;
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0]), static_cast<f32_t>(cell[1])};
        f32_t height = 0;
        [[maybe_unused]] nk_status_t const read = kit_t::terrain_heights(heights.data(), pack.data(), point.data(), 1,
                                                                         &height, nullptr);
        return height == height ? static_cast<f64_t>(height) : 0;
    }

    /** Whether the sight line from @p viewer to @p target clears the hills; one leaving does. */
    bool clears(earth_fixed_t viewer, earth_fixed_t target) const noexcept {
        std::array<f64_t, 3> const viewer_cell = grid.grid_point(viewer), target_cell = grid.grid_point(target);
        std::array<f32_t, 3> const viewpoint {static_cast<f32_t>(viewer_cell[0]), static_cast<f32_t>(viewer_cell[1]),
                                              static_cast<f32_t>(viewer_cell[2])};
        std::array<f32_t, 3> const aim {static_cast<f32_t>(target_cell[0]), static_cast<f32_t>(target_cell[1]),
                                        static_cast<f32_t>(target_cell[2])};
        f32_t clearance = 0;
        [[maybe_unused]] nk_status_t const tested = kit_t::clearances(heights.data(), pack.data(), viewpoint.data(),
                                                                      nullptr, 1, aim.data(), 1, terrain_refraction_k,
                                                                      &clearance, sizeof(f32_t), nullptr);
        return !(clearance < 0);
    }
};

/** The stored form of @p observation, standing on @p hills when it records no altitude, else at the
 *  ellipsoid. */
static earth_observation_t encoded(usearch_geospatial_observation_t const& observation, hills_t const* hills) noexcept {
    f64_t const ground = hills && std::isnan(observation.altitude_metres)
                             ? hills->ground_metres(observation.latitude_degrees, observation.longitude_degrees)
                             : 0;
    return to_earth_observation(
        {observation.latitude_degrees, observation.longitude_degrees, observation.altitude_metres,
         observation.accuracy_metres, observation.heading_degrees, observation.pitch_degrees,
         observation.heading_accuracy_degrees, observation.field_of_view_degrees, observation.range_metres,
         observation.begin_seconds, observation.end_seconds},
        ground);
}

/** Whether @p entry matches @p query under @p relation within @p radius_metres, its sight line
 *  clearing @p hills past @c near, where under @c sees the entry looks at the query. */
static bool related(relation_t relation, f64_t radius_metres, earth_observation_t const& query,
                    earth_observation_t const& entry, hills_t const* hills) noexcept {
    if (!admits(relation, radius_metres, 0, query, entry)) return false;
    if (!hills || relation == relation_t::near_k) return true;
    return relation == relation_t::sees_k ? hills->clears(entry.position, query.position)
                                          : hills->clears(query.position, entry.position);
}

/** The matches of one handle: keys, metres and a count per query, copied off the arena. */
struct handle_matches_t {
    nk::matrix<usearch_key_t> keys;
    nk::matrix<usearch_distance_t> metres;
    nk::vector<std::size_t> counts;
};

/**
 *  @brief One collection of @p count observations of @p kind and @c recall_samples_k held-out
 *      queries, on the GPU and in RAM alike, searched under @p relation within @p radius_metres,
 *      over @p hills when given.
 *
 *  Every match the GPU reports must be admitted, hold the pair's exact chord in metres and ascend,
 *  and the GPU's recall at ten must reach the RAM index's at the same breadth, give or take a
 *  twentieth. A walk as wide as the collection must return the exact matches themselves.
 */
static void test_geospatial_vram_collection(u64_t seed, std::int32_t ordinal, observations_kind_t kind,
                                            std::size_t count, usearch_geospatial_relation_t relation,
                                            f64_t radius_metres, std::size_t expansion,
                                            hills_t const* hills = nullptr) {
    constexpr std::size_t wanted_k = 10, batches_k = 3;
    checks_held_t const held;
    nk::vector<usearch_geospatial_observation_t> observations = generate_observations(kind, count, seed);
    nk::vector<usearch_geospatial_observation_t> queries = generate_observations(kind, recall_samples_k, ~seed);
    // Over hills, every other camera stands on the ground and the rest hover over every hill.
    for (std::size_t offset = 1; hills && offset < count; offset += 2)
        observations[offset].altitude_metres = 60 + offset % 31;
    for (std::size_t offset = 1; hills && offset < recall_samples_k; offset += 2)
        queries[offset].altitude_metres = 60 + offset % 31;
    usearch_geospatial_terrain_t const terrain = hills ? hills->terrain() : usearch_geospatial_terrain_t {};
    nk::vector<usearch_key_t> keys = make_vector<usearch_key_t>(count);
    for (std::size_t offset = 0; offset != count; ++offset) keys[offset] = us::tools::key_of_offset(seed, offset);
    usearch_geospatial_tolerance_t const tolerance {radius_metres, 0};

    // The exact admitted chords of every query, ascending.
    nk::vector<earth_observation_t> stored = make_vector<earth_observation_t>(count);
    for (std::size_t entry = 0; entry != count; ++entry) stored[entry] = encoded(observations[entry], hills);
    nk::matrix<f64_t> exact = make_matrix<f64_t>(recall_samples_k, count);
    nk::vector<std::size_t> exact_counts = make_vector<std::size_t>(recall_samples_k);
    for (std::size_t query = 0; query != recall_samples_k; ++query) {
        earth_observation_t const probe = encoded(queries[query], hills);
        for (std::size_t entry = 0; entry != count; ++entry)
            if (related(static_cast<relation_t>(relation), radius_metres, probe, stored[entry], hills))
                exact(query, exact_counts[query]++) =
                    std::sqrt(static_cast<f64_t>(chord_squared(probe.position, stored[entry].position))) /
                    centimetres_per_metre_k;
        nk::vector_span<f64_t> const chords = exact.span().row(query).as_vector();
        std::span<f64_t> const admitted = to_writable_span(chords).first(exact_counts[query]);
        std::sort(admitted.begin(), admitted.end());
    }

    usearch_outcome_t outcome {};
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena;
    arena_init.outcome = &outcome;
    usearch_arena_init(&arena_init);
    expect_true(!outcome.message);

    // Builds and searches one handle, the GPU's when an executor is given, its outputs on host
    // pages either way.
    auto const build_and_search = [&](usearch_geospatial_index_t index, usearch_executor_t executor) {
        handle_matches_t matched;
        for (std::size_t batch = 0; batch != batches_k && held(); ++batch) {
            std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
            usearch_geospatial_insert_t insert {};
            insert.index = index;
            insert.executor = executor;
            insert.outcome = &outcome;
            insert.count = bound - first;
            insert.keys = {keys.data() + first, sizeof(usearch_key_t)};
            insert.observations = observations.data() + first;
            usearch_geospatial_insert(&insert);
            expect_true(!outcome.message);
        }
        usearch_matches_view_t matches {};
        usearch_geospatial_search_t search {};
        search.index = index;
        search.executor = executor;
        search.outcome = &outcome;
        search.count = recall_samples_k;
        search.queries = queries.data();
        search.relation = relation;
        search.tolerance = tolerance;
        search.matches = &matches;
        search.arena = arena;
        search.wanted = wanted_k;
        search.expansion_factor = expansion;
        if (held()) usearch_geospatial_search(&search);
        if (!expect_true(!outcome.message)) return matched;
        matched.keys = make_matrix<usearch_key_t>(recall_samples_k, matches.columns);
        matched.metres = make_matrix<usearch_distance_t>(recall_samples_k, matches.columns);
        matched.counts = make_vector<std::size_t>(recall_samples_k);
        std::memcpy(matched.keys.data(), matches.keys, matched.keys.size() * sizeof(usearch_key_t));
        std::memcpy(matched.metres.data(), matches.scores, matched.metres.size() * sizeof(usearch_distance_t));
        std::memcpy(matched.counts.data(), matches.counts, recall_samples_k * sizeof(std::size_t));
        return matched;
    };

    // Matches within the tenth exact chord, of ten or of every admitted entry when fewer exist.
    auto const recall = [&](handle_matches_t const& matched) {
        std::size_t hits = 0, possible = 0;
        for (std::size_t query = 0; query != recall_samples_k && !matched.counts.empty(); ++query) {
            std::size_t const expected = min_of(wanted_k, exact_counts[query]);
            if (!expected) continue;
            f64_t const threshold = exact(query, expected - 1) * (1 + 1e-9) + 1e-6;
            std::size_t found = 0;
            for (std::size_t rank = 0; rank != matched.counts[query]; ++rank)
                found += matched.metres(query, rank) <= threshold;
            hits += min_of(found, expected), possible += expected;
        }
        return possible ? static_cast<f64_t>(hits) / possible : 1.0;
    };

    usearch_geospatial_index_t ram_index = nullptr;
    usearch_geospatial_init_ram_t ram_init {};
    ram_init.index = &ram_index;
    ram_init.outcome = &outcome;
    ram_init.terrain = terrain;
    usearch_geospatial_init_ram(&ram_init);
    expect_true(!outcome.message);
    handle_matches_t const ram_matches = !held() ? handle_matches_t {} : build_and_search(ram_index, nullptr);

    std::int32_t const devices[] = {ordinal};
    usearch_vram_options_t const vram {devices, 1};
    usearch_geospatial_index_t vram_index = nullptr;
    usearch_geospatial_init_vram_t vram_init {};
    vram_init.index = &vram_index;
    vram_init.outcome = &outcome;
    vram_init.terrain = terrain;
    vram_init.vram = vram;
    if (held()) usearch_geospatial_init_vram(&vram_init);
    expect_true(!outcome.message);
    usearch_executor_t executor = nullptr;
    usearch_executor_for_vram_t placement {};
    placement.executor = &executor;
    placement.outcome = &outcome;
    placement.vram = vram;
    if (held()) usearch_executor_for_vram(&placement);
    expect_true(!outcome.message);
    handle_matches_t const vram_matches = !held() ? handle_matches_t {} : build_and_search(vram_index, executor);

    for (std::size_t query = 0; query != recall_samples_k && held(); ++query) {
        earth_observation_t const probe = encoded(queries[query], hills);
        expect_equal(vram_matches.counts[query], min_of(wanted_k, exact_counts[query]));
        for (std::size_t rank = 0; rank != vram_matches.counts[query]; ++rank) {
            std::optional<u64_t> const minted = us::tools::offset_of_key(seed, vram_matches.keys(query, rank));
            if (!expect_true(minted && *minted < count)) return;
            u64_t const entry = *minted;
            f64_t const metres = vram_matches.metres(query, rank);
            f64_t const chord = std::sqrt(static_cast<f64_t>(chord_squared(probe.position, stored[entry].position))) /
                                centimetres_per_metre_k;
            expect_true(related(static_cast<relation_t>(relation), radius_metres, probe, stored[entry], hills));
            expect_true(std::abs(metres - chord) <= 1e-6 * max_of(chord, 1.0));
            expect_true(count > expansion || (rank < exact_counts[query] &&
                                              std::abs(metres - exact(query, rank)) <= 1e-6 * max_of(chord, 1.0)));
            expect_true(!rank || vram_matches.metres(query, rank - 1) <= metres);
        }
    }
    f64_t const ram_recall = recall(ram_matches), vram_recall = recall(vram_matches);
    if (vram_recall + recall_margin_k < ram_recall)
        fmt::print(stderr, "  recall at ten over {} observations: {:.3f} on the GPU, {:.3f} in RAM\n", count,
                   vram_recall, ram_recall);
    expect_true(vram_recall + recall_margin_k >= ram_recall);

    // One round on the GPU keeps every entry under the octant rule and concludes a verdict.
    usearch_refine_report_t report {};
    usearch_geospatial_refine_t refine {};
    refine.index = vram_index;
    refine.executor = executor;
    refine.outcome = &outcome;
    refine.report = &report;
    if (held()) usearch_geospatial_refine(&refine);
    expect_true(!outcome.message);
    expect_true(report.verdict == usearch_refine_improved_k || report.verdict == usearch_refine_converged_k);
    expect_true(report.neighborhoods != 0 && report.opening.audited == count);

    usearch_geospatial_free(vram_index);
    usearch_geospatial_free(ram_index);
    usearch_executor_free(executor);
    usearch_arena_free(arena);
}

/** Containment on the GPU through the C verb: every query's count of entries within the radius
 *  equals the host oracle's over the same cities, and no reported key lies outside it. */
static void test_geospatial_vram_within(u64_t seed, std::int32_t ordinal) {
    constexpr std::size_t count_k = 512, queries_k = 16;
    checks_held_t const held;
    constexpr f64_t radius_metres_k = 150;
    nk::vector<usearch_geospatial_observation_t> observations = generate_observations(observations_kind_t::city_k,
                                                                                      count_k, seed);
    nk::vector<usearch_geospatial_observation_t> queries = generate_observations(observations_kind_t::city_k, queries_k,
                                                                                 ~seed);
    nk::vector<usearch_key_t> keys = make_vector<usearch_key_t>(count_k);
    for (std::size_t offset = 0; offset != count_k; ++offset) keys[offset] = us::tools::key_of_offset(seed, offset);

    nk::vector<earth_observation_t> stored = make_vector<earth_observation_t>(count_k);
    for (std::size_t entry = 0; entry != count_k; ++entry) stored[entry] = encoded(observations[entry], nullptr);
    std::array<std::size_t, queries_k> expected {};
    for (std::size_t query = 0; query != queries_k; ++query) {
        earth_observation_t const probe = encoded(queries[query], nullptr);
        for (std::size_t entry = 0; entry != count_k; ++entry)
            expected[query] += related(relation_t::near_k, radius_metres_k, probe, stored[entry], nullptr);
    }

    usearch_outcome_t outcome {};
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena;
    arena_init.outcome = &outcome;
    usearch_arena_init(&arena_init);
    expect_true(!outcome.message);

    std::int32_t const devices[] = {ordinal};
    usearch_vram_options_t const vram {devices, 1};
    usearch_geospatial_index_t index = nullptr;
    usearch_geospatial_init_vram_t init {};
    init.index = &index;
    init.outcome = &outcome;
    init.vram = vram;
    if (held()) usearch_geospatial_init_vram(&init);
    expect_true(!outcome.message);
    usearch_executor_t executor = nullptr;
    usearch_executor_for_vram_t placement {};
    placement.executor = &executor;
    placement.outcome = &outcome;
    placement.vram = vram;
    if (held()) usearch_executor_for_vram(&placement);
    expect_true(!outcome.message);

    usearch_geospatial_insert_t insert {};
    insert.index = index;
    insert.executor = executor;
    insert.outcome = &outcome;
    insert.count = count_k;
    insert.keys = {keys.data(), sizeof(usearch_key_t)};
    insert.observations = observations.data();
    if (held()) usearch_geospatial_insert(&insert);
    expect_true(!outcome.message);

    usearch_matches_tape_view_t matches {};
    std::size_t written = 0;
    usearch_geospatial_within_t within {};
    within.index = index;
    within.executor = executor;
    within.outcome = &outcome;
    within.count = queries_k;
    within.queries = queries.data();
    within.relation = usearch_geospatial_relation_near_k;
    within.tolerance = {radius_metres_k, 0};
    within.matches = &matches;
    within.written = &written;
    within.arena = arena;
    if (held()) usearch_geospatial_within(&within);
    expect_true(!outcome.message);
    for (std::size_t query = 0; query != queries_k && held(); ++query) {
        expect_equal(matches.offsets[query + 1] - matches.offsets[query], expected[query]);
        for (std::size_t match = matches.offsets[query]; match != matches.offsets[query + 1]; ++match)
            expect_true(matches.scores[match] <= radius_metres_k * (1 + 1e-6));
    }

    usearch_geospatial_free(index);
    usearch_executor_free(executor);
    usearch_arena_free(arena);
}

template <>
void run_geospatial_vram_cases<cell_executor_t>(gpu_device_t const& device) {
    if (device.vendor != cell_vendor_k) return;
    // The C interface names devices by their place in the listing, which `device` belongs to.
    std::int32_t const ordinal = static_cast<std::int32_t>(&device - test::gpu_devices().data());
    u64_t const seed = test_run.settings.seed;
    char const* const vendor = constant_of(cell_vendor_k);
    auto const run = [&](char const* call, std::string_view arguments, auto&& check) {
        run_test_case(fmt::format("{}({}, {})", call, vendor, arguments).c_str(), check);
    };
    run("test_geospatial_vram_collection", "observations_kind_t::global_k, usearch_geospatial_relation_near_k", [&] {
        test_geospatial_vram_collection(stream_key(seed, "global"), ordinal, observations_kind_t::global_k, 4096,
                                        usearch_geospatial_relation_near_k, 0, 64);
    });
    run("test_geospatial_vram_collection", "observations_kind_t::city_k, usearch_geospatial_relation_near_k", [&] {
        test_geospatial_vram_collection(stream_key(seed, "city"), ordinal, observations_kind_t::city_k, 4096,
                                        usearch_geospatial_relation_near_k, 0, 64);
    });
    run("test_geospatial_vram_collection", "observations_kind_t::cameras_k, usearch_geospatial_relation_sees_k", [&] {
        test_geospatial_vram_collection(stream_key(seed, "sees"), ordinal, observations_kind_t::cameras_k, 4096,
                                        usearch_geospatial_relation_sees_k, 300, 256);
    });
    run("test_geospatial_vram_collection", "observations_kind_t::cameras_k, usearch_geospatial_relation_seen_k", [&] {
        test_geospatial_vram_collection(stream_key(seed, "seen"), ordinal, observations_kind_t::cameras_k, 4096,
                                        usearch_geospatial_relation_seen_k, 300, 256);
    });
    run("test_geospatial_vram_within", "usearch_geospatial_relation_near_k",
        [&] { test_geospatial_vram_within(stream_key(seed, "within"), ordinal); });
    hills_t const hills;
    run("test_geospatial_vram_collection",
        "observations_kind_t::cameras_k, usearch_geospatial_relation_visible_k, &hills", [&] {
            test_geospatial_vram_collection(stream_key(seed, "visible"), ordinal, observations_kind_t::cameras_k, 256,
                                            usearch_geospatial_relation_visible_k, 0, 256, &hills);
        });
    run("test_geospatial_vram_collection", "observations_kind_t::cameras_k, usearch_geospatial_relation_sees_k, &hills",
        [&] {
            test_geospatial_vram_collection(stream_key(seed, "sees hills"), ordinal, observations_kind_t::cameras_k,
                                            256, usearch_geospatial_relation_sees_k, 300, 256, &hills);
        });
    run("test_geospatial_vram_collection", "observations_kind_t::cameras_k, usearch_geospatial_relation_seen_k, &hills",
        [&] {
            test_geospatial_vram_collection(stream_key(seed, "seen hills"), ordinal, observations_kind_t::cameras_k,
                                            256, usearch_geospatial_relation_seen_k, 300, 256, &hills);
        });
}

#elif USEARCH_ARCH_METAL_

/** The Apple cell: no geospatial kernel serves Apple GPUs to the f64 contract, so placing an index
 *  on one is refused with @c usearch_status_unsupported_k. */
template <>
void run_geospatial_vram_cases<metal_executor_t>(gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::apple_k) return;
    // The C interface names devices by their place in the listing, which `device` belongs to.
    std::int32_t const devices[] = {static_cast<std::int32_t>(&device - test::gpu_devices().data())};
    run_test_case("usearch_geospatial_init_vram", [&] {
        usearch_outcome_t outcome {};
        usearch_geospatial_index_t index = nullptr;
        usearch_geospatial_init_vram_t init {};
        init.index = &index;
        init.outcome = &outcome;
        init.vram = {devices, 1};
        usearch_geospatial_init_vram(&init);
        expect_status(outcome, usearch_status_unsupported_k);
        expect_true(!index);
        usearch_geospatial_free(index);
    });
}

#endif

} // namespace unum::usearch::test::geospatial
