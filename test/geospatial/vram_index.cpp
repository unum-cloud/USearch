/**
 *  @file test/geospatial/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The geospatial VRAM cell: observations on an NVIDIA GPU beside the RAM index on the same
 *      observations, both through the C interface, judged by recall against the exact matches of
 *      each relation, and every reported chord against the exact one.
 *
 *  Collections are uniform over the globe, a dense square kilometre of one city, or that city's
 *  cameras with a heading, a field of view and a range each, which the @c sees and @c seen
 *  relations narrow by view cone on the device. Rows insert in three batches, so the later batches
 *  walk graphs whose reverse links earlier batches merged.
 *
 *  Over rolling hills, half the cameras record no altitude and stand on the ground, and every
 *  relation past @c near also needs a clear sight line. Those collections are no wider than the
 *  walk, so the device judges every entry and its matches equal the exact ones rank for rank: its
 *  clearances must decide as the CPU's do.
 */
#include <cmath> // `std::asin`, `std::sqrt`, `std::sin`, `std::cos`, `std::isnan`

#include <algorithm> // `std::sort`
#include <array>     // `std::array` - the grid points and sight lines
#include <limits>    // `std::numeric_limits` - unrecorded fields
#include <vector>    // `std::vector` - observations, exact matches

#include "usearch/geospatial_index.h"
#include "usearch/geospatial/ram_metric.hpp" // `i32_serial_geospatial_kit_t` - the CPU's sight lines
#include "usearch/utilities/vram_device.hpp" // `gpu_device_t` - the listing the C interface indexes

#include "harness.hpp"

namespace unum::usearch::test::geospatial {
using namespace unum::usearch::geospatial;

#if USEARCH_ARCH_CUDA_

/** How a collection draws its observations, as the file's docblock describes. */
enum class observations_kind_t { global_k, city_k, cameras_k };

/** Draws @p count observations of @p kind. */
static std::vector<usearch_geospatial_observation_t> generate_observations(observations_kind_t kind, std::size_t count,
                                                                           u64_t seed) {
    constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
    constexpr f64_t degrees_k = 180.0 / 3.14159265358979323846;
    splitmix64_t random_generator {seed};
    std::vector<usearch_geospatial_observation_t> observations(count);
    for (usearch_geospatial_observation_t& observation : observations) {
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

    std::vector<f32_t> heights = std::vector<f32_t>(rows_k * columns_k);
    std::vector<byte_t> pack;
    terrain_t grid {rows_k, columns_k, 51.505, -0.108, 0.0001, 0.0001, nullptr, columns_k * sizeof(f32_t)};

    /** Raises hills of 25 metres around 30 over about 600 metres either way, and packs them. */
    hills_t() {
        for (std::size_t row = 0; row != rows_k; ++row)
            for (std::size_t column = 0; column != columns_k; ++column)
                heights[row * columns_k + column] = static_cast<f32_t>(30 + 25 * std::sin(row * 0.11) *
                                                                                std::cos(column * 0.07));
        grid.heights = heights.data();
        nk_size_t bytes = 0;
        [[maybe_unused]] nk_status_t const sized = kit_t::terrain_pack_size(rows_k, columns_k, &bytes);
        pack.resize(bytes);
        std::array<f64_t, 2> const steps = grid.step_metres();
        [[maybe_unused]] nk_status_t const packed = kit_t::terrain_pack(
            heights.data(), rows_k, columns_k, grid.heights_stride, steps[0], steps[1], pack.data(), nullptr);
    }

    /** The grid as a handle borrows it. */
    usearch_geospatial_terrain_t borrowed() const noexcept {
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
    std::vector<usearch_key_t> keys;
    std::vector<usearch_distance_t> metres;
    std::vector<std::size_t> counts;
};

/**
 *  @brief One collection of @p count observations of @p kind and @p queries_count held-out queries,
 *      on the GPU and in RAM alike, searched under @p relation within @p radius_metres, over
 *      @p hills when given.
 *
 *  Every match the GPU reports must be admitted, carry the pair's exact chord in metres and ascend,
 *  and the GPU's recall at ten must reach the RAM index's at the same breadth, give or take a
 *  twentieth. A walk as wide as the collection must land the exact matches themselves.
 */
static std::size_t check_geospatial_vram_collection(u64_t seed, std::int32_t ordinal, observations_kind_t kind,
                                                    std::size_t count, std::size_t queries_count,
                                                    usearch_geospatial_relation_t relation, f64_t radius_metres,
                                                    std::size_t expansion, hills_t const* hills = nullptr) {
    constexpr std::size_t wanted_k = 10, batches_k = 3;
    std::vector<usearch_geospatial_observation_t> observations = generate_observations(kind, count, seed);
    std::vector<usearch_geospatial_observation_t> queries = generate_observations(kind, queries_count, ~seed);
    // Over hills, every other camera stands on the ground and the rest hover over every hill.
    for (std::size_t row = 1; hills && row < count; row += 2) observations[row].altitude_metres = 60 + row % 31;
    for (std::size_t row = 1; hills && row < queries_count; row += 2) queries[row].altitude_metres = 60 + row % 31;
    usearch_geospatial_terrain_t const terrain = hills ? hills->borrowed() : usearch_geospatial_terrain_t {};
    std::vector<usearch_key_t> keys(count);
    for (std::size_t row = 0; row != count; ++row) keys[row] = us::tools::key_of_row(seed, row);
    usearch_geospatial_tolerance_t const tolerance {radius_metres, 0};
    std::size_t failures = 0;

    // The exact admitted chords of every query, ascending.
    std::vector<earth_observation_t> stored(count);
    for (std::size_t row = 0; row != count; ++row) stored[row] = encoded(observations[row], hills);
    std::vector<std::vector<f64_t>> exact(queries_count);
    for (std::size_t query = 0; query != queries_count; ++query) {
        earth_observation_t const probe = encoded(queries[query], hills);
        for (std::size_t row = 0; row != count; ++row)
            if (related(static_cast<relation_t>(relation), radius_metres, probe, stored[row], hills))
                exact[query].push_back(
                    std::sqrt(static_cast<f64_t>(chord_squared(probe.position, stored[row].position))) /
                    centimetres_per_metre_k);
        std::sort(exact[query].begin(), exact[query].end());
    }

    usearch_outcome_t outcome {};
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena;
    arena_init.outcome = &outcome;
    usearch_arena_init(&arena_init);
    failures += expect(!outcome.message);

    // Builds and searches one handle, the GPU's when an executor is given, landing on host pages.
    auto const build_and_search = [&](usearch_geospatial_index_t index, usearch_executor_t executor) {
        handle_matches_t matched;
        for (std::size_t batch = 0; batch != batches_k && !failures; ++batch) {
            std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
            usearch_geospatial_insert_t insert {};
            insert.index = index;
            insert.executor = executor;
            insert.outcome = &outcome;
            insert.count = bound - first;
            insert.keys = {keys.data() + first, sizeof(usearch_key_t)};
            insert.observations = observations.data() + first;
            usearch_geospatial_insert(&insert);
            failures += expect(!outcome.message);
        }
        usearch_matches_view_t matches {};
        usearch_geospatial_search_t search {};
        search.index = index;
        search.executor = executor;
        search.outcome = &outcome;
        search.count = queries_count;
        search.queries = queries.data();
        search.relation = relation;
        search.tolerance = tolerance;
        search.matches = &matches;
        search.arena = arena;
        search.wanted = wanted_k;
        search.expansion_factor = expansion;
        if (!failures) usearch_geospatial_search(&search);
        failures += expect(!outcome.message);
        if (failures) return matched;
        matched.keys.assign(matches.keys, matches.keys + queries_count * matches.columns);
        matched.metres.assign(matches.scores, matches.scores + queries_count * matches.columns);
        matched.counts.assign(matches.counts, matches.counts + queries_count);
        return matched;
    };

    // Matches within the tenth exact chord, of ten or of every admitted entry when fewer exist.
    auto const recall = [&](handle_matches_t const& matched) {
        std::size_t hits = 0, possible = 0;
        for (std::size_t query = 0; query != queries_count && !matched.counts.empty(); ++query) {
            std::size_t const expected = min_of(wanted_k, exact[query].size());
            if (!expected) continue;
            f64_t const threshold = exact[query][expected - 1] * (1 + 1e-9) + 1e-6;
            std::size_t found = 0;
            for (std::size_t rank = 0; rank != matched.counts[query]; ++rank)
                found += matched.metres[query * wanted_k + rank] <= threshold;
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
    failures += expect(!outcome.message);
    handle_matches_t const ram_matches = failures ? handle_matches_t {} : build_and_search(ram_index, nullptr);

    std::int32_t const devices[] = {ordinal};
    usearch_vram_options_t const vram {devices, 1};
    usearch_geospatial_index_t vram_index = nullptr;
    usearch_geospatial_init_vram_t vram_init {};
    vram_init.index = &vram_index;
    vram_init.outcome = &outcome;
    vram_init.terrain = terrain;
    vram_init.vram = vram;
    if (!failures) usearch_geospatial_init_vram(&vram_init);
    failures += expect(!outcome.message);
    usearch_executor_t executor = nullptr;
    usearch_executor_for_vram_t placement {};
    placement.executor = &executor;
    placement.outcome = &outcome;
    placement.vram = vram;
    if (!failures) usearch_executor_for_vram(&placement);
    failures += expect(!outcome.message);
    handle_matches_t const vram_matches = failures ? handle_matches_t {} : build_and_search(vram_index, executor);

    for (std::size_t query = 0; query != queries_count && !failures; ++query) {
        earth_observation_t const probe = encoded(queries[query], hills);
        failures += expect_eq(vram_matches.counts[query], min_of(wanted_k, exact[query].size()));
        for (std::size_t rank = 0; rank != vram_matches.counts[query]; ++rank) {
            std::optional<u64_t> const minted = us::tools::row_of_key(seed, vram_matches.keys[query * wanted_k + rank]);
            if (expect(minted && *minted < count)) return failures + 1;
            u64_t const row = *minted;
            f64_t const metres = vram_matches.metres[query * wanted_k + rank];
            f64_t const chord = std::sqrt(static_cast<f64_t>(chord_squared(probe.position, stored[row].position))) /
                                centimetres_per_metre_k;
            failures += expect(related(static_cast<relation_t>(relation), radius_metres, probe, stored[row], hills));
            failures += expect(std::abs(metres - chord) <= 1e-6 * max_of(chord, 1.0));
            failures += expect(
                count > expansion ||
                (rank < exact[query].size() && std::abs(metres - exact[query][rank]) <= 1e-6 * max_of(chord, 1.0)));
            failures += expect(!rank || vram_matches.metres[query * wanted_k + rank - 1] <= metres);
        }
    }
    f64_t const ram_recall = recall(ram_matches), vram_recall = recall(vram_matches);
    if (vram_recall + 0.05 < ram_recall)
        fmt::print(stderr, "  recall at ten over {} observations: {:.3f} on the GPU, {:.3f} in RAM\n", count,
                   vram_recall, ram_recall);
    failures += expect(vram_recall + 0.05 >= ram_recall);

    usearch_geospatial_free(vram_index);
    usearch_geospatial_free(ram_index);
    usearch_executor_free(executor);
    usearch_arena_free(arena);
    return failures;
}

std::size_t check_geospatial_vram(settings_t const& settings, gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::nvidia_k || !settings.filter.selects("geospatial_vram")) return 0;
    // The C interface names devices by their place in the listing, which `device` belongs to.
    std::int32_t const ordinal = static_cast<std::int32_t>(&device - test::gpu_devices().data());
    std::size_t failures = 0;
    failures += report(
        "geospatial_vram global",
        check_geospatial_vram_collection(stream_key(settings.seed, "global"), ordinal, observations_kind_t::global_k,
                                         4096, 128, usearch_geospatial_relation_near_k, 0, 64));
    failures += report(
        "geospatial_vram city",
        check_geospatial_vram_collection(stream_key(settings.seed, "city"), ordinal, observations_kind_t::city_k, 4096,
                                         128, usearch_geospatial_relation_near_k, 0, 64));
    failures += report(
        "geospatial_vram sees",
        check_geospatial_vram_collection(stream_key(settings.seed, "sees"), ordinal, observations_kind_t::cameras_k,
                                         4096, 128, usearch_geospatial_relation_sees_k, 300, 256));
    failures += report(
        "geospatial_vram seen",
        check_geospatial_vram_collection(stream_key(settings.seed, "seen"), ordinal, observations_kind_t::cameras_k,
                                         4096, 128, usearch_geospatial_relation_seen_k, 300, 256));
    hills_t const hills;
    failures += report(
        "geospatial_vram visible over hills",
        check_geospatial_vram_collection(stream_key(settings.seed, "visible"), ordinal, observations_kind_t::cameras_k,
                                         256, 128, usearch_geospatial_relation_visible_k, 0, 256, &hills));
    failures += report("geospatial_vram sees over hills",
                       check_geospatial_vram_collection(stream_key(settings.seed, "sees hills"), ordinal,
                                                        observations_kind_t::cameras_k, 256, 128,
                                                        usearch_geospatial_relation_sees_k, 300, 256, &hills));
    failures += report("geospatial_vram seen over hills",
                       check_geospatial_vram_collection(stream_key(settings.seed, "seen hills"), ordinal,
                                                        observations_kind_t::cameras_k, 256, 128,
                                                        usearch_geospatial_relation_seen_k, 300, 256, &hills));
    return failures;
}

#else

std::size_t check_geospatial_vram(settings_t const&, gpu_device_t const&) { return 0; }

#endif

} // namespace unum::usearch::test::geospatial
