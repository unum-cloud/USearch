/**
 *  @file test/geospatial/ram_index.cpp
 *  @author Ash Vardanian
 *  @date September 3, 2026
 *  @brief The geospatial suite: observations as integer Earth-fixed centimetres under the
 *      squared-chord metric, through the basic scenarios and SafeGraphs.
 *
 *  Then the encoding, the relations, the geometry, the octant selector, the long-link draws and the
 *  radius walk, each against brute force, and last a saved index served from storage.
 *
 *  Values are drawn inside a one-degree box around London and converted to Earth-fixed centimetres,
 *  so the corpus is a dense patch of a curved surface where the tangent-plane cones are exact in
 *  bearing. Distances are integers, so every oracle comparison is exact.
 */
#include <cmath>   // `std::abs`, `std::sqrt`, `std::isnan`
#include <cstddef> // `std::ptrdiff_t` - the ledger's balance
#include <cstdlib> // `std::abs` over integers

#include <array>  // `std::array` - the handmade triple
#include <bit>    // `std::bit_cast` - the keys
#include <limits> // `std::numeric_limits` - the missing-winner sentinel
#include <memory> // `std::unique_ptr`
#include <new>    // `std::nothrow`
#include <vector> // `std::vector`

#include "usearch/geospatial/ram_metric.hpp"

#include "basic.hpp"

namespace unum::usearch::test::geospatial {
using namespace unum::usearch::geospatial;

/** Candidates one metric round may stage: past every round the scenarios open. */
constexpr std::size_t geospatial_staging_limit_k = 4096;

/** The squared chord in plain double arithmetic - the independent oracle. */
static i64_t scalar_chord_squared(earth_fixed_t first, earth_fixed_t second) noexcept {
    f64_t const delta_x = static_cast<f64_t>(first.x_centimetres) - static_cast<f64_t>(second.x_centimetres);
    f64_t const delta_y = static_cast<f64_t>(first.y_centimetres) - static_cast<f64_t>(second.y_centimetres);
    f64_t const delta_z = static_cast<f64_t>(first.z_centimetres) - static_cast<f64_t>(second.z_centimetres);
    return static_cast<i64_t>(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
}

/** The metric every check scores with: serial squares, staging rounds wider than any scenario
 *  walks. */
static chord_metric_t make_chord_metric() {
    chord_metric_t metric;
    [[maybe_unused]] status_t const sized = metric.init(geospatial_staging_limit_k);
    return metric;
}

/** An observation of nothing but @p position. */
static earth_observation_t observed_at(earth_fixed_t position) noexcept {
    earth_observation_t observation;
    observation.position = position;
    return observation;
}

/** Three positions on two axes, so every pairwise squared chord is a Pythagorean integer. */
struct positions_handmade_t {
    std::array<earth_observation_t, 3> values;
    std::array<std::array<i64_t, 3>, 3> distances;
};

struct positions_fixture_t {
    using index_t =
        geospatial::ram_index<us::utilities::uuid_t, flaky_waiter_t, standard_atomic_ref, counting_allocator_t>;
    using value_t = typename index_t::value_t;
    using allocator_t = typename index_t::allocator_t;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static chord_metric_t make_metric() { return make_chord_metric(); }
    static earth_observation_codec_t make_codec() { return {}; }
    static us::utilities::uuid_t make_key(std::size_t ordinal) {
        return std::bit_cast<us::utilities::uuid_t>(
            std::array<u64_t, 2> {mix_bits(ordinal ^ 0x6E0ull), 0x6E05ull ^ ordinal});
    }
    static i64_t oracle(earth_observation_t const& query, earth_observation_t const& stored) noexcept {
        return scalar_chord_squared(query.position, stored.position);
    }
    static bool value_equals(earth_observation_t const& first, earth_observation_t const& second) noexcept {
        return first == second;
    }

    static positions_handmade_t make_handmade(allocator_t const&) {
        return {{observed_at({0, 0, 0}), observed_at({300, 0, 0}), observed_at({0, 400, 0})},
                {{{0, 90000, 160000}, {90000, 0, 250000}, {160000, 250000, 0}}}};
    }

    /** Uniform geodetic draws inside a one-degree box around London, at the surface. */
    static std::vector<earth_observation_t> make_values(std::size_t count, u64_t seed, allocator_t const&) {
        constexpr f64_t centre_latitude_k = 51.5, centre_longitude_k = -0.1, half_width_k = 0.5;
        splitmix64_t random_generator {seed};
        std::vector<earth_observation_t> values;
        for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
            f64_t const latitude = centre_latitude_k + (2.0 * uniform_unit(random_generator) - 1.0) * half_width_k;
            f64_t const longitude = centre_longitude_k + (2.0 * uniform_unit(random_generator) - 1.0) * half_width_k;
            values.push_back(observed_at(to_earth_fixed({latitude, longitude, 0.0})));
        }
        return values;
    }
};

static_assert(us::basic::value<earth_observation_t>);
static_assert(us::basic::metric<chord_metric_t, earth_observation_t, i64_t>);
static_assert(us::basic::transparent_metric<chord_metric_t>);
static_assert(us::basic::query<geodetic_t, chord_metric_t, earth_observation_t>);
static_assert(us::basic::value_codec<earth_observation_codec_t, earth_observation_t>);

/** The datum: known Earth-fixed images of the equator and the pole, and a round trip
 *  through both conversions within a centimetre and a ten-millionth of a degree. */
static std::size_t check_positions_conversion() {
    std::size_t failures = 0;
    earth_fixed_t const equator = to_earth_fixed({0.0, 0.0, 0.0});
    failures += expect_eq(equator.x_centimetres, 637813700);
    failures += expect_eq(equator.y_centimetres, 0);
    failures += expect_eq(equator.z_centimetres, 0);
    earth_fixed_t const pole = to_earth_fixed({90.0, 0.0, 0.0});
    failures += expect(std::abs(pole.x_centimetres) <= 1);
    failures += expect_eq(pole.z_centimetres, 635675231);

    std::array<geodetic_t, 4> const samples {geodetic_t {51.5, -0.1, 35.0}, geodetic_t {-33.9, 151.2, 0.0},
                                             geodetic_t {89.9, 45.0, 1000.0}, geodetic_t {0.5, 179.99, -50.0}};
    for (geodetic_t const& sample : samples) {
        geodetic_t const returned = to_geodetic(to_earth_fixed(sample));
        failures += expect(std::abs(returned.latitude_degrees - sample.latitude_degrees) < 1e-6);
        failures += expect(std::abs(returned.longitude_degrees - sample.longitude_degrees) < 1e-6);
        failures += expect(std::abs(returned.altitude_metres - sample.altitude_metres) < 0.02);
    }
    return failures;
}

/** The frame at any position is orthonormal at the quantization scale, and displacing
 *  along one of its axes lands in the cone that axis names. */
static std::size_t check_positions_frame() {
    std::size_t failures = 0;
    std::array<earth_fixed_t, 4> const bases {to_earth_fixed({51.5, -0.1, 0.0}), to_earth_fixed({-33.9, 151.2, 0.0}),
                                              to_earth_fixed({90.0, 0.0, 0.0}), to_earth_fixed({0.0, 0.0, 0.0})};
    for (earth_fixed_t const& base : bases) {
        tangent_frame_t const frame = tangent_frame(base);
        auto const dot = [](std::array<i32_t, 3> const& first, std::array<i32_t, 3> const& second) noexcept {
            return static_cast<i64_t>(first[0]) * second[0] + static_cast<i64_t>(first[1]) * second[1] +
                   static_cast<i64_t>(first[2]) * second[2];
        };
        i64_t const unit_squared = frame_scale_k * frame_scale_k;
        failures += expect(std::abs(dot(frame.east, frame.east) - unit_squared) < unit_squared / 1000);
        failures += expect(std::abs(dot(frame.north, frame.north) - unit_squared) < unit_squared / 1000);
        failures += expect(std::abs(dot(frame.east, frame.north)) < unit_squared / 1000);
        std::array<i32_t, 3> const position {base.x_centimetres, base.y_centimetres, base.z_centimetres};
        i64_t const position_squared = dot(position, position);
        failures += expect(std::abs(dot(frame.east, position)) * 1000 < position_squared);
        failures += expect(std::abs(dot(frame.north, position)) * 1000 < position_squared);

        // Steps of a few metres, each well inside one cone: the bits are north sign, east sign,
        // north dominates.
        auto const stepped = [&](i64_t east_units, i64_t north_units) noexcept {
            constexpr i64_t unit_centimetres_k = 250;
            auto const along = [&](std::size_t axis) noexcept {
                i64_t const displacement = east_units * frame.east[axis] + north_units * frame.north[axis];
                return static_cast<i32_t>(displacement * unit_centimetres_k / frame_scale_k);
            };
            return earth_fixed_t {base.x_centimetres + along(0), base.y_centimetres + along(1),
                                  base.z_centimetres + along(2)};
        };
        failures += expect_eq(octant_of(frame, base, stepped(4, 1)), 0u);
        failures += expect_eq(octant_of(frame, base, stepped(1, 4)), 1u);
        failures += expect_eq(octant_of(frame, base, stepped(-4, 1)), 2u);
        failures += expect_eq(octant_of(frame, base, stepped(-1, 4)), 3u);
        failures += expect_eq(octant_of(frame, base, stepped(4, -1)), 4u);
        failures += expect_eq(octant_of(frame, base, stepped(1, -4)), 5u);
        failures += expect_eq(octant_of(frame, base, stepped(-4, -1)), 6u);
        failures += expect_eq(octant_of(frame, base, stepped(-1, -4)), 7u);
    }
    return failures;
}

/** Brute-force cone winners around @p base over every other node, ascending by chord. */
static std::array<std::size_t, surface_keep_octants_t::cones_k> //
brute_force_cone_winners(std::vector<earth_observation_t> const& values, std::size_t base) {
    constexpr std::size_t missing_k = std::numeric_limits<std::size_t>::max();
    tangent_frame_t const frame = tangent_frame(values[base].position);
    std::array<std::size_t, surface_keep_octants_t::cones_k> winners;
    winners.fill(missing_k);
    for (std::size_t candidate = 0; candidate != values.size(); ++candidate) {
        if (candidate == base) continue;
        unsigned const cone = octant_of(frame, values[base].position, values[candidate].position);
        if (winners[cone] == missing_k || chord_squared(values[base].position, values[candidate].position) <
                                              chord_squared(values[base].position, values[winners[cone]].position))
            winners[cone] = candidate;
    }
    return winners;
}

/** The selector over a full pool keeps exactly the brute-force cone winners, then the
 *  longest of the rest up to its budget, and emits them ascending. */
static std::size_t check_positions_selector(u64_t seed) {
    constexpr std::size_t nodes_k = 96;
    constexpr std::size_t probes_k = 8;
    using index_t = positions_fixture_t::index_t;
    using context_t = index_t::traversal_context_t;
    using scored_node_t = index_t::scored_node_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = positions_fixture_t::make_index(ledger);
        index_t& index = *index_ptr;
        chord_metric_t metric = make_chord_metric();
        context_t context {stream_key(seed, "selector"), counting_allocator_t {ledger}};
        auto values = positions_fixture_t::make_values(nodes_k, stream_key(seed, "selector"),
                                                       counting_allocator_t {ledger});
        auto const stored = values;
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<index_t::node_id_t> assigned_ids;
        failures += basic_build<positions_fixture_t>(index, metric, values, assigned_ids, context,
                                                     us::basic::keep_closest_t {basic_connectivity_k});

        surface_keep_octants_t const selector;
        for (std::size_t probe = 0; probe != probes_k; ++probe) {
            std::size_t const base = probe * (nodes_k / probes_k);
            // The pool is every other node, ascending by chord - what a wide enough walk would
            // gather.
            failures += expect(succeeded(context.results.reserve(nodes_k)));
            failures += expect(succeeded(context.selected_neighbors.reserve(selector.connectivity)));
            context.results.clear();
            for (std::size_t candidate = 0; candidate != nodes_k; ++candidate)
                if (candidate != base)
                    context.results.insert_reserved(scored_node_t {
                        assigned_ids[candidate], chord_squared(stored[base].position, stored[candidate].position)});
            metric.reset(stored[base]);
            selector(index, metric, context);

            auto const winners = brute_force_cone_winners(stored, base);
            std::size_t winners_count = 0;
            for (std::size_t cone = 0; cone != winners.size(); ++cone)
                winners_count += winners[cone] != std::numeric_limits<std::size_t>::max();
            failures += expect_eq(context.selected_neighbors.size(), selector.connectivity);
            for (std::size_t cone = 0; cone != winners.size(); ++cone) {
                if (winners[cone] == std::numeric_limits<std::size_t>::max()) continue;
                bool kept = false;
                for (std::size_t slot = 0; slot != context.selected_neighbors.size() && !kept; ++slot)
                    kept = context.selected_neighbors[slot].node_id == assigned_ids[winners[cone]];
                failures += expect(kept);
            }
            // The retained slots hold the pool's longest non-winners, so the last one is the pool's
            // farthest.
            scored_node_t const farthest = context.results[context.results.size() - 1];
            scored_node_t const last_kept = context.selected_neighbors[context.selected_neighbors.size() - 1];
            failures += expect_eq(last_kept.node_id, farthest.node_id);
            for (std::size_t slot = 1; slot != context.selected_neighbors.size(); ++slot)
                failures += expect(context.selected_neighbors[slot - 1].distance <=
                                   context.selected_neighbors[slot].distance);
            failures += expect(winners_count <= selector.connectivity);
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), std::ptrdiff_t {0});
    return failures;
}

/** Long-link draws add published, distinct out-edges to a node without touching its
 *  cone edges, and a lone node draws nothing. */
static std::size_t check_positions_long_links(u64_t seed) {
    constexpr std::size_t nodes_k = 128;
    constexpr std::size_t draws_k = 4;
    using index_t = positions_fixture_t::index_t;
    using context_t = index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = positions_fixture_t::make_index(ledger);
        index_t& index = *index_ptr;
        chord_metric_t metric = make_chord_metric();
        context_t context {stream_key(seed, "long links"), counting_allocator_t {ledger}};
        auto values = positions_fixture_t::make_values(nodes_k, stream_key(seed, "long links"),
                                                       counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<index_t::node_id_t> assigned_ids;
        failures += basic_build<positions_fixture_t>(index, metric, values, assigned_ids, context,
                                                     surface_keep_octants_t {});

        splitmix64_t generator {stream_key(seed, "lone node")};
        for (std::size_t probe = 0; probe != nodes_k; probe += nodes_k / 8) {
            index_t::node_id_t const node = assigned_ids[probe];
            auto const before = index.degree_of(node);
            failures += expect(bool(before));
            failures += expect_eq(draw_long_links(index, node, draws_k, metric, context, generator),
                                  status_t::success_k);
            auto const after = index.degree_of(node);
            failures += expect(bool(after));
            failures += expect(after.value >= before.value);
            failures += expect(after.value <= before.value + draws_k);
            us::basic::ram_cursor<index_t> cursor {index, counting_allocator_t {ledger}};
            failures += expect_eq(cursor.seek(node), status_t::success_k);
            for (index_t::node_id_t const neighbor : cursor.neighbors()) {
                failures += expect(neighbor != node);
                failures += expect(static_cast<std::size_t>(neighbor) < index.size());
            }
        }
    }
    {
        std::unique_ptr<index_t> const index_ptr = positions_fixture_t::make_index(ledger);
        index_t& index = *index_ptr;
        chord_metric_t metric = make_chord_metric();
        context_t context {stream_key(seed, "lone node"), counting_allocator_t {ledger}};
        auto values = positions_fixture_t::make_values(1, stream_key(seed, "lone node"), counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(1)));
        std::vector<index_t::node_id_t> assigned_ids;
        failures += basic_build<positions_fixture_t>(index, metric, values, assigned_ids, context,
                                                     surface_keep_octants_t {});
        splitmix64_t generator {stream_key(seed, "lone node")};
        failures += expect_eq(draw_long_links(index, assigned_ids[0], draws_k, metric, context, generator),
                              status_t::success_k);
        failures += expect_eq(index.degree_of(assigned_ids[0]).value, 0u);
    }
    failures += expect_eq(ledger.live_bytes.load(), std::ptrdiff_t {0});
    return failures;
}

/** The radius walk under the spanner margin matches brute force exactly on a small
 *  corpus, and the unwidened walk never over-reports. */
static std::size_t check_positions_radius(u64_t seed) {
    constexpr std::size_t nodes_k = 256;
    constexpr std::size_t probes_k = 6;
    using index_t = positions_fixture_t::index_t;
    using context_t = index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = positions_fixture_t::make_index(ledger);
        index_t& index = *index_ptr;
        chord_metric_t metric = make_chord_metric();
        context_t context {stream_key(seed, "radius"), counting_allocator_t {ledger}};
        auto values = positions_fixture_t::make_values(nodes_k, stream_key(seed, "radius"),
                                                       counting_allocator_t {ledger});
        auto const stored = values;
        failures += expect(bool(index.reserve(nodes_k)));
        std::vector<index_t::node_id_t> assigned_ids;
        failures += basic_build<positions_fixture_t>(index, metric, values, assigned_ids, context,
                                                     surface_keep_octants_t {});

        std::size_t const widest_row = surface_keep_octants_t {}.connectivity;
        std::array<i64_t, 3> const radii_centimetres {200'000, 800'000, 2'500'000}; // 2 km, 8 km, 25 km.
        for (std::size_t probe = 0; probe != probes_k; ++probe) {
            earth_fixed_t const query = stored[probe * (nodes_k / probes_k)].position;
            for (i64_t const radius : radii_centimetres) {
                std::size_t expected = 0;
                for (earth_observation_t const& candidate : stored)
                    expected += chord_squared(query, candidate.position) <= radius * radius;
                radius_result_t const complete = within(index, query, radius, radius_margin_t::spanner_k, widest_row,
                                                        metric, context);
                failures += expect(bool(complete));
                failures += expect_eq(complete.count, expected);
                failures += expect(complete.pool >= complete.count);
                for (std::size_t position = 0; position != complete.count; ++position)
                    failures += expect(context.results[position].distance <= radius * radius);
                radius_result_t const plain = within(index, query, radius, radius_margin_t::none_k, widest_row, metric,
                                                     context);
                failures += expect(bool(plain));
                failures += expect(plain.count <= expected);
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), std::ptrdiff_t {0});
    return failures;
}

/** An observation decodes within the rounding it was stored at, every rounding outward, and NaN
 *  fields come back as the defaults that only widen tests. */
static std::size_t check_observations_round_trip() {
    std::size_t failures = 0;
    geodetic_observation_t full;
    full.latitude_degrees = 51.5, full.longitude_degrees = -0.12, full.altitude_metres = 35.0;
    full.accuracy_metres = 4.2, full.heading_degrees = 271.3, full.pitch_degrees = -12.0;
    full.heading_accuracy_degrees = 2.0, full.field_of_view_degrees = 69.0, full.range_metres = 180.4;
    full.begin_seconds = 1'760'000'000.1234, full.end_seconds = 1'760'000'020.5;
    earth_observation_t const stored = to_earth_observation(full);
    geodetic_observation_t const returned = to_geodetic_observation(stored);
    failures += expect(std::abs(returned.latitude_degrees - full.latitude_degrees) < 1e-6);
    failures += expect(std::abs(returned.longitude_degrees - full.longitude_degrees) < 1e-6);
    failures += expect(std::abs(returned.altitude_metres - full.altitude_metres) < 0.02);
    failures += expect_eq(returned.accuracy_metres, 5.0);
    failures += expect_eq(returned.range_metres, 181.0);
    failures += expect(std::abs(returned.heading_degrees - full.heading_degrees) < 0.01);
    failures += expect(std::abs(returned.pitch_degrees - full.pitch_degrees) <= 0.71);
    // The field widens by the heading's three deviations each side and one step of rounding, and
    // rounds up.
    failures += expect(returned.field_of_view_degrees >= 69.0 + 12.0 + 1.40625);
    failures += expect(returned.field_of_view_degrees < 69.0 + 12.0 + 2 * 1.40625 + 1e-9);
    failures += expect(returned.begin_seconds <= full.begin_seconds && returned.end_seconds >= full.end_seconds);
    failures += expect(returned.end_seconds - returned.begin_seconds < 20.5);

    geodetic_observation_t bare;
    bare.latitude_degrees = -33.9, bare.longitude_degrees = 151.2;
    geodetic_observation_t const vague = to_geodetic_observation(to_earth_observation(bare, 12.0));
    failures += expect(std::abs(vague.altitude_metres - 12.0) < 0.02);
    failures += expect_eq(vague.accuracy_metres, 0.0);
    failures += expect(std::isnan(vague.heading_degrees) && std::isnan(vague.field_of_view_degrees));
    failures += expect(std::isnan(vague.pitch_degrees) && std::isnan(vague.range_metres));
    failures += expect(std::isnan(vague.begin_seconds) && std::isnan(vague.end_seconds));
    return failures;
}

/** Each relation against hand-placed observations: radius widening by both deviations, time slack,
 *  a camera facing east that sees the east and not the west, and a view's range. */
static std::size_t check_observations_admits() {
    std::size_t failures = 0;
    auto at = [](f64_t latitude, f64_t longitude) {
        geodetic_observation_t observation;
        observation.latitude_degrees = latitude, observation.longitude_degrees = longitude;
        observation.altitude_metres = 2.0;
        return observation;
    };
    // A degree of longitude at 51.5° is about 69.4 km, so 0.001° is about 69 m.
    geodetic_observation_t query = at(51.5, 0.0), east = at(51.5, 0.001), west = at(51.5, -0.001);
    earth_observation_t const stored_query = to_earth_observation(query);
    earth_observation_t const stored_east = to_earth_observation(east);
    failures += expect(admits(relation_t::near_k, 70.0, 0, stored_query, stored_east));
    failures += expect(!admits(relation_t::near_k, 60.0, 0, stored_query, stored_east));
    east.accuracy_metres = 4.0;
    failures += expect(admits(relation_t::near_k, 60.0, 0, stored_query, to_earth_observation(east)));
    failures += expect(admits(relation_t::near_k, 0.0, 0, stored_query, stored_east));

    // Intervals: [100 s, 110 s] against [115 s, 120 s] overlap only with 5 s of slack.
    query.begin_seconds = 100, query.end_seconds = 110;
    geodetic_observation_t later = at(51.5, 0.0);
    later.begin_seconds = 115, later.end_seconds = 120;
    earth_observation_t const timed_query = to_earth_observation(query), timed_later = to_earth_observation(later);
    failures += expect(!admits(relation_t::near_k, 10.0, 4'000, timed_query, timed_later));
    failures += expect(admits(relation_t::near_k, 10.0, 5'000, timed_query, timed_later));
    failures += expect(admits(relation_t::near_k, 10.0, 0, stored_query, timed_later));

    // A level camera facing east with a 60° field.
    geodetic_observation_t camera = at(51.5, 0.0);
    camera.heading_degrees = 90.0, camera.pitch_degrees = 0.0, camera.field_of_view_degrees = 60.0;
    earth_observation_t const stored_camera = to_earth_observation(camera);
    earth_observation_t const stored_west = to_earth_observation(west);
    failures += expect(admits(relation_t::seen_k, 100.0, 0, stored_camera, stored_east));
    failures += expect(!admits(relation_t::seen_k, 100.0, 0, stored_camera, stored_west));
    failures += expect(admits(relation_t::sees_k, 100.0, 0, stored_east, stored_camera));
    failures += expect(!admits(relation_t::sees_k, 100.0, 0, stored_west, stored_camera));
    camera.range_metres = 50.0;
    failures += expect(!admits(relation_t::seen_k, 100.0, 0, to_earth_observation(camera), stored_east));
    return failures;
}

/** A saved monolith read back from storage, every observation a fixed 32-byte row the view reads
 *  without a directory and decodes in place. */
static std::size_t check_positions_nvme(u64_t seed) {
    using scratch_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using metric_t = chord_metric<i32_serial_geospatial_kit_t, scratch_t>;
    using nvme_index_t =
        us::basic::nvme_index<earth_observation_codec_t, us::utilities::uuid_t, u32_t, i64_t, blocking_files_engine<>>;
    allocation_ledger_t ledger;
    std::vector<earth_observation_t> const values = positions_fixture_t::make_values(64, stream_key(seed, "nvme"),
                                                                                     counting_allocator_t {ledger});
    auto make_value = [&](std::size_t entry, counting_allocator_t const&) { return values[entry]; };
    auto make_metric = [](scratch_t const& allocator, std::size_t staging) noexcept {
        metric_t metric {allocator};
        status_t const ready = metric.init(staging);
        return expected<metric_t> {std::move(metric), ready};
    };
    auto query_of = [&](std::size_t row, scratch_t const&) noexcept {
        return expected<earth_observation_t> {values[row], status_t::success_k};
    };
    auto matches = [&](std::size_t entry, earth_observation_t const& view) noexcept { return view == values[entry]; };
    return test_basic_nvme_view<positions_fixture_t, nvme_index_t>(seed, values.size(), "chord", make_value,
                                                                   make_metric, query_of, matches);
}

std::size_t check_geospatial(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_scenarios<positions_fixture_t>(settings.seed);
    failures += test_basic_safegraphs_roundtrip<positions_fixture_t>(settings.seed, "uuid", "chord");
    failures += test_basic_safegraphs_hostile<positions_fixture_t>(settings.seed, "chord");
    failures += check_observations_round_trip();
    failures += check_observations_admits();
    failures += check_positions_conversion();
    failures += check_positions_frame();
    failures += check_positions_selector(settings.seed);
    failures += check_positions_long_links(settings.seed);
    failures += check_positions_radius(settings.seed);
    failures += check_positions_nvme(settings.seed);
    return report("geospatial", failures);
}

} // namespace unum::usearch::test::geospatial
