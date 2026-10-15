/**
 *  @file test/shapes/ram_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The shapes suite: the basic scenarios over packed 3D clouds, then the kits, bounds, codec
 *      and packing.
 *
 *  Clouds are eight planted prototypes of 12 to 16 points, each drawn with a few jittered points,
 *  one in four mirrored, and under a random rigid motion, so distances cluster by prototype while
 *  raw coordinates do not. Every cloud lands in [8, 16) on each axis, where one storage ulp exceeds
 *  the rounding of the packed coordinates twice over, so unpacking returns the inserted pose
 *  exactly, a reload packs the same bytes, and the scenarios' checks of identical answers after a
 *  reload hold in @c f16 as well as @c f32. The oracle is the serial kit's 1×1 call.
 */
#include <cmath>   // `std::sqrt`, `std::fabs`, `std::fmax`
#include <cstring> // `std::memcmp` - unpacked poses and packs compared as bytes

#include <array>       // `std::array` - points, handmade triples, the kernel rounds
#include <bit>         // `std::bit_cast` - the keys
#include <limits>      // `std::numeric_limits` - the oracle's failure sentinel
#include <memory>      // `std::unique_ptr`
#include <new>         // `std::nothrow`
#include <span>        // `std::span`
#include <type_traits> // `std::is_same_v` - the storage precision
#include <vector>      // `std::vector`

#include "usearch/shapes/ram_index.hpp"

#include "basic.hpp"

namespace unum::usearch::test::shapes {
using namespace unum::usearch::shapes;

/** Points in the smallest generated cloud. */
constexpr std::size_t shapes_min_points_k = 12;

/** Points in the largest generated cloud, which sizes every stack buffer below. */
constexpr std::size_t shapes_max_points_k = 16;

/** Candidates one round may stage: past every round the scenarios open, whole-graph searches over
 *  three node chunks included, so no walk saturates on the metric. */
constexpr std::size_t shapes_staging_limit_k = 4096;

/** Bytes of the oracle's workspace, enough for Gromov-Wasserstein between two of the largest clouds
 *  in @c f64. */
constexpr std::size_t shapes_oracle_workspace_k = 32 * 1024;

/** The four shape kinds, each checked through every fixture's kit. */
constexpr std::array<nk_kernel_kind_t, 4> shapes_kernels_k {nk_kernel_chamfers_packed_k, nk_kernel_hausdorffs_packed_k,
                                                            nk_kernel_wassersteinprocrustes_packed_k,
                                                            nk_kernel_gromovwassersteins_packed_k};

/** One storage rounding at @p magnitude: relative 1e-3 in @c f16 and 1e-6 in @c f32, above an ulp
 *  of either. */
template <typename scalar_type_>
constexpr f64_t storage_precision(f64_t magnitude) noexcept {
    return magnitude * (std::is_same_v<scalar_type_, nk::f16_t> ? 1e-3 : 1e-6);
}

/** The largest coordinate magnitude of @p cloud, the scale its storage rounds at. */
template <typename scalar_type_>
f64_t cloud_extent(std::vector<scalar_type_> const& cloud) noexcept {
    f64_t extent = 0;
    for (scalar_type_ const coordinate : cloud) extent = std::fmax(extent, std::fabs(f64_t(float(coordinate))));
    return extent;
}

static std::array<f64_t, 3> cross_product(std::array<f64_t, 3> const& first,
                                          std::array<f64_t, 3> const& second) noexcept {
    return {first[1] * second[2] - first[2] * second[1], first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0]};
}

/** Rotates @p point by the unit quaternion @p rotation, scalar part first, as
 *  v + 2w(u × v) + 2u × (u × v). */
static std::array<f64_t, 3> rotate_point(std::array<f64_t, 3> const& point,
                                         std::array<f64_t, 4> const& rotation) noexcept {
    std::array<f64_t, 3> const axis {rotation[1], rotation[2], rotation[3]};
    std::array<f64_t, 3> const twist = cross_product(axis, point);
    std::array<f64_t, 3> const turn = cross_product(axis, twist);
    std::array<f64_t, 3> rotated;
    for (std::size_t dimension = 0; dimension != 3; ++dimension)
        rotated[dimension] = point[dimension] + 2 * (rotation[0] * twist[dimension] + turn[dimension]);
    return rotated;
}

/** Draws @p count clouds of @b [points,3] interleaved coordinates in @p scalar_type_, as the file's
 *  docblock says. */
template <typename scalar_type_>
std::vector<std::vector<scalar_type_>> make_clouds(std::size_t count, u64_t seed) {
    constexpr std::size_t prototypes_k = 8;
    constexpr std::size_t jittered_points_k = 4;
    splitmix64_t random_generator {seed};
    uniform_between<f64_t> const centered {-1, 1};
    uniform_below<u32_t> const prototype_distribution {prototypes_k};
    uniform_below<u32_t> const extra_points_distribution {shapes_max_points_k - shapes_min_points_k + 1};
    uniform_below<u32_t> const mirror_distribution {4};

    std::array<std::vector<std::array<f64_t, 3>>, prototypes_k> prototypes;
    for (std::vector<std::array<f64_t, 3>>& prototype : prototypes) {
        prototype.resize(shapes_min_points_k + extra_points_distribution(random_generator));
        for (std::array<f64_t, 3>& point : prototype)
            point = {centered(random_generator), centered(random_generator), centered(random_generator)};
    }

    std::vector<std::vector<scalar_type_>> clouds(count);
    for (std::vector<scalar_type_>& cloud : clouds) {
        std::vector<std::array<f64_t, 3>> points = prototypes[prototype_distribution(random_generator)];
        uniform_below<u32_t> const point_distribution {static_cast<u32_t>(points.size())};
        for (std::size_t jitter = 0; jitter != jittered_points_k; ++jitter)
            for (f64_t& coordinate : points[point_distribution(random_generator)])
                coordinate += 0.15 * centered(random_generator);
        f64_t const mirror = mirror_distribution(random_generator) == 0 ? -1 : 1;
        std::array<f64_t, 4> rotation {centered(random_generator), centered(random_generator),
                                       centered(random_generator), centered(random_generator)};
        f64_t const norm = std::sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2] +
                                     rotation[3] * rotation[3]);
        if (norm > 0)
            for (f64_t& component : rotation) component /= norm;
        else rotation = {1, 0, 0, 0};
        // Prototype points stay within 2 of their origin, so the motion keeps every coordinate in
        // [8.5, 15.5].
        std::array<f64_t, 3> const translation {12 + 1.5 * centered(random_generator),
                                                12 + 1.5 * centered(random_generator),
                                                12 + 1.5 * centered(random_generator)};
        cloud.reserve(points.size() * 3);
        for (std::array<f64_t, 3> point : points) {
            point[0] *= mirror;
            std::array<f64_t, 3> const moved = rotate_point(point, rotation);
            for (std::size_t dimension = 0; dimension != 3; ++dimension)
                cloud.push_back(scalar_type_(moved[dimension] + translation[dimension]));
        }
    }
    return clouds;
}

/** The oracle every metric distance is held to: the serial kit's 1×1 call between two packs,
 *  sharing nothing with the metric's staging, bounds or workspace growth; the largest distance when
 *  the kernel refuses. */
template <typename scalar_type_>
typename shapes_kit<scalar_type_, nk_cap_serial_k>::distance_t serial_distance(nk_kernel_kind_t kernel,
                                                                               void const* first,
                                                                               void const* second) noexcept {
    using kit_t = shapes_kit<scalar_type_, nk_cap_serial_k>;
    using distance_t = typename kit_t::distance_t;
    alignas(nk_default_alignment_k) byte_t workspace[shapes_oracle_workspace_k];
    distance_t distance = 0;
    nk_status_t status = nk_missing_kernel_k;
    switch (kernel) {
    case nk_kernel_chamfers_packed_k:
        status = kit_t::chamfers(&first, 1, &second, 1, &distance, sizeof(distance_t), nullptr, nullptr);
        break;
    case nk_kernel_hausdorffs_packed_k:
        status = kit_t::hausdorffs(&first, 1, &second, 1, &distance, sizeof(distance_t), nullptr, nullptr);
        break;
    case nk_kernel_wassersteinprocrustes_packed_k:
        status = kit_t::wassersteinprocrustes(&first, 1, &second, 1, &distance, sizeof(distance_t), nullptr, workspace,
                                              sizeof(workspace), nullptr);
        break;
    case nk_kernel_gromovwassersteins_packed_k:
        status = kit_t::gromovwassersteins(&first, 1, &second, 1, &distance, sizeof(distance_t), nullptr, workspace,
                                           sizeof(workspace), nullptr);
        break;
    default: break;
    }
    return status == nk_success_k ? distance : (std::numeric_limits<distance_t>::max)();
}

/** Three dumbbells of lengths 1, 2 and 3 whose packs lie along their first axes, so every kind
 *  between two of them is half the gap of their lengths. */
template <typename value_type_, typename distance_type_>
struct shapes_handmade {
    std::array<value_type_, 3> values;
    std::array<std::array<distance_type_, 3>, 3> distances;
};

/** One kit scoring one kind: values packed from @ref make_clouds through the scenario's ledgered
 *  allocator, and the serial kit's call as the oracle. */
template <typename kit_type_, nk_kernel_kind_t kernel_>
struct shapes_fixture {
    using kit_t = kit_type_;
    using scalar_t = typename kit_t::scalar_t;
    using index_t =
        shapes::ram_index<kit_t, us::utilities::uuid_t, flaky_waiter_t, standard_atomic_ref, counting_allocator_t>;
    using value_t = typename index_t::value_t;
    using allocator_t = typename index_t::allocator_t;
    using distance_t = typename index_t::distance_t;
    using metric_t = shapes_metric<kit_t>;
    static constexpr nk_kernel_kind_t kernel_k = kernel_;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric(kernel_, forkunion_allocator_t {});
        [[maybe_unused]] status_t const sized = metric.init(shapes_staging_limit_k);
        return metric;
    }
    static codec<kit_t> make_codec() noexcept { return {}; }
    static us::utilities::uuid_t make_key(std::size_t ordinal) {
        return std::bit_cast<us::utilities::uuid_t>(std::array<u64_t, 2> {mix_bits(ordinal ^ 0x5A5A5Aull), ordinal});
    }
    static distance_t oracle(value_t const& query, value_t const& stored) noexcept {
        return serial_distance<scalar_t>(kernel_, query[0], stored[0]);
    }

    /** Whether two values hold the same cloud, as the codec saves it. */
    static bool value_equals(value_t const& first, value_t const& second) noexcept {
        if (first.max_points() != second.max_points()) return false;
        codec<kit_t> const shapes_codec;
        std::array<scalar_t, shapes_max_points_k * 3> first_points, second_points;
        bool const saved = succeeded(shapes_codec.save(first, std::as_writable_bytes(std::span {first_points}))) &&
                           succeeded(shapes_codec.save(second, std::as_writable_bytes(std::span {second_points})));
        std::size_t const bytes = shapes_codec.storage_requirements(first);
        return saved && std::memcmp(first_points.data(), second_points.data(), bytes) == 0;
    }

    /** Packs @p cloud alone through @p allocator, as the codec's load does. */
    static value_t pack(std::span<scalar_t const> cloud, allocator_t const& allocator) noexcept {
        std::array<std::span<scalar_t const>, 1> const clouds {cloud};
        return value_t::make(clouds, allocator, 0).value;
    }

    static shapes_handmade<value_t, distance_t> make_handmade(allocator_t const& allocator) {
        shapes_handmade<value_t, distance_t> handmade {{}, {{{0, 0.5, 1}, {0.5, 0, 0.5}, {1, 0.5, 0}}}};
        for (std::size_t length = 1; length != 4; ++length) {
            std::array<scalar_t, 6> const dumbbell {0, 0, 0, length, 0, 0};
            handmade.values[length - 1] = pack(dumbbell, allocator);
        }
        return handmade;
    }

    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        std::vector<value_t> values;
        values.reserve(count);
        for (std::vector<scalar_t> const& cloud : make_clouds<scalar_t>(count, seed))
            values.push_back(pack(cloud, allocator));
        return values;
    }
};

using chamfer_fixture_t = shapes_fixture<shapes_kit<nk::f32_t, nk_cap_any_k>, nk_kernel_chamfers_packed_k>;
using procrustes_fixture_t =
    shapes_fixture<shapes_kit<nk::f16_t, nk_cap_any_k>, nk_kernel_wassersteinprocrustes_packed_k>;

static_assert(us::basic::value<chamfer_fixture_t::value_t>);
static_assert(metric<chamfer_fixture_t::metric_t, chamfer_fixture_t::value_t, chamfer_fixture_t::distance_t>);
static_assert(metric<procrustes_fixture_t::metric_t, procrustes_fixture_t::value_t, procrustes_fixture_t::distance_t>);
static_assert(transparent_metric<procrustes_fixture_t::metric_t>);
static_assert(value_codec<codec<chamfer_fixture_t::kit_t>, chamfer_fixture_t::value_t>);
static_assert(value_codec<codec<procrustes_fixture_t::kit_t>, procrustes_fixture_t::value_t>);

/** One round of @p metric: the first pack as the query, the rest staged, drained within @p radius
 *  into @p distances. */
template <typename fixture_type_>
std::size_t score_round(typename fixture_type_::metric_t& metric,
                        std::vector<typename fixture_type_::value_t> const& packs,
                        std::span<typename fixture_type_::distance_t> distances,
                        typename fixture_type_::distance_t radius) {
    std::size_t failures = 0;
    metric.reset(packs[0]);
    for (std::size_t candidate = 0; candidate != distances.size(); ++candidate)
        failures += expect(metric.push(packs[1 + candidate]) == metric_staging_t::staged_k);
    metric.drain(distances, radius);
    failures += expect_eq(metric.status(), status_t::success_k);
    return failures;
}

/** Every kind through the fixture's kit against the serial oracle: exact at an unbounded radius,
 *  never above the oracle at a zero radius, where the bounded kinds report their bounds, and near
 *  zero for the query's mirror image. */
template <typename fixture_type_>
std::size_t check_shapes_kernels(u64_t seed) {
    using scalar_t = typename fixture_type_::scalar_t;
    using value_t = typename fixture_type_::value_t;
    using distance_t = typename fixture_type_::distance_t;
    using metric_t = typename fixture_type_::metric_t;
    constexpr std::size_t candidates_k = 8;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        std::vector<std::vector<scalar_t>> clouds = make_clouds<scalar_t>(candidates_k + 1,
                                                                          stream_key(seed, "kernels"));
        // The first candidate is the query with one coordinate negated, which every scalar holds
        // exactly.
        clouds[1] = clouds[0];
        for (std::size_t point = 0; point != clouds[1].size() / 3; ++point)
            clouds[1][point * 3] = scalar_t(-float(clouds[1][point * 3]));
        f64_t const mirror_tolerance = storage_precision<scalar_t>(cloud_extent(clouds[0]));
        std::vector<value_t> packs;
        for (std::vector<scalar_t> const& cloud : clouds) packs.push_back(fixture_type_::pack(cloud, allocator));

        for (nk_kernel_kind_t const kernel : shapes_kernels_k) {
            metric_t metric(kernel, forkunion_allocator_t {});
            failures += expect_eq(metric.init(candidates_k), status_t::success_k);
            std::array<distance_t, candidates_k> exact, bounded;
            failures += score_round<fixture_type_>(metric, packs, exact, unbounded_radius<distance_t>());
            failures += score_round<fixture_type_>(metric, packs, bounded, distance_t {});
            failures += expect(exact[0] <= mirror_tolerance);
            for (std::size_t candidate = 0; candidate != candidates_k; ++candidate) {
                distance_t const truth = serial_distance<scalar_t>(kernel, packs[0][0], packs[1 + candidate][0]);
                failures += expect_near(exact[candidate], truth);
                failures += expect(bounded[candidate] <= truth + 1e-3 * std::fmax(truth, 1.0));
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** A save returns each cloud's inserted pose within one storage rounding, and a reload, which packs
 *  the saved pose again, scores against the query as before to the storage precision. */
template <typename fixture_type_>
std::size_t check_shapes_codec(u64_t seed) {
    using scalar_t = typename fixture_type_::scalar_t;
    using value_t = typename fixture_type_::value_t;
    using distance_t = typename fixture_type_::distance_t;
    constexpr std::size_t clouds_k = 8;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        auto const shapes_codec = fixture_type_::make_codec();
        std::vector<std::vector<scalar_t>> const clouds = make_clouds<scalar_t>(clouds_k, stream_key(seed, "codec"));
        std::vector<value_t> packs, reloaded;
        f64_t extent = 0;
        for (std::vector<scalar_t> const& cloud : clouds) {
            packs.push_back(fixture_type_::pack(cloud, allocator));
            failures += expect_eq(shapes_codec.storage_requirements(packs.back()), cloud.size() * sizeof(scalar_t));
            std::vector<scalar_t> saved(cloud.size());
            failures += expect_eq(shapes_codec.save(packs.back(), std::as_writable_bytes(std::span {saved})),
                                  status_t::success_k);
            f64_t const tolerance = storage_precision<scalar_t>(cloud_extent(cloud));
            std::size_t drifted = 0;
            for (std::size_t coordinate = 0; coordinate != cloud.size(); ++coordinate)
                drifted += std::fabs(f64_t(float(saved[coordinate])) - f64_t(float(cloud[coordinate]))) > tolerance;
            failures += expect_eq(drifted, 0u);
            reloaded.emplace_back(allocator);
            failures += expect_eq(shapes_codec.load(std::as_bytes(std::span {saved}), reloaded.back()),
                                  status_t::success_k);
            extent = std::fmax(extent, cloud_extent(cloud));
        }
        for (std::size_t candidate = 1; candidate != clouds_k; ++candidate) {
            distance_t const before = serial_distance<scalar_t>(fixture_type_::kernel_k, packs[0][0],
                                                                packs[candidate][0]);
            distance_t const after = serial_distance<scalar_t>(fixture_type_::kernel_k, reloaded[0][0],
                                                               reloaded[candidate][0]);
            failures += expect(std::fabs(f64_t(after) - f64_t(before)) <= storage_precision<scalar_t>(extent));
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** What the index relies on, as inserts, queries and loads all pack through @c make with no
 *  capabilities: one cloud packs into the same bytes every time, and a stored shape scores zero
 *  against a query packed from the same coordinates, exactly under every kind but
 *  Gromov-Wasserstein, whose loss reaches zero only to the fast-math tolerance. */
template <typename fixture_type_>
std::size_t check_shapes_packing(u64_t seed) {
    using scalar_t = typename fixture_type_::scalar_t;
    using value_t = typename fixture_type_::value_t;
    using distance_t = typename fixture_type_::distance_t;
    using metric_t = typename fixture_type_::metric_t;
    using query_t = shapes::value<scalar_t>;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        auto const shapes_codec = fixture_type_::make_codec();
        for (std::vector<scalar_t> const& cloud : make_clouds<scalar_t>(4, stream_key(seed, "packing"))) {
            value_t first(allocator), second(allocator);
            failures += expect_eq(shapes_codec.load(std::as_bytes(std::span {cloud}), first), status_t::success_k);
            failures += expect_eq(shapes_codec.load(std::as_bytes(std::span {cloud}), second), status_t::success_k);
            std::size_t const bytes = nk::shapes_pack_size<scalar_t>(cloud.size() / 3, 0).value;
            failures += expect(std::memcmp(first[0], second[0], bytes) == 0);

            // The query comes from another allocator, as a call builds one in its arena.
            std::array<std::span<scalar_t const>, 1> const clouds {cloud};
            query_t const query = query_t::make(clouds, {}, 0).value;
            for (nk_kernel_kind_t const kernel : shapes_kernels_k) {
                metric_t metric(kernel, forkunion_allocator_t {});
                failures += expect_eq(metric.init(1), status_t::success_k);
                metric.reset(query);
                failures += expect(metric.push(second) == metric_staging_t::staged_k);
                std::array<distance_t, 1> distances {1};
                metric.drain(distances, unbounded_radius<distance_t>());
                failures += expect_eq(metric.status(), status_t::success_k);
                if (kernel == nk_kernel_gromovwassersteins_packed_k) failures += expect_near(distances[0], 0.0);
                else failures += expect_eq(distances[0], distance_t {});
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** A saved monolith read back from storage, every cloud a ragged row of points the view locates
 *  through its directory and each round packs again. */
std::size_t check_shapes_nvme(u64_t seed) {
    using fixture_t = chamfer_fixture_t;
    using scalar_t = fixture_t::scalar_t;
    using scratch_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using metric_t = shapes_metric<fixture_t::kit_t, scratch_t>;
    using query_t = nk::shapes_packed<scalar_t, std::allocator_traits<scratch_t>::rebind_alloc<char>>;
    using nvme_index_t = us::basic::nvme_index<codec<fixture_t::kit_t>, us::utilities::uuid_t, fixture_t::distance_t,
                                               blocking_files_engine<>>;
    std::vector<std::vector<scalar_t>> const clouds = make_clouds<scalar_t>(32, stream_key(seed, "nvme"));

    auto make_cloud = [&](std::size_t entry, counting_allocator_t const& allocator) {
        return fixture_t::pack(clouds[entry], allocator);
    };
    auto make_metric = [](scratch_t const& allocator, std::size_t staging) noexcept {
        metric_t metric {fixture_t::kernel_k, allocator};
        status_t const ready = metric.init(staging);
        return expected<metric_t> {std::move(metric), ready};
    };
    auto query_of = [&](std::size_t row, scratch_t const& allocator) noexcept {
        std::array<std::span<scalar_t const>, 1> const query_clouds {clouds[row]};
        auto [packed, packed_status] = query_t::make(query_clouds, allocator, 0);
        return expected<query_t> {std::move(packed), status_of(packed_status)};
    };
    auto matches = [&](std::size_t entry, nk::matrix_view<scalar_t> const& view) noexcept {
        std::vector<scalar_t> const& cloud = clouds[entry];
        return view.extent(0) * 3 == cloud.size() &&
               std::memcmp(view.data(), cloud.data(), cloud.size() * sizeof(scalar_t)) == 0;
    };
    return test_basic_nvme_view<fixture_t, nvme_index_t>(seed, clouds.size(), "chamfer", make_cloud, make_metric,
                                                         query_of, matches);
}

std::size_t check_shapes(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_scenarios<chamfer_fixture_t>(settings.seed);
    failures += test_basic_scenarios<procrustes_fixture_t>(settings.seed);
    failures += test_basic_safegraphs_roundtrip<chamfer_fixture_t>(settings.seed, "uuid", "chamfer");
    failures += test_basic_safegraphs_roundtrip<procrustes_fixture_t>(settings.seed, "uuid", "wasserstein_procrustes");
    failures += test_basic_safegraphs_hostile<chamfer_fixture_t>(settings.seed, "chamfer");
    failures += test_basic_safegraphs_hostile<procrustes_fixture_t>(settings.seed, "wasserstein_procrustes");
    failures += check_shapes_kernels<chamfer_fixture_t>(settings.seed);
    failures += check_shapes_kernels<procrustes_fixture_t>(settings.seed);
    failures += check_shapes_codec<chamfer_fixture_t>(settings.seed);
    failures += check_shapes_codec<procrustes_fixture_t>(settings.seed);
    failures += check_shapes_packing<chamfer_fixture_t>(settings.seed);
    failures += check_shapes_packing<procrustes_fixture_t>(settings.seed);
    failures += check_shapes_nvme(settings.seed);
    return report("shapes", failures);
}

} // namespace unum::usearch::test::shapes
