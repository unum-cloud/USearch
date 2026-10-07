/**
 *  @file test/basic/meshes.cpp
 *  @author Ash Vardanian
 *  @date August 26, 2026
 *  @brief The meshes unit of the basic-index suite.
 *
 *  3D point clouds compared through the first Gromov-Wasserstein lower bound - sorted intra-cloud
 *  distance spectra - the floating-distance, derived-state, rigid-motion-invariant corner, with the
 *  metric built on NumKong's kernels the way real caller metrics are.
 *
 *  Clouds are planted prototypes under random rigid motions with a few jittered points,
 *  so the spectra cluster while the raw coordinates do not - a broken metric that leaks
 *  coordinates fails the recall bar loudly. The spectrum is derived state: cached per
 *  value, never serialized, and recomputed by the codec on load.
 */
#include <cmath> // `std::sqrt`

#include <array>  // `std::array` - the handmade tetrahedra
#include <limits> // `std::numeric_limits` - the failed kernel's sentinel
#include <span>   // `std::span` - the metric's drain surface
#include <vector>

#include <numkong/spatial.hpp> // `nk::euclidean` - the spectra oracle's oracle

#include "basic.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

/** One vertex; plain floats, so codec roundtrips are bit-exact. */
struct point3_t {
    f32_t x {}, y {}, z {};
};

/** One cloud with its cached spectrum: the sorted multiset of all pairwise
 *  intra-cloud distances - the rigid-motion-invariant signature the metric compares. */
struct mesh_t {
    std::vector<point3_t> points;
    std::vector<f32_t> spectrum;
};

static f32_t point_distance(point3_t first, point3_t second) noexcept {
    std::array<f32_t, 3> const first_coordinates {first.x, first.y, first.z};
    std::array<f32_t, 3> const second_coordinates {second.x, second.y, second.z};
    auto const [distance, status] = nk::euclidean<nk::f32_t>(first_coordinates, second_coordinates);
    if (nk::failed(status)) return std::numeric_limits<f32_t>::max();
    return float(distance);
}

/** Recomputes the cached spectrum from the points - the generator's epilogue and
 *  the codec's on-load derivation. */
static void refresh_spectrum(mesh_t& mesh) {
    mesh.spectrum.clear();
    for (std::size_t first = 0; first != mesh.points.size(); ++first)
        for (std::size_t second = first + 1; second != mesh.points.size(); ++second)
            mesh.spectrum.push_back(point_distance(mesh.points[first], mesh.points[second]));
    std::sort(mesh.spectrum.begin(), mesh.spectrum.end());
}

/** The first GW lower bound between equal-size clouds: the root-mean-square gap
 *  between their sorted spectra, through NumKong's Euclidean kernel - the tier's
 *  intended shape, a caller metric built on the SIMD kernel library. */
static f64_t spectra_distance(std::span<f32_t const> first, std::span<f32_t const> second) noexcept {
    if (first.empty()) return 0.0;
    auto const [gap, status] = nk::euclidean<nk::f32_t>(first, second);
    if (nk::failed(status)) return std::numeric_limits<f64_t>::max();
    return static_cast<f64_t>(float(gap)) / std::sqrt(static_cast<f64_t>(first.size()));
}

/** The hand-rolled scalar twin of @c spectra_distance - the independent oracle:
 *  plain f64 accumulation, no library kernels, so a NumKong dispatch bug cannot
 *  cancel out of the scenarios' distance audits. */
static f64_t scalar_spectra_distance(std::span<f32_t const> first, std::span<f32_t const> second) noexcept {
    f64_t sum_of_squares = 0;
    for (std::size_t pair = 0; pair != first.size(); ++pair) {
        f64_t const gap = static_cast<f64_t>(first[pair]) - static_cast<f64_t>(second[pair]);
        sum_of_squares += gap * gap;
    }
    return first.size() ? std::sqrt(sum_of_squares / static_cast<f64_t>(first.size())) : 0.0;
}

/** The unit's batched metric: stages spans over the cached spectra, never
 *  recomputing one - "cached per push" is the whole point of the derived state. */
struct spectra_metric_t {
    std::span<f32_t const> query_spectrum_;
    std::vector<std::span<f32_t const>> staged_;

    void reset(mesh_t const& query) noexcept {
        query_spectrum_ = query.spectrum;
        staged_.clear();
    }
    basic_metric_staging_t push(mesh_t const& value) noexcept {
        return staged_.push_back(value.spectrum), basic_metric_staging_t::staged_k;
    }
    void drain(std::span<f64_t> distances, f64_t) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = spectra_distance(query_spectrum_, staged_[lane]);
        staged_.clear();
    }
};

/** Serializes the points back to back - never the spectrum, which the load path
 *  re-derives, the way norms re-derive in the dense tiers. The slice length is the point
 *  count, so nothing is self-delimited, and the interchange sees a ragged @c f32 cloud of
 *  @c [*,3] rows. */
struct mesh_codec_t {
    inline static constexpr dtype_t value_dtype_k = dtype_t::f32_k;
    inline static constexpr column_shape_t value_shape_k = matrix_shape_t {0, 3};

    std::size_t storage_requirements(mesh_t const& value) const noexcept {
        return value.points.size() * sizeof(point3_t);
    }
    status_t save(mesh_t const& value, bytes_span_t output) const noexcept {
        if (output.size() < storage_requirements(value)) return status_t::overflow_k;
        for (std::size_t point = 0; point != value.points.size(); ++point) {
            auto const point_bytes = std::bit_cast<std::array<std::byte, sizeof(point3_t)>>(value.points[point]);
            for (std::size_t position = 0; position != point_bytes.size(); ++position)
                output[point * sizeof(point3_t) + position] = point_bytes[position];
        }
        return status_t::success_k;
    }
    status_t load(bytes_view_t input, mesh_t& value) const noexcept {
        if (input.size() % sizeof(point3_t)) return status_t::malformed_k;
        std::size_t const points_count = input.size() / sizeof(point3_t);
        value.points.clear();
        for (std::size_t point = 0; point != points_count; ++point) {
            std::array<std::byte, sizeof(point3_t)> point_bytes;
            for (std::size_t position = 0; position != point_bytes.size(); ++position)
                point_bytes[position] = input[point * sizeof(point3_t) + position];
            value.points.push_back(std::bit_cast<point3_t>(point_bytes));
        }
        refresh_spectrum(value);
        return status_t::success_k;
    }
};

/** Rotates @p point by the unit quaternion @p rotation - the rigid motion the
 *  spectra must be blind to. */
static point3_t rotate_point(point3_t point, std::array<f32_t, 4> rotation) noexcept {
    f32_t const qw = rotation[0], qx = rotation[1], qy = rotation[2], qz = rotation[3];
    f32_t const tx = 2 * (qy * point.z - qz * point.y);
    f32_t const ty = 2 * (qz * point.x - qx * point.z);
    f32_t const tz = 2 * (qx * point.y - qy * point.x);
    return {point.x + qw * tx + (qy * tz - qz * ty), //
            point.y + qw * ty + (qz * tx - qx * tz), //
            point.z + qw * tz + (qx * ty - qy * tx)};
}

/** A regular tetrahedron scaled so every edge has length @p edge. */
static mesh_t make_tetrahedron(f32_t edge) {
    f32_t const scale = edge / (2.0f * std::sqrt(2.0f));
    mesh_t mesh;
    mesh.points = {{scale, scale, scale}, {scale, -scale, -scale}, {-scale, scale, -scale}, {-scale, -scale, scale}};
    refresh_spectrum(mesh);
    return mesh;
}

/** Three tetrahedra with edges 1, 2 & 3: six equal spectrum entries each, so the
 *  pairwise spectra distances are exactly the edge gaps. */
struct meshes_handmade_t {
    std::array<mesh_t, 3> values;
    std::array<std::array<f64_t, 3>, 3> distances;
};

template <typename wait_type_ = dummy_waiter_t, template <typename> class atomic_ref_template_ = standard_atomic_ref>
struct meshes_fixture {
    using index_t =
        basic_ram_index<mesh_t, us::uuid_t, u64_t, f64_t, wait_type_, atomic_ref_template_, counting_allocator_t>;

    static index_t make_index(allocation_ledger_t& ledger) { return index_t {counting_allocator_t {ledger}}; }
    static spectra_metric_t make_metric() { return {}; }
    static mesh_codec_t make_codec() { return {}; }
    static us::uuid_t make_key(std::size_t ordinal) {
        return std::bit_cast<us::uuid_t>(std::array<u64_t, 2> {mix_bits(ordinal ^ 0x3D3D3Dull), ordinal});
    }
    static f64_t oracle(mesh_t const& query, mesh_t const& stored) noexcept {
        return scalar_spectra_distance(query.spectrum, stored.spectrum);
    }
    static bool value_equals(mesh_t const& first, mesh_t const& second) noexcept {
        if (first.points.size() != second.points.size()) return false;
        for (std::size_t point = 0; point != first.points.size(); ++point)
            if (first.points[point].x != second.points[point].x || first.points[point].y != second.points[point].y ||
                first.points[point].z != second.points[point].z)
                return false;
        return true;
    }

    static meshes_handmade_t make_handmade() {
        return {{make_tetrahedron(1), make_tetrahedron(2), make_tetrahedron(3)}, {{{0, 1, 2}, {1, 0, 1}, {2, 1, 0}}}};
    }

    static std::vector<mesh_t> make_values(std::size_t count, u64_t seed) {
        constexpr std::size_t prototypes_k = 8;
        constexpr std::size_t points_per_mesh_k = 24;
        constexpr std::size_t jittered_points_k = 4;
        splitmix64_t random_generator {seed};
        uniform_below<u32_t> coordinate_distribution {2001};
        uniform_below<u32_t> prototype_distribution {prototypes_k};
        uniform_below<u32_t> point_distribution {points_per_mesh_k};
        auto draw_signed = [&](f32_t amplitude) noexcept {
            return (static_cast<f32_t>(coordinate_distribution(random_generator)) / 1000.0f - 1.0f) * amplitude;
        };

        std::array<mesh_t, prototypes_k> prototypes;
        for (auto& prototype : prototypes) {
            for (std::size_t point = 0; point != points_per_mesh_k; ++point)
                prototype.points.push_back({draw_signed(1), draw_signed(1), draw_signed(1)});
            refresh_spectrum(prototype);
        }

        std::vector<mesh_t> values;
        for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
            mesh_t mesh;
            mesh.points = prototypes[prototype_distribution(random_generator)].points;
            for (std::size_t jitter = 0; jitter != jittered_points_k; ++jitter) {
                point3_t& point = mesh.points[point_distribution(random_generator)];
                point.x += draw_signed(0.15f), point.y += draw_signed(0.15f), point.z += draw_signed(0.15f);
            }
            std::array<f32_t, 4> rotation {draw_signed(1), draw_signed(1), draw_signed(1), draw_signed(1)};
            f32_t const norm = std::sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] +
                                         rotation[2] * rotation[2] + rotation[3] * rotation[3]);
            if (norm > 0)
                for (auto& component : rotation) component /= norm;
            else rotation = {1, 0, 0, 0};
            point3_t const translation {draw_signed(2), draw_signed(2), draw_signed(2)};
            for (auto& point : mesh.points) {
                point = rotate_point(point, rotation);
                point.x += translation.x, point.y += translation.y, point.z += translation.z;
            }
            refresh_spectrum(mesh);
            values.push_back(std::move(mesh));
        }
        return values;
    }
};

using meshes_fixture_t = meshes_fixture<>;

static_assert(basic_value<mesh_t>);
static_assert(basic_metric<spectra_metric_t, mesh_t, f64_t>);
static_assert(!transparent_metric<spectra_metric_t>);
static_assert(basic_value_codec<mesh_codec_t, mesh_t>);

/** The invariance arm: a rigidly moved cloud must sit at spectra distance zero
 *  from its original - the property that makes the metric a GW lower bound. */
static std::size_t check_meshes_rigid_invariance() {
    std::size_t failures = 0;
    mesh_t const original = make_tetrahedron(2);
    mesh_t moved = original;
    constexpr f32_t half_k = 0.5f;
    std::array<f32_t, 4> const rotation {half_k, half_k, half_k, half_k}; // A unit quaternion.
    for (auto& point : moved.points) {
        point = rotate_point(point, rotation);
        point.x += 3.5f, point.y -= 1.25f, point.z += 0.75f;
    }
    refresh_spectrum(moved);
    failures += expect_near(meshes_fixture_t::oracle(original, moved), 0.0);
    failures += expect_near(meshes_fixture_t::oracle(original, make_tetrahedron(3)), 1.0);
    return failures;
}

/** Kernel-vs-scalar parity: the NumKong-built metric distance and the scalar
 *  oracle must agree over sampled cloud pairs - each also runs implicitly through
 *  every scenario's per-result audit. */
static std::size_t check_meshes_kernel_parity(u64_t seed) {
    std::size_t failures = 0;
    auto values = meshes_fixture_t::make_values(8, stream_key(seed, "kernel_parity"));
    for (std::size_t pair = 0; pair + 1 < values.size(); pair += 2) {
        auto const& first = values[pair].spectrum;
        auto const& second = values[pair + 1].spectrum;
        failures += expect_near(spectra_distance(first, second), scalar_spectra_distance(first, second));
    }
    return failures;
}

std::size_t test::basic::check_meshes(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_scenarios<meshes_fixture_t>(settings.seed);
    failures += test_basic_scenarios<meshes_fixture<flaky_waiter_t>>(settings.seed);
    failures += test_basic_scenarios<meshes_fixture<dummy_waiter_t, fu::preferred_atomic_ref>>(settings.seed);
    failures += check_meshes_rigid_invariance();
    failures += check_meshes_kernel_parity(settings.seed);
    // The uuid key column.
    failures += test_basic_safegraphs_roundtrip<meshes_fixture_t>(settings.seed, "uuid", "rmsd");
    failures += test_basic_safegraphs_hostile<meshes_fixture_t>(settings.seed, "rmsd");
    failures += test_basic_safegraphs_distributed<meshes_fixture_t>(settings.seed, "rmsd");
    return report("meshes", failures);
}
