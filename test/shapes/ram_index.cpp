/**
 *  @file test/shapes/ram_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The shapes suite: the basic scenarios over packed 3D clouds, then the kits, bounds, codec
 *      and packing.
 *
 *  Clouds are eight planted prototypes of 12 to 16 points, each drawn with a few jittered points,
 *  one in four mirrored, and under a random rigid motion, so distances cluster by prototype while
 *  raw coordinates do not. Every cloud falls in [8, 16) on each axis, where one storage ulp exceeds
 *  the rounding of the packed coordinates twice over, so unpacking returns the inserted pose
 *  exactly, a reload packs the same bytes, and the scenarios' checks of identical answers after a
 *  reload hold in @c f16 as well as @c f32. The scenarios hold every tier to the serial one through
 *  the public metric, and the kernels to an @c f64 reference over the raw clouds.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::sqrt`, `std::fabs`, `std::fmax`
#include <cstring> // `std::memcmp` - unpacked poses and packs compared as bytes

#include <algorithm>   // `std::min_element` - the best of a full round
#include <array>       // `std::array` - points, handmade triples, the kernel rounds
#include <bit>         // `std::bit_cast` - the keys
#include <limits>      // `std::numeric_limits` - the references' infinities, the oracle's failure sentinel
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
    for (scalar_type_ const coordinate : cloud)
        extent = std::fmax(extent, std::fabs(widened_value<scalar_type_>(coordinate)));
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
    split_mix_generator_t random_generator {seed};
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
                cloud.push_back(narrowed_value<scalar_type_>(moved[dimension] + translation[dimension]));
        }
    }
    return clouds;
}

/** A cloud's points in @c f64, centered on its centroid and turned onto its principal axes. */
using shapes_points_t = std::vector<std::array<f64_t, 3>>;

/** A transport plan between @c n and @c m points in units of 1 / (n × m), row-major. */
using shapes_plan_t = std::vector<u32_t>;

/** Diagonalizes the symmetric row-major @p matrix by cyclic Jacobi rotations into its eigenvalues,
 *  descending, and their unit eigenvectors, the columns of @p vectors. */
static std::array<f64_t, 3> symmetric_eigen_reference(std::array<f64_t, 9> matrix,
                                                      std::array<f64_t, 9>& vectors) noexcept {
    vectors = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (std::size_t sweep = 0; sweep != 64; ++sweep)
        for (std::size_t p = 0; p != 2; ++p)
            for (std::size_t q = p + 1; q != 3; ++q) {
                f64_t const off = matrix[p * 3 + q];
                if (off == 0) continue;
                f64_t const theta = (matrix[q * 3 + q] - matrix[p * 3 + p]) / (2 * off);
                f64_t const tangent = (theta < 0 ? -1.0 : 1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                f64_t const cosine = 1 / std::sqrt(tangent * tangent + 1), sine = tangent * cosine;
                for (std::size_t k = 0; k != 3; ++k) {
                    f64_t const kp = matrix[k * 3 + p], kq = matrix[k * 3 + q];
                    matrix[k * 3 + p] = cosine * kp - sine * kq, matrix[k * 3 + q] = sine * kp + cosine * kq;
                }
                for (std::size_t k = 0; k != 3; ++k) {
                    f64_t const pk = matrix[p * 3 + k], qk = matrix[q * 3 + k];
                    matrix[p * 3 + k] = cosine * pk - sine * qk, matrix[q * 3 + k] = sine * pk + cosine * qk;
                }
                for (std::size_t k = 0; k != 3; ++k) {
                    f64_t const kp = vectors[k * 3 + p], kq = vectors[k * 3 + q];
                    vectors[k * 3 + p] = cosine * kp - sine * kq, vectors[k * 3 + q] = sine * kp + cosine * kq;
                }
            }
    std::array<f64_t, 3> values {matrix[0], matrix[4], matrix[8]};
    for (std::size_t sorted = 0; sorted != 2; ++sorted)
        for (std::size_t later = sorted + 1; later != 3; ++later)
            if (values[sorted] < values[later]) {
                std::swap(values[sorted], values[later]);
                for (std::size_t k = 0; k != 3; ++k) std::swap(vectors[k * 3 + sorted], vectors[k * 3 + later]);
            }
    return values;
}

/** @p cloud of interleaved coordinates in @c f64, centered and turned onto the eigenvectors of
 *  its covariance, which every kind's 8 flips start from, so the signs of the axes are free. */
template <typename scalar_type_>
shapes_points_t shapes_points_reference(std::span<scalar_type_ const> cloud) {
    std::size_t const count = cloud.size() / 3;
    std::array<f64_t, 3> centroid {};
    for (std::size_t point = 0; point != count; ++point)
        for (std::size_t axis = 0; axis != 3; ++axis)
            centroid[axis] += widened_value<scalar_type_>(cloud[point * 3 + axis]) / static_cast<f64_t>(count);
    shapes_points_t centered(count);
    std::array<f64_t, 9> covariance {};
    for (std::size_t point = 0; point != count; ++point) {
        for (std::size_t axis = 0; axis != 3; ++axis)
            centered[point][axis] = widened_value<scalar_type_>(cloud[point * 3 + axis]) - centroid[axis];
        for (std::size_t row = 0; row != 3; ++row)
            for (std::size_t column = 0; column != 3; ++column)
                covariance[row * 3 + column] += centered[point][row] * centered[point][column];
    }
    std::array<f64_t, 9> axes;
    symmetric_eigen_reference(covariance, axes);
    shapes_points_t turned(count);
    for (std::size_t point = 0; point != count; ++point)
        for (std::size_t axis = 0; axis != 3; ++axis)
            turned[point][axis] = axes[0 * 3 + axis] * centered[point][0] + axes[1 * 3 + axis] * centered[point][1] +
                                  axes[2 * 3 + axis] * centered[point][2];
    return turned;
}

static f64_t squared_distance(std::array<f64_t, 3> const& first, std::array<f64_t, 3> const& second) noexcept {
    f64_t sum = 0;
    for (std::size_t axis = 0; axis != 3; ++axis) sum += (first[axis] - second[axis]) * (first[axis] - second[axis]);
    return sum;
}

/** Chamfer or Hausdorff between @p first and @p second, the best of the 8 flips of the axes. */
static f64_t nearest_reference(nk_kernel_kind_t kernel, shapes_points_t const& first,
                               shapes_points_t const& second) noexcept {
    f64_t best = std::numeric_limits<f64_t>::infinity();
    for (unsigned flip = 0; flip != 8; ++flip) {
        shapes_points_t flipped = second;
        for (std::array<f64_t, 3>& point : flipped)
            for (std::size_t axis = 0; axis != 3; ++axis) point[axis] *= (flip >> axis) & 1 ? -1 : 1;
        f64_t sums[2] {}, maxima[2] {};
        for (std::size_t side = 0; side != 2; ++side) {
            shapes_points_t const& from = side ? flipped : first;
            shapes_points_t const& to = side ? first : flipped;
            for (std::array<f64_t, 3> const& point : from) {
                f64_t nearest = std::numeric_limits<f64_t>::infinity();
                for (std::array<f64_t, 3> const& partner : to)
                    nearest = std::fmin(nearest, squared_distance(point, partner));
                sums[side] += std::sqrt(nearest) / static_cast<f64_t>(from.size());
                maxima[side] = std::fmax(maxima[side], std::sqrt(nearest));
            }
        }
        best = std::fmin(
            best, kernel == nk_kernel_chamfers_packed_k ? (sums[0] + sums[1]) / 2 : std::fmax(maxima[0], maxima[1]));
    }
    return best;
}

/** The exact transport plan between @c n rows and @c m columns of uniform weights under @p costs,
 *  @b [n,m], by successive shortest paths: each row supplies m units, each column takes n. */
static shapes_plan_t transport_reference(std::size_t rows, std::size_t columns, std::vector<f64_t> const& costs) {
    shapes_plan_t plan(rows * columns, 0);
    std::vector<u32_t> supply(rows, static_cast<u32_t>(columns)), demand(columns, static_cast<u32_t>(rows));
    std::size_t const nodes = rows + columns;
    std::vector<f64_t> distance(nodes);
    std::vector<std::size_t> parent(nodes);
    // Rounding may leave a residual cycle a hair below zero, which an exact relaxation would chase.
    f64_t slack = 0;
    for (f64_t const cost : costs) slack = std::fmax(slack, std::fabs(cost) * 1e-12);
    for (std::size_t moved = 0; moved != rows * columns;) {
        // Rows with supply left start at zero; a row reaches a column along any arc, and a column
        // returns along an arc carrying flow, at the negated cost.
        for (std::size_t node = 0; node != nodes; ++node)
            distance[node] = node < rows && supply[node] ? 0.0 : std::numeric_limits<f64_t>::infinity(),
            parent[node] = nodes;
        for (bool relaxed = true; relaxed;) {
            relaxed = false;
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column) {
                    f64_t const cost = costs[row * columns + column];
                    if (distance[row] + cost < distance[rows + column] - slack)
                        distance[rows + column] = distance[row] + cost, parent[rows + column] = row, relaxed = true;
                    if (plan[row * columns + column] && distance[rows + column] - cost < distance[row] - slack)
                        distance[row] = distance[rows + column] - cost, parent[row] = rows + column, relaxed = true;
                }
        }
        std::size_t sink = nodes;
        for (std::size_t column = 0; column != columns; ++column)
            if (demand[column] && (sink == nodes || distance[rows + column] < distance[sink])) sink = rows + column;
        u32_t units = demand[sink - rows];
        std::size_t node = sink;
        for (; parent[node] != nodes; node = parent[node])
            if (node < rows) units = min_of(units, plan[node * columns + parent[node] - rows]);
        units = min_of(units, supply[node]);
        supply[node] -= units, demand[sink - rows] -= units, moved += units;
        for (node = sink; parent[node] != nodes; node = parent[node])
            if (node < rows) plan[node * columns + parent[node] - rows] -= units;
            else plan[parent[node] * columns + node - rows] += units;
    }
    return plan;
}

/** The orthogonal R of determinant @p sign maximizing tr(R × H) for the row-major @p cross H,
 *  which is Σ w × b × aᵀ, so R takes column cloud b onto row cloud a, through H = U × Σ × Vᵀ. */
static std::array<f64_t, 9> procrustes_rotation_reference(std::array<f64_t, 9> const& cross, f64_t sign) noexcept {
    std::array<f64_t, 9> gram {}, right, left;
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column)
            for (std::size_t k = 0; k != 3; ++k) gram[row * 3 + column] += cross[k * 3 + row] * cross[k * 3 + column];
    std::array<f64_t, 3> const squares = symmetric_eigen_reference(gram, right);
    for (std::size_t k = 0; k != 2; ++k) {
        f64_t const singular = std::sqrt(std::fmax(squares[k], 0.0));
        for (std::size_t row = 0; row != 3; ++row)
            left[row * 3 + k] = (cross[row * 3 + 0] * right[0 * 3 + k] + cross[row * 3 + 1] * right[1 * 3 + k] +
                                 cross[row * 3 + 2] * right[2 * 3 + k]) /
                                singular;
    }
    for (std::size_t row = 0; row != 3; ++row)
        left[row * 3 + 2] = left[((row + 1) % 3) * 3 + 0] * left[((row + 2) % 3) * 3 + 1] -
                            left[((row + 2) % 3) * 3 + 0] * left[((row + 1) % 3) * 3 + 1];
    f64_t along = 0;
    for (std::size_t row = 0; row != 3; ++row)
        along += left[row * 3 + 2] * (cross[row * 3 + 0] * right[0 * 3 + 2] + cross[row * 3 + 1] * right[1 * 3 + 2] +
                                      cross[row * 3 + 2] * right[2 * 3 + 2]);
    if (along < 0)
        for (std::size_t row = 0; row != 3; ++row) left[row * 3 + 2] = -left[row * 3 + 2];
    auto const rotation_with = [&](f64_t last) noexcept {
        std::array<f64_t, 9> rotation {};
        for (std::size_t row = 0; row != 3; ++row)
            for (std::size_t column = 0; column != 3; ++column)
                for (std::size_t k = 0; k != 3; ++k)
                    rotation[row * 3 + column] += (k == 2 ? last : 1.0) * right[row * 3 + k] * left[column * 3 + k];
        return rotation;
    };
    std::array<f64_t, 9> const rotation = rotation_with(1.0);
    f64_t const determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                              rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                              rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
    return determinant * sign < 0 ? rotation_with(-1.0) : rotation;
}

/** Wasserstein-Procrustes squared: from each of the 8 flips, exact plans alternating with the best
 *  rotation of the flip's handedness until the plan's cost stops falling, the best of the 8. Leaves
 *  the best plans of either handedness in @p proper_plan and @p improper_plan. */
static f64_t procrustes_reference(shapes_points_t const& first, shapes_points_t const& second,
                                  shapes_plan_t& proper_plan, shapes_plan_t& improper_plan) {
    std::size_t const rows = first.size(), columns = second.size();
    f64_t const mass = static_cast<f64_t>(rows * columns);
    f64_t best = std::numeric_limits<f64_t>::infinity(), best_proper = best, best_improper = best;
    std::vector<f64_t> costs(rows * columns);
    for (unsigned flip = 0; flip != 8; ++flip) {
        bool const proper = (((flip >> 0) ^ (flip >> 1) ^ (flip >> 2)) & 1) == 0;
        std::array<f64_t, 9> rotation {};
        for (std::size_t axis = 0; axis != 3; ++axis) rotation[axis * 4] = (flip >> axis) & 1 ? -1 : 1;
        for (f64_t start_best = std::numeric_limits<f64_t>::infinity();;) {
            for (std::size_t column = 0; column != columns; ++column) {
                std::array<f64_t, 3> turned {};
                for (std::size_t axis = 0; axis != 3; ++axis)
                    for (std::size_t k = 0; k != 3; ++k) turned[axis] += rotation[axis * 3 + k] * second[column][k];
                for (std::size_t row = 0; row != rows; ++row)
                    costs[row * columns + column] = squared_distance(first[row], turned);
            }
            shapes_plan_t const plan = transport_reference(rows, columns, costs);
            f64_t value = 0;
            for (std::size_t cell = 0; cell != plan.size(); ++cell) value += plan[cell] * costs[cell] / mass;
            if (!(value < start_best)) break;
            start_best = value, best = std::fmin(best, value);
            if (proper && value < best_proper) best_proper = value, proper_plan = plan;
            if (!proper && value < best_improper) best_improper = value, improper_plan = plan;
            std::array<f64_t, 9> cross {};
            for (std::size_t row = 0; row != rows; ++row)
                for (std::size_t column = 0; column != columns; ++column)
                    for (std::size_t b_axis = 0; b_axis != 3; ++b_axis)
                        for (std::size_t a_axis = 0; a_axis != 3; ++a_axis)
                            cross[b_axis * 3 + a_axis] += plan[row * columns + column] / mass * second[column][b_axis] *
                                                          first[row][a_axis];
            rotation = procrustes_rotation_reference(cross, proper ? 1.0 : -1.0);
        }
    }
    return best;
}

/** Gromov-Wasserstein's square loss: 32 Frank-Wolfe steps with exact line searches from the best
 *  proper and improper Wasserstein-Procrustes plans and from the product plan, the least loss. */
static f64_t gromov_reference(shapes_points_t const& first, shapes_points_t const& second) {
    std::size_t const rows = first.size(), columns = second.size(), cells = rows * columns;
    f64_t const mass = static_cast<f64_t>(cells);
    shapes_plan_t proper_plan, improper_plan;
    procrustes_reference(first, second, proper_plan, improper_plan);
    auto const distances = [](shapes_points_t const& points, std::vector<f64_t>& matrix) {
        std::size_t const count = points.size();
        matrix.resize(count * count);
        f64_t mean_square = 0;
        for (std::size_t i = 0; i != count; ++i)
            for (std::size_t k = 0; k != count; ++k)
                matrix[i * count + k] = std::sqrt(squared_distance(points[i], points[k])),
                                   mean_square += squared_distance(points[i], points[k]) /
                                                  static_cast<f64_t>(count * count);
        return mean_square;
    };
    std::vector<f64_t> row_distances, column_distances;
    f64_t const mean_squares = distances(first, row_distances) + distances(second, column_distances);
    // Dx × V × Dy for the plan V, into `product`
    auto const sandwich = [&](std::vector<f64_t> const& plan, std::vector<f64_t>& product) {
        product.assign(cells, 0.0);
        for (std::size_t k = 0; k != rows; ++k)
            for (std::size_t l = 0; l != columns; ++l)
                if (plan[k * columns + l] != 0)
                    for (std::size_t i = 0; i != rows; ++i)
                        for (std::size_t j = 0; j != columns; ++j)
                            product[i * columns + j] += plan[k * columns + l] * row_distances[i * rows + k] *
                                                        column_distances[l * columns + j];
    };
    auto const inner = [&](std::vector<f64_t> const& left, std::vector<f64_t> const& right) {
        f64_t sum = 0;
        for (std::size_t cell = 0; cell != cells; ++cell) sum += left[cell] * right[cell];
        return sum;
    };
    auto const weights = [&](shapes_plan_t const& units) {
        std::vector<f64_t> plan(cells);
        for (std::size_t cell = 0; cell != cells; ++cell) plan[cell] = units[cell] / mass;
        return plan;
    };

    f64_t best_loss = std::numeric_limits<f64_t>::infinity();
    std::vector<f64_t> plan, gradient, vertex, vertex_gradient, costs(cells);
    for (std::size_t start = 0; start != 3; ++start) {
        plan = start == 0   ? weights(proper_plan)
               : start == 1 ? weights(improper_plan)
                            : std::vector<f64_t>(cells, 1 / mass);
        sandwich(plan, gradient);
        f64_t gradient_plan = inner(gradient, plan);
        for (std::size_t step = 0; step != 32; ++step) {
            for (std::size_t cell = 0; cell != cells; ++cell) costs[cell] = -gradient[cell];
            vertex = weights(transport_reference(rows, columns, costs));
            sandwich(vertex, vertex_gradient);
            f64_t const gradient_vertex = inner(gradient, vertex), vertex_vertex = inner(vertex_gradient, vertex);
            f64_t const curvature = -2 * (vertex_vertex - 2 * gradient_vertex + gradient_plan);
            f64_t const slope = -4 * (gradient_vertex - gradient_plan);
            f64_t gamma = curvature + slope < 0 ? 1 : 0;
            if (curvature > 0) gamma = std::fmin(std::fmax(-slope / (2 * curvature), 0.0), 1.0);
            if (gamma == 0) break;
            for (std::size_t cell = 0; cell != cells; ++cell)
                plan[cell] = (1 - gamma) * plan[cell] + gamma * vertex[cell],
                gradient[cell] = (1 - gamma) * gradient[cell] + gamma * vertex_gradient[cell];
            gradient_plan = inner(gradient, plan);
        }
        best_loss = std::fmin(best_loss, mean_squares - 2 * gradient_plan);
    }
    return best_loss;
}

/** The distance of @p kernel between two raw clouds, in @c f64, through no kit and no pack. */
template <typename scalar_type_>
f64_t shapes_distance_reference(nk_kernel_kind_t kernel, std::span<scalar_type_ const> first_cloud,
                                std::span<scalar_type_ const> second_cloud) {
    shapes_points_t const first = shapes_points_reference(first_cloud), second = shapes_points_reference(second_cloud);
    shapes_plan_t proper_plan, improper_plan;
    switch (kernel) {
    case nk_kernel_chamfers_packed_k:
    case nk_kernel_hausdorffs_packed_k: return nearest_reference(kernel, first, second);
    case nk_kernel_wassersteinprocrustes_packed_k:
        return std::sqrt(procrustes_reference(first, second, proper_plan, improper_plan));
    case nk_kernel_gromovwassersteins_packed_k: return std::sqrt(std::fmax(gromov_reference(first, second), 0.0)) / 2;
    default: return std::numeric_limits<f64_t>::quiet_NaN();
    }
}

/** How far a kit's distance between two clouds may stray from @ref shapes_distance_reference.
 *  Packing rounds each turned point once into @p scalar_type_, moving it by at most √3 × u × r,
 *  with u the unit roundoff and r the largest radius of its cloud, and every kind moves by at
 *  most the sum of the two shifts. */
template <typename scalar_type_>
f64_t shapes_storage_bound(std::span<scalar_type_ const> first_cloud, std::span<scalar_type_ const> second_cloud) {
    f64_t const unit_roundoff = std::ldexp(1.0, -static_cast<int>(scalar_type_::mantissa_bits()) - 1);
    f64_t radii = 0;
    for (shapes_points_t const& points :
         {shapes_points_reference(first_cloud), shapes_points_reference(second_cloud)}) {
        f64_t largest = 0;
        for (std::array<f64_t, 3> const& point : points) largest = std::fmax(largest, squared_distance(point, {}));
        radii += std::sqrt(largest);
    }
    return std::sqrt(3.0) * unit_roundoff * radii;
}

/** Three dumbbells of lengths 1, 2 and 3 whose packs lie along their first axes, so every kind
 *  between two of them is half the gap of their lengths. */
template <typename value_type_, typename distance_type_>
struct shapes_handmade {
    std::array<value_type_, 3> values;
    std::array<std::array<distance_type_, 3>, 3> distances;
};

/** One kit scoring one kind: values packed from @ref make_clouds through the scenario's ledgered
 *  allocator, held to the serial tier's metric, which every tier must match. */
template <typename kit_type_, nk_kernel_kind_t kernel_>
struct shapes_fixture {
    using kit_t = kit_type_;
    using scalar_t = typename kit_t::scalar_t;
    using index_t =
        shapes::ram_index<kit_t, util::uuid128_t, flaky_waiter_t, fu::standard_atomic_ref, counting_allocator_t>;
    using value_t = typename index_t::value_t;
    using allocator_t = typename index_t::allocator_t;
    using distance_t = typename index_t::distance_t;
    using metric_t = shapes_metric<kit_t>;
    static constexpr nk_kernel_kind_t kernel_k = kernel_;

    /** Procrustes solves a rotation per distance, so its scenarios grow half the nodes - measured:
     *  half keeps its @c test_basic_scenarios at the 173 s it took over 24 samples. */
    static constexpr std::size_t growth_nodes_k = basic_growth_nodes_k /
                                                  (kernel_ == nk_kernel_wassersteinprocrustes_packed_k ? 2 : 1);

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric(kernel_, forkunion_allocator_t {});
        return metric;
    }
    static codec<kit_t> make_codec() noexcept { return {}; }
    static util::uuid128_t make_key(std::size_t ordinal) {
        return std::bit_cast<util::uuid128_t>(std::array<u64_t, 2> {split_mix_bits(ordinal ^ 0x5A5A5Aull), ordinal});
    }
    static distance_t oracle(value_t const& query, value_t const& stored) noexcept {
        shapes_metric<typename shapes_ladder<scalar_t, nk_cap_serial_k>::kit_t> metric(kernel_,
                                                                                       forkunion_allocator_t {});
        metric.reset(query);
        std::array<distance_t, 1> distances {};
        if (metric.push(stored) != metric_staging_t::staged_k) return (std::numeric_limits<distance_t>::max)();
        metric.drain(distances, unbounded_radius<distance_t>());
        return metric.status() == status_code_t::success_k ? distances[0] : (std::numeric_limits<distance_t>::max)();
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
        auto [packed, pack_status] = value_t::make(clouds, allocator, kit_t::pack_size, kit_t::pack);
        if (failed(pack_status)) std::abort();
        return std::move(packed);
    }

    static shapes_handmade<value_t, distance_t> make_handmade(allocator_t const& allocator) {
        shapes_handmade<value_t, distance_t> handmade {{}, {{{0, 0.5, 1}, {0.5, 0, 0.5}, {1, 0.5, 0}}}};
        for (std::size_t length = 1; length != 4; ++length) {
            std::array<scalar_t, 6> dumbbell {};
            dumbbell[3] = make_scalar<scalar_t>(static_cast<unsigned>(length));
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

using chamfer_fixture_t = shapes_fixture<shapes_ladder<nk::f32_t, nk_cap_any_k>::kit_t, nk_kernel_chamfers_packed_k>;
using procrustes_fixture_t =
    shapes_fixture<shapes_ladder<nk::f16_t, nk_cap_any_k>::kit_t, nk_kernel_wassersteinprocrustes_packed_k>;

static_assert(us::basic::a_value<chamfer_fixture_t::value_t>);
static_assert(a_metric<chamfer_fixture_t::metric_t, chamfer_fixture_t::value_t, chamfer_fixture_t::distance_t>);
static_assert(
    a_metric<procrustes_fixture_t::metric_t, procrustes_fixture_t::value_t, procrustes_fixture_t::distance_t>);
static_assert(a_transparent_metric<procrustes_fixture_t::metric_t>);
static_assert(a_value_codec<codec<chamfer_fixture_t::kit_t>, chamfer_fixture_t::value_t>);
static_assert(a_value_codec<codec<procrustes_fixture_t::kit_t>, procrustes_fixture_t::value_t>);

/** Rounds of @p metric: the first pack as the query, the rest staged as they fit, drained within
 *  @p radius into @p distances. */
template <typename fixture_type_, typename metric_type_>
void score_round(metric_type_& metric, std::vector<typename fixture_type_::value_t> const& packs,
                 std::span<typename fixture_type_::distance_t> distances, typename fixture_type_::distance_t radius) {
    for (std::size_t first = 0; first != distances.size();) {
        metric.reset(packs[0]);
        std::size_t staged = 0;
        while (first + staged != distances.size() &&
               metric.push(packs[1 + first + staged]) == metric_staging_t::staged_k)
            ++staged;
        expect_true(staged != 0);
        metric.drain(distances.subspan(first, staged), radius);
        expect_equal(metric.status(), status_code_t::success_k);
        first += staged;
    }
}

/** Every kind through the fixture's kit at an unbounded radius: within the storage bound of
 *  the @c f64 reference, which scores the query's mirror image zero, as close as rounding lets
 *  it to the serial tier, and past a zero radius where the distance is. */
template <typename fixture_type_>
void test_shapes_kernels(u64_t seed) {
    using scalar_t = typename fixture_type_::scalar_t;
    using value_t = typename fixture_type_::value_t;
    using distance_t = typename fixture_type_::distance_t;
    using metric_t = typename fixture_type_::metric_t;
    using serial_metric_t = shapes_metric<typename shapes_ladder<scalar_t, nk_cap_serial_k>::kit_t>;
    constexpr std::size_t candidates_k = 8;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        std::vector<std::vector<scalar_t>> clouds = make_clouds<scalar_t>(candidates_k + 1,
                                                                          stream_key(seed, "kernels"));
        // The first candidate is the query with one coordinate negated, which every scalar holds
        // exactly.
        clouds[1] = clouds[0];
        for (std::size_t point = 0; point != clouds[1].size() / 3; ++point)
            clouds[1][point * 3] = -clouds[1][point * 3];
        std::vector<value_t> packs;
        for (std::vector<scalar_t> const& cloud : clouds) packs.push_back(fixture_type_::pack(cloud, allocator));

        error_stats_t kernel_errors;
        for (nk_kernel_kind_t const kernel : shapes_kernels_k) {
            metric_t metric(kernel, forkunion_allocator_t {});
            serial_metric_t serial_metric(kernel, forkunion_allocator_t {});
            std::array<distance_t, candidates_k> exact, zero_radius, serial;
            score_round<fixture_type_>(metric, packs, exact, unbounded_radius<distance_t>());
            score_round<fixture_type_>(metric, packs, zero_radius, distance_t {});
            score_round<fixture_type_>(serial_metric, packs, serial, unbounded_radius<distance_t>());
            for (std::size_t candidate = 0; candidate != candidates_k; ++candidate) {
                std::span<scalar_t const> const query = clouds[0], stored = clouds[1 + candidate];
                f64_t const reference = shapes_distance_reference(kernel, query, stored);
                expect_true(std::fabs(exact[candidate] - reference) <= shapes_storage_bound(query, stored));
                kernel_errors.add(exact[candidate], serial[candidate]);
                expect_true(exact[candidate] > 0 ? zero_radius[candidate] > 0
                                                 : zero_radius[candidate] == exact[candidate]);
            }
        }
        expect_errors("shapes kernels", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                      kernel_errors, 0);
    }
    expect_equal(ledger.live_bytes, 0);
}

/** A save returns each cloud's inserted pose within one storage rounding, and a reload, which packs
 *  the saved pose again, scores against the query as before to the storage precision. */
template <typename fixture_type_>
void test_shapes_codec(u64_t seed) {
    using scalar_t = typename fixture_type_::scalar_t;
    using value_t = typename fixture_type_::value_t;
    using distance_t = typename fixture_type_::distance_t;
    constexpr std::size_t clouds_k = 8;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        auto const shapes_codec = fixture_type_::make_codec();
        std::vector<std::vector<scalar_t>> const clouds = make_clouds<scalar_t>(clouds_k, stream_key(seed, "codec"));
        std::vector<value_t> packs, reloaded;
        f64_t extent = 0;
        for (std::vector<scalar_t> const& cloud : clouds) {
            packs.push_back(fixture_type_::pack(cloud, allocator));
            expect_equal(shapes_codec.storage_requirements(packs.back()), cloud.size() * sizeof(scalar_t));
            std::vector<scalar_t> saved(cloud.size());
            expect_equal(shapes_codec.save(packs.back(), std::as_writable_bytes(std::span {saved})),
                         status_code_t::success_k);
            f64_t const tolerance = storage_precision<scalar_t>(cloud_extent(cloud));
            std::size_t drifted = 0;
            for (std::size_t coordinate = 0; coordinate != cloud.size(); ++coordinate)
                drifted += std::fabs(widened_value<scalar_t>(saved[coordinate]) -
                                     widened_value<scalar_t>(cloud[coordinate])) > tolerance;
            expect_equal(drifted, 0u);
            reloaded.emplace_back(allocator);
            expect_equal(shapes_codec.load(std::as_bytes(std::span {saved}), reloaded.back()),
                         status_code_t::success_k);
            extent = std::fmax(extent, cloud_extent(cloud));
        }
        for (std::size_t candidate = 1; candidate != clouds_k; ++candidate) {
            distance_t const before = fixture_type_::oracle(packs[0], packs[candidate]);
            distance_t const after = fixture_type_::oracle(reloaded[0], reloaded[candidate]);
            expect_true(std::fabs(f64_t(after) - f64_t(before)) <= storage_precision<scalar_t>(extent));
        }
    }
    expect_equal(ledger.live_bytes, 0);
}

/** What the index relies on, as inserts, queries and loads all pack through the kit's own
 *  kernels: one cloud packs into the same bytes every time, and a stored shape scores zero
 *  against a query packed from the same coordinates, exactly under every kind but
 *  Gromov-Wasserstein, whose loss reaches zero only to the fast-math tolerance. */
template <typename fixture_type_>
void test_shapes_packing(u64_t seed) {
    using scalar_t = typename fixture_type_::scalar_t;
    using value_t = typename fixture_type_::value_t;
    using distance_t = typename fixture_type_::distance_t;
    using metric_t = typename fixture_type_::metric_t;
    using query_t = shapes::value<scalar_t>;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        auto const shapes_codec = fixture_type_::make_codec();
        error_stats_t transport_errors;
        for (std::vector<scalar_t> const& cloud : make_clouds<scalar_t>(4, stream_key(seed, "packing"))) {
            value_t first(allocator), second(allocator);
            expect_equal(shapes_codec.load(std::as_bytes(std::span {cloud}), first), status_code_t::success_k);
            expect_equal(shapes_codec.load(std::as_bytes(std::span {cloud}), second), status_code_t::success_k);
            nk_size_t bytes = 0;
            nk_status_t const pack_size_status = fixture_type_::kit_t::pack_size(cloud.size() / 3, &bytes);
            expect_true(pack_size_status == nk_success_k && std::memcmp(first[0], second[0], bytes) == 0);

            // The query comes from another allocator, as a call builds one in its arena.
            std::array<std::span<scalar_t const>, 1> const clouds {cloud};
            auto const [query, query_status] = query_t::make(clouds, {}, fixture_type_::kit_t::pack_size,
                                                             fixture_type_::kit_t::pack);
            if (!expect_true(succeeded(query_status))) continue;
            for (nk_kernel_kind_t const kernel : shapes_kernels_k) {
                metric_t metric(kernel, forkunion_allocator_t {});
                metric.reset(query);
                expect_true(metric.push(second) == metric_staging_t::staged_k);
                std::array<distance_t, 1> distances {1};
                metric.drain(distances, unbounded_radius<distance_t>());
                expect_equal(metric.status(), status_code_t::success_k);
                if (kernel == nk_kernel_gromovwassersteins_packed_k) transport_errors.add(distances[0], 0.0);
                else expect_equal(distances[0], distance_t {});
            }
        }
        expect_errors("transport self distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                      transport_errors, 0);
    }
    expect_equal(ledger.live_bytes, 0);
}

/** The sharded index of Chamfer shards over the heap, and the clouds its cells hold. */
using chamfer_shard_index_t = shapes::ram_index<chamfer_fixture_t::kit_t, util::uuid128_t, flaky_waiter_t,
                                                fu::standard_atomic_ref, forkunion_allocator_t>;
using chamfer_indexes_t = us::basic::ram_indexes<chamfer_shard_index_t>;

/** @p count clouds from @p seed under keys of their own, packed afresh by every call of
 *  @ref values, as a sharded cell's plans consume them. */
struct chamfer_entries_t {
    using scalar_t = chamfer_fixture_t::scalar_t;
    using value_t = chamfer_shard_index_t::value_t;

    std::vector<std::vector<scalar_t>> clouds;
    std::vector<util::uuid128_t> keys;

    chamfer_entries_t(std::size_t count, u64_t seed) : clouds(make_clouds<scalar_t>(count, seed)), keys(count) {
        for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = chamfer_fixture_t::make_key(seed + entry);
    }
    value_t pack(std::size_t entry) const {
        std::array<std::span<scalar_t const>, 1> const cloud {clouds[entry]};
        auto [packed, pack_status] = value_t::make(cloud, forkunion_allocator_t {no_memory_domain_k},
                                                   chamfer_fixture_t::kit_t::pack_size, chamfer_fixture_t::kit_t::pack);
        if (failed(pack_status)) std::abort();
        return std::move(packed);
    }
    std::vector<value_t> values(std::size_t count) const {
        std::vector<value_t> packed;
        for (std::size_t entry = 0; entry != count; ++entry) packed.push_back(pack(entry));
        return packed;
    }
};

/** Clouds over sharded indexes, as @c test_basic_sharded checks, held to the best a full round of
 *  the same metric over every cloud scores. */
void test_shapes_sharded(u64_t seed) {
    using distance_t = chamfer_fixture_t::distance_t;
    // One neighborhood of entries per sampled query, more than a walk reaches.
    constexpr std::size_t entries_k = recall_samples_k * basic_connectivity_k;
    chamfer_entries_t const entries {entries_k + recall_samples_k, stream_key(seed, "plural")};
    std::vector<chamfer_shard_index_t::value_t> const packs = entries.values(entries_k + recall_samples_k);
    chamfer_fixture_t::metric_t metric = chamfer_fixture_t::make_metric(), oracle = chamfer_fixture_t::make_metric();
    auto query_of = [&](std::size_t sample) { return entries.pack(entries_k + sample); };
    auto best_of = [&](std::size_t sample) {
        distance_t best = unbounded_radius<distance_t>();
        std::array<distance_t, 4> distances;
        for (std::size_t first = 0; first != entries_k;) {
            oracle.reset(packs[entries_k + sample]);
            std::size_t staged = 0;
            while (first + staged != entries_k && oracle.push(packs[first + staged]) == metric_staging_t::staged_k)
                ++staged;
            oracle.drain(std::span {distances}.first(staged), unbounded_radius<distance_t>());
            best = min_of(best, *std::min_element(distances.begin(), distances.begin() + staged));
            first += staged;
        }
        return best;
    };
    return test_basic_sharded<chamfer_indexes_t>(
        seed, metric, chamfer_fixture_t::make_codec(), std::span<util::uuid128_t const> {entries.keys}.first(entries_k),
        [&] { return entries.values(entries_k); }, query_of, best_of);
}

/** 600 clouds in a sharded index of two shards cut into two sharded indexes of two, as
 *  @c test_basic_sharded_cut checks. */
void test_shapes_sharded_cut(u64_t seed) {
    constexpr std::size_t entries_k = 600;
    chamfer_entries_t const entries {entries_k, stream_key(seed, "plural_cut")};
    chamfer_fixture_t::metric_t metric = chamfer_fixture_t::make_metric();
    return test_basic_sharded_cut<chamfer_indexes_t>(seed, metric, chamfer_fixture_t::make_codec(),
                                                     std::span<util::uuid128_t const> {entries.keys},
                                                     [&] { return entries.values(entries_k); });
}

/** Clouds sampled from two shards, as @c test_basic_sharded_sample checks. */
void test_shapes_sample(u64_t seed) {
    constexpr std::size_t entries_k = basic_growth_nodes_k;
    chamfer_entries_t const entries {entries_k, stream_key(seed, "plural_sample")};
    chamfer_fixture_t::metric_t metric = chamfer_fixture_t::make_metric();
    return test_basic_sharded_sample<chamfer_indexes_t>(
        seed, metric, std::span<util::uuid128_t const> {entries.keys}, [&] { return entries.values(entries_k); },
        no_fixed_coordinates_t {});
}

/** Clouds walked by a cursor over two shards, as @c test_basic_sharded_cursor checks. */
void test_shapes_cursor(u64_t seed) {
    constexpr std::size_t entries_k = basic_growth_nodes_k;
    chamfer_entries_t const entries {entries_k, stream_key(seed, "plural_cursor")};
    chamfer_fixture_t::metric_t metric = chamfer_fixture_t::make_metric();
    return test_basic_sharded_cursor<chamfer_indexes_t>(seed, metric, std::span<util::uuid128_t const> {entries.keys},
                                                        [&] { return entries.values(entries_k); });
}

/** Chamfer over storage: every cloud a ragged entry of points the view locates through its
 *  directory and each round packs again. */
struct chamfer_nvme_fixture_t {
    using ram_fixture_t = chamfer_fixture_t;
    using kit_t = ram_fixture_t::kit_t;
    using scalar_t = ram_fixture_t::scalar_t;
    using shard_index_t =
        shapes::ram_index<kit_t, util::uuid128_t, flaky_waiter_t, fu::standard_atomic_ref, forkunion_allocator_t>;
    template <typename engine_type_, typename node_id_type_ = u32_t>
    using nvme_t = us::basic::nvme_indexes<codec<kit_t>, util::uuid128_t, node_id_type_, ram_fixture_t::distance_t,
                                           engine_type_, fu::pool_t>;
    using scratch_metric_t = shapes_metric<kit_t, basic_nvme_scratch_t>;
    using query_t = nk::shapes_packed<scalar_t, std::allocator_traits<basic_nvme_scratch_t>::rebind_alloc<char>>;
    using coordinates_t = no_fixed_coordinates_t;
    static constexpr std::string_view metric_token_k = "chamfer";

    std::vector<std::vector<scalar_t>> clouds;

    explicit chamfer_nvme_fixture_t(u64_t seed) : clouds(make_clouds<scalar_t>(32, stream_key(seed, "nvme"))) {}
    std::size_t size() const noexcept { return clouds.size(); }
    template <typename index_type_>
    typename index_type_::value_t make_value(std::size_t entry,
                                             typename index_type_::allocator_t const& allocator) const {
        std::array<std::span<scalar_t const>, 1> const cloud {clouds[entry]};
        auto [packed, pack_status] = index_type_::value_t::make(cloud, allocator, kit_t::pack_size, kit_t::pack);
        if (failed(pack_status)) std::abort();
        return std::move(packed);
    }
    static expected<scratch_metric_t> make_scratch_metric(basic_nvme_scratch_t const& allocator, std::size_t) noexcept {
        return {scratch_metric_t {ram_fixture_t::kernel_k, allocator}, status_code_t::success_k};
    }
    static expected<query_t> pack(std::span<scalar_t const> points, basic_nvme_scratch_t const& allocator) noexcept {
        std::array<std::span<scalar_t const>, 1> const cloud {points};
        auto [packed, packed_status] = query_t::make(cloud, allocator, kit_t::pack_size, kit_t::pack);
        return {std::move(packed), status_of(packed_status)};
    }
    expected<query_t> query_of(std::size_t entry, basic_nvme_scratch_t const& allocator) const noexcept {
        return pack(clouds[entry], allocator);
    }
    static expected<query_t> stored_query(nk::matrix_view<scalar_t> const& stored,
                                          basic_nvme_scratch_t const& allocator) noexcept {
        return pack({stored.data(), stored.extent(0) * 3}, allocator);
    }
    bool matches(std::size_t entry, nk::matrix_view<scalar_t> const& view) const noexcept {
        std::vector<scalar_t> const& cloud = clouds[entry];
        return view.extent(0) * 3 == cloud.size() &&
               std::memcmp(view.data(), cloud.data(), cloud.size() * sizeof(scalar_t)) == 0;
    }
};

void run_shapes_cases() {
    u64_t const seed = test_run.settings.seed;
    run_test_case("test_basic_scenarios<chamfer_fixture_t>", test_basic_scenarios<chamfer_fixture_t>, seed);
    run_test_case("test_basic_scenarios<procrustes_fixture_t>", test_basic_scenarios<procrustes_fixture_t>, seed);
    run_test_case("test_basic_safegraphs_round_trip<chamfer_fixture_t>",
                  test_basic_safegraphs_round_trip<chamfer_fixture_t>, seed, "uuid", "chamfer");
    run_test_case("test_basic_safegraphs_round_trip<procrustes_fixture_t>",
                  test_basic_safegraphs_round_trip<procrustes_fixture_t>, seed, "uuid", "wasserstein-procrustes");
    run_test_case("test_basic_safegraphs_hostile<chamfer_fixture_t>", test_basic_safegraphs_hostile<chamfer_fixture_t>,
                  seed, "chamfer");
    run_test_case("test_basic_safegraphs_hostile<procrustes_fixture_t>",
                  test_basic_safegraphs_hostile<procrustes_fixture_t>, seed, "wasserstein-procrustes");
    run_test_case("test_shapes_kernels<chamfer_fixture_t>", test_shapes_kernels<chamfer_fixture_t>, seed);
    run_test_case("test_shapes_kernels<procrustes_fixture_t>", test_shapes_kernels<procrustes_fixture_t>, seed);
    run_test_case("test_shapes_codec<chamfer_fixture_t>", test_shapes_codec<chamfer_fixture_t>, seed);
    run_test_case("test_shapes_codec<procrustes_fixture_t>", test_shapes_codec<procrustes_fixture_t>, seed);
    run_test_case("test_shapes_packing<chamfer_fixture_t>", test_shapes_packing<chamfer_fixture_t>, seed);
    run_test_case("test_shapes_packing<procrustes_fixture_t>", test_shapes_packing<procrustes_fixture_t>, seed);
    run_test_case("test_shapes_sharded", test_shapes_sharded, seed);
    run_test_case("test_shapes_sharded_cut", test_shapes_sharded_cut, seed);
    run_test_case("test_shapes_sample", test_shapes_sample, seed);
    run_test_case("test_shapes_cursor", test_shapes_cursor, seed);
    run_basic_nvme_cases("chamfer_nvme_fixture_t", chamfer_nvme_fixture_t {seed}, seed);
}

} // namespace unum::usearch::test::shapes
