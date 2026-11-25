/**
 *  @file test/dense/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date April 21, 2026
 *  @brief Smoke tests for the 1-to-1, 1-to-4, and tiled kernels of @c dense/ram_metric.hpp against
 *      NumKong's monolithic 1-to-1 reference.
 *
 *  The reference is @c nk::dot / @c nk::angular / @c nk::sqeuclidean, and every scalar the library
 *  stores runs at the serial floor and at every rung a CPU class ships. USearch's @c distance and
 *  @c distances entry points route a kit (per scalar + capability) through a single trampoline, the
 *  metric a runtime argument of its epilogue. The test:
 *
 *  1. Walks every stored scalar in shrinking dynamic range and every @c (kernel,scalar,capability)
 *     tuple inside @c test_ram_metric. The concept @c dense::a_quads_supported_kernel gates each
 *     call, so combinations not yet adapted compile to no-ops and new kits light up their cells.
 *  2. For each supported tuple and three dimensions, computes a baseline distance with NumKong's
 *     monolithic API - @c nk::dot returns the raw inner product, wrapped to <tt>1 - dot</tt> for the dot
 *     kernel; @c nk::sqeuclidean is compared to USearch's squared euclidean), then calls
 *     @c distance (1-to-1) and @c distances (1-to-4) and compares with a relative tolerance.
 *  3. For tuples with a matrix engine, @c check_tiles packs one tile of queries plus a ragged tail,
 *     streams stored rows against the pack, finalizes the raw dots through the quads kit, and
 *     compares every cell against @c distance, skipping where the machine lacks the engine.
 *  4. Each check counts into the run, so a dtype unit's case reports every tuple that strayed.
 *
 *  The input vectors are populated with cyclic values from {0..7} - representable exactly in every
 *  stored scalar (BF16..E2M1, i8/u8, i4/u4) - at four different cyclic rotations so the four
 *  candidates differ from the query and from each other.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring> // `std::memcpy` - the dot lanes a finalizer takes

#include <array>
#include <limits>
#include <string_view>

#include <fmt/format.h>

#include <numkong/dot.hpp>
#include <numkong/set.hpp>
#include <numkong/spatial.hpp>
#include <numkong/vector.hpp>

#include "usearch/dense/ram_metric.hpp"

#include "harness.hpp"

namespace unum::usearch::test::dense {
using namespace unum::usearch::dense;

/** A ragged tail, a whole number of registers, and a long odd row. */
constexpr std::array<std::size_t, 3> dimensions_to_test = {7, 64, 257};

/** True iff a tile engine is compiled for this tuple. The tiles ladder always names a
 *  kit - @c no_tiles_kit_t at minimum - so the gate is its @c has_tiles_k verdict, never
 *  the alias's existence. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept a_tiles_supported_kernel = a_quads_supported_kernel<kernel_, scalar_, caps_> &&
                                   tiles_ladder<scalar_, caps_>::has_tiles_k;

/** Populates the input @p vec with small scalars - universally representable in any stored datatype
 *  from BF16 to E2M1 nibble-sized floats, but at a different cyclic rotation. */
template <typename scalar_>
static void fill_cyclic(nk::vector_span<scalar_> vec, std::size_t rotation) noexcept {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    for (std::size_t dimension_index = 0; dimension_index < vec.size(); ++dimension_index)
        vec[dimension_index] = make_scalar<scalar_>(
            universally_representable_values[(dimension_index + rotation) % universally_representable_values.size()]);
}

/** The family a kernel's distances compare in: Hamming counts bits exactly, every other kernel
 *  finalizes in floats. */
template <nk_kernel_kind_t kernel_>
constexpr comparison_family_t dense_comparison_family() noexcept {
    return kernel_ == nk_kernel_hamming_k ? comparison_family_t::exact_k : comparison_family_t::approximate_k;
}

/** Computes the reference distance via NumKong's monolithic 1-to-1 API, adjusted so the value
 *  matches USearch's convention - dot returns (1-inner), angular returns (1-cos_sim) clamped to
 *  [0,2], euclidean is sqrt'd via @c nk::euclidean. */
template <nk_kernel_kind_t kernel_, typename scalar_>
static f32_t reference_distance(nk::vector_view<scalar_> first_vector,
                                nk::vector_view<scalar_> second_vector) noexcept {
    if constexpr (kernel_ == nk_kernel_dot_k) {
        auto const [inner_product, status] = nk::dot<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return (std::numeric_limits<f32_t>::max)();
        return f32_t(1) - static_cast<f32_t>(inner_product);
    }
    else if constexpr (kernel_ == nk_kernel_angular_k) {
        auto const [angular_value, status] = nk::angular<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return (std::numeric_limits<f32_t>::max)();
        return static_cast<f32_t>(angular_value);
    }
    else if constexpr (kernel_ == nk_kernel_euclidean_k) {
        auto const [euclidean_value, status] = nk::euclidean<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return (std::numeric_limits<f32_t>::max)();
        return static_cast<f32_t>(euclidean_value);
    }
    else if constexpr (kernel_ == nk_kernel_hamming_k) {
        auto const [hamming_value, status] = nk::hamming<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return (std::numeric_limits<f32_t>::max)();
        return static_cast<f32_t>(hamming_value);
    }
    else if constexpr (kernel_ == nk_kernel_jaccard_k) {
        auto const [jaccard_value, status] = nk::jaccard<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return (std::numeric_limits<f32_t>::max)();
        return static_cast<f32_t>(jaccard_value);
    }
}

/** No-op overload picked when the (kernel, scalar, caps) tuple isn't wired up - lets the caller
 *  enumerate the full capability list without if-constexpr gates inline. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void check_quads(std::size_t /*dim*/) noexcept {}

/** Runs one (kernel, scalar, caps, dim) check - returns the number of mismatches across both the
 *  1-to-1 and 1-to-4 paths. Body instantiates only for adapted tuples. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires a_quads_supported_kernel<kernel_, scalar_, caps_>
static void check_quads(std::size_t dim) noexcept {
    // With a single capability bit requested, exactly one cascade row can match, so the
    // winner is either that tier or the serial fallback. Anything else means a row names
    // a kit from another backend - numerically indistinguishable, hence the static check.
    static_assert(quads_ladder<scalar_, caps_>::kit_t::capability_k == caps_ ||
                      quads_ladder<scalar_, caps_>::kit_t::capability_k == nk_cap_serial_k,
                  "Capability cascade selected a tier that was never requested");
    using quads_kit_t = typename quads_ladder<scalar_, caps_>::kit_t;

    nk::vector<scalar_> query = make_vector<scalar_>(dim);
    nk::vector<scalar_> first_candidate = make_vector<scalar_>(dim);
    nk::vector<scalar_> second_candidate = make_vector<scalar_>(dim);
    nk::vector<scalar_> third_candidate = make_vector<scalar_>(dim);
    nk::vector<scalar_> fourth_candidate = make_vector<scalar_>(dim);
    fill_cyclic(query.span(), 0);
    fill_cyclic(first_candidate.span(), 1);
    fill_cyclic(second_candidate.span(), 2);
    fill_cyclic(third_candidate.span(), 3);
    fill_cyclic(fourth_candidate.span(), 4);

    auto query_view = query.view();
    auto first_candidate_view = first_candidate.view();
    auto second_candidate_view = second_candidate.view();
    auto third_candidate_view = third_candidate.view();
    auto fourth_candidate_view = fourth_candidate.view();

    auto query_norms = norms<quads_kit_t>(query_view);
    auto first_candidate_norms = norms<quads_kit_t>(first_candidate_view);
    auto second_candidate_norms = norms<quads_kit_t>(second_candidate_view);
    auto third_candidate_norms = norms<quads_kit_t>(third_candidate_view);
    auto fourth_candidate_norms = norms<quads_kit_t>(fourth_candidate_view);

    f32_t const first_reference = reference_distance<kernel_>(query_view, first_candidate_view);
    f32_t const second_reference = reference_distance<kernel_>(query_view, second_candidate_view);
    f32_t const third_reference = reference_distance<kernel_>(query_view, third_candidate_view);
    f32_t const fourth_reference = reference_distance<kernel_>(query_view, fourth_candidate_view);

    f32_t const first_measured = distance<quads_kit_t>(kernel_, query_view, query_norms,
                                                       first_candidate_view, //
                                                       first_candidate_norms);
    f32_t const second_measured = distance<quads_kit_t>(kernel_, query_view, query_norms,
                                                        second_candidate_view, //
                                                        second_candidate_norms);
    f32_t const third_measured = distance<quads_kit_t>(kernel_, query_view, query_norms,
                                                       third_candidate_view, //
                                                       third_candidate_norms);
    f32_t const fourth_measured = distance<quads_kit_t>(kernel_, query_view, query_norms,
                                                        fourth_candidate_view, //
                                                        fourth_candidate_norms);

    error_stats_t single_errors, quad_errors;
    single_errors.add(first_measured, first_reference);
    single_errors.add(second_measured, second_reference);
    single_errors.add(third_measured, third_reference);
    single_errors.add(fourth_measured, fourth_reference);

    auto distances_array = distances<quads_kit_t>(
        kernel_, query_view, query_norms,
        {first_candidate_view, second_candidate_view, third_candidate_view, fourth_candidate_view},
        {first_candidate_norms, second_candidate_norms, third_candidate_norms, fourth_candidate_norms});
    quad_errors.add(distances_array[0], first_reference);
    quad_errors.add(distances_array[1], second_reference);
    quad_errors.add(distances_array[2], third_reference);
    quad_errors.add(distances_array[3], fourth_reference);
    expect_errors("one-to-one distances", dense_comparison_family<kernel_>(), comparison_verdict_t::assert_k,
                  single_errors, 0);
    expect_errors("one-to-four distances", dense_comparison_family<kernel_>(), comparison_verdict_t::assert_k,
                  quad_errors, 0);
}

/** No-op sibling picked when no tile engine exists for the tuple. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static void check_tiles() noexcept {}

/** A tiles kit against the 4-way path as the oracle: one tile of queries plus a ragged tail packs
 *  once, stored rows stream against the pack where they lie, and every raw dot, finalized by the
 *  quads kit under the runtime metric, must match the 1-to-1 distance. Skips silently when the
 *  machine lacks the engine at runtime. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires a_tiles_supported_kernel<kernel_, scalar_, caps_>
static void check_tiles() noexcept {
    nk_capability_t available = 0;
    if (nk_capabilities_enabled_cpu(&available) != nk_success_k || (available & caps_) != caps_) return;
    if (nk_configure_thread_cpu(available) != nk_success_k) return;

    using tiles_kit_t = typename tiles_ladder<scalar_, caps_>::kit_t;
    using quads_kit_t = typename quads_ladder<scalar_, caps_>::kit_t;
    using raw_dot_t = typename scalar_::dot_result_t::raw_t;
    constexpr std::size_t queries_k = tiles_kit_t::tile_columns_k + 3, rows_k = 37, dimensions_k = 128;

    nk::matrix<scalar_> storage = make_matrix<scalar_>(queries_k + rows_k, dimensions_k);
    for (std::size_t row = 0; row != queries_k + rows_k; ++row) fill_cyclic(storage.row(row).as_vector(), row);
    nk::matrix_view<scalar_> const queries = storage(nk::range(std::size_t {0}, queries_k), nk::slice);
    nk::matrix_view<scalar_> const rows = storage(nk::range(queries_k, queries_k + rows_k), nk::slice);
    std::size_t const stride = static_cast<std::size_t>(queries.stride_bytes(0));
    nk::vector<typename scalar_::dot_result_t> query_sumsqs = make_vector<typename scalar_::dot_result_t>(queries_k);
    for (std::size_t query = 0; query != queries_k; ++query)
        query_sumsqs[query] = norms<quads_kit_t>(queries.row(query).as_vector()).sumsq;

    nk_size_t packed_bytes = 0;
    expect_true(tiles_kit_t::pack_size(queries_k, dimensions_k, &packed_bytes) == nk_success_k);
    nk::vector<nk::u8_t> packed = make_vector<nk::u8_t>(packed_bytes);
    expect_true(tiles_kit_t::pack(queries.raw_values_data(), queries_k, dimensions_k, stride, packed.data(), 0,
                                  queries_k, nullptr) == nk_success_k);
    nk::matrix<raw_dot_t> dots = make_matrix<raw_dot_t>(rows_k, queries_k);
    expect_true(tiles_kit_t::dots_packed(rows.raw_values_data(), packed.data(), dots.data(), rows_k, queries_k,
                                         dimensions_k, stride, queries_k * sizeof(raw_dot_t), nullptr, 0,
                                         nullptr) == nk_success_k);

    error_stats_t tile_errors;
    for (std::size_t row = 0; row != rows_k; ++row) {
        auto const row_vector = rows.row(row).as_vector();
        auto const row_norms = norms<quads_kit_t>(row_vector);
        std::span<raw_dot_t const> const row_dots = to_span(dots.row(row).as_vector());
        for (std::size_t first = 0; first < queries_k; first += 4) {
            std::size_t const lanes = min_of<std::size_t>(4, queries_k - first);
            typename quads_kit_t::finalized_vec_t lane_dots {}, lane_sumsqs {};
            std::memcpy(&lane_dots, row_dots.subspan(first, lanes).data(), lanes * sizeof(raw_dot_t));
            std::memcpy(&lane_sumsqs, to_span(query_sumsqs).subspan(first, lanes).data(), lanes * sizeof(raw_dot_t));
            std::array<f32_t, 4> const tiled = convert_to_distances<quads_kit_t>(kernel_, lane_dots, row_norms.sumsq,
                                                                                 lane_sumsqs);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                auto const query_vector = queries.row(first + lane).as_vector();
                tile_errors.add(tiled[lane], distance<quads_kit_t>(kernel_, row_vector, row_norms, query_vector,
                                                                   norms<quads_kit_t>(query_vector)));
            }
        }
    }
    expect_errors("tiled distances", dense_comparison_family<kernel_>(), comparison_verdict_t::assert_k, tile_errors,
                  0);
}

template <nk_kernel_kind_t kernel_, typename scalar_>
static void check_kernel() noexcept {
    for (std::size_t const requested : dimensions_to_test) {
        // Packed scalars hold whole bytes of dimensions, so a ragged request rounds up a byte.
        std::size_t const dim = round_up_to_multiple<std::size_t>(requested, nk::dimensions_per_value<scalar_>());
        // Every rung a CPU class ships, newest silicon first: x86, then Arm, each by generation,
        // then WebAssembly, then the serial floor.
        check_quads<kernel_, scalar_, nk_cap_diamond_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_sierra_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_genoa_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_alder_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_icelake_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_skylake_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_haswell_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_neonfp8_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_neonbfdot_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_neonfhm_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_neonsdot_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_neon_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_v128relaxed_k>(dim);
        check_quads<kernel_, scalar_, nk_cap_serial_k>(dim);
    }
    check_tiles<kernel_, scalar_, nk_cap_graniteamx_k>();
    check_tiles<kernel_, scalar_, nk_cap_sapphireamx_k>();
    check_tiles<kernel_, scalar_, nk_cap_smebi32_k>();
    check_tiles<kernel_, scalar_, nk_cap_sme_k>();
}

/** Every kernel of the stored @p scalar_ at every rung, against NumKong's reference. */
template <typename scalar_>
static void test_ram_metric() noexcept {
    check_kernel<nk_kernel_dot_k, scalar_>();
    check_kernel<nk_kernel_angular_k, scalar_>();
    check_kernel<nk_kernel_euclidean_k, scalar_>();
}

/** Binary inputs run dot/hamming/jaccard instead of dot/angular/euclidean. */
template <>
[[maybe_unused]] void test_ram_metric<nk::u1x8_t>() noexcept {
    check_kernel<nk_kernel_dot_k, nk::u1x8_t>();
    check_kernel<nk_kernel_hamming_k, nk::u1x8_t>();
    check_kernel<nk_kernel_jaccard_k, nk::u1x8_t>();
}

} // namespace unum::usearch::test::dense
