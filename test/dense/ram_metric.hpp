/**
 *  @file test/dense/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date April 21, 2026
 *  @brief Smoke tests for @c dense/ram_metric.hpp 1-to-1, 1-to-4, and tiled kernels against
 *      NumKong's monolithic 1-to-1 reference - @c nk::dot / @c nk::angular / @c nk::sqeuclidean -
 *      shared by the per-dtype shards.
 *
 *  USearch's @c distance and @c distances entry points route a kit (per scalar +
 *  capability) through a single trampoline, the metric a runtime argument of its epilogue.
 *  The test:
 *
 *  1. Enumerates one NumKong scalar per shard and every @c (kernel,scalar,capability) tuple
 *     inside @c check_scalar. The @c dense_quads_supported concept gates each call so
 *     combinations not yet adapted are no-ops at compile time without breaking the build - new
 *     kits and chains light up the matching cells automatically.
 *  2. For each supported tuple and a range of dimensions, computes a baseline distance with
 *     NumKong's monolithic API - @c nk::dot returns the raw inner product, wrapped to `1 - dot`
 *     for the dot kernel; @c nk::sqeuclidean is compared to USearch's squared euclidean), then
 *     calls @c distance (1-to-1) and @c distances (1-to-4) and compares with a relative tolerance.
 *  3. For tuples with a matrix engine, @c check_tiles packs one tile of queries plus a ragged
 *     tail, streams stored rows against the pack, finalizes the raw dots through the quads kit,
 *     and compares every cell against the @c distance oracle, skipping silently when the machine
 *     lacks the engine at runtime.
 *  4. Each check returns its own failure count; counts roll up through the shard entry points to
 *     the suite's @c main without any global mutable state.
 *
 *  The input vectors are populated with cyclic values from {0..7} - representable
 *  exactly in every supported scalar (Float64..E2M1, i8/u8, i4/u4) - at four different
 *  cyclic rotations so the four candidates differ from the query and from each other.
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

#include "harness.hpp"
#include "usearch/dense/ram_metric.hpp"

namespace unum::usearch::test::dense {

constexpr std::array<std::size_t, 7> dimensions_to_test = {7, 8, 32, 64, 128, 257, 1024};

/** True iff a tile engine is compiled for this tuple. The tiles chain always names a
 *  kit - @c no_tiles_kit_t at minimum - so the gate is its @c has_tiles_k verdict, never
 *  the alias's existence. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept tiles_supported = dense_quads_supported<kernel_, scalar_, caps_> &&
                          dense_tiles_chain<scalar_, caps_>::has_tiles_k;

/** Populates the input @p vec with small scalars - universally representable in any
 *  supported datatype from Float64 to E2M1 nibble-sized floats, but at a different cyclic rotation. */
template <typename scalar_>
static void fill_cyclic(nk::vector_span<scalar_> vec, std::size_t rotation) noexcept {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    for (std::size_t dimension_index = 0; dimension_index < vec.size(); ++dimension_index)
        vec[dimension_index] = make_scalar<scalar_>(
            universally_representable_values[(dimension_index + rotation) % universally_representable_values.size()]);
}

/** Computes the reference distance via NumKong's monolithic 1-to-1 API, adjusted
 *  so the value matches USearch's convention - dot returns (1-inner), angular returns (1-cos_sim)
 *  clamped to [0,2], euclidean is sqrt'd via @c nk::euclidean. */
template <nk_kernel_kind_t kernel_, typename scalar_>
static f32_t reference_distance(nk::vector_view<scalar_> first_vector,
                                nk::vector_view<scalar_> second_vector) noexcept {
    if constexpr (kernel_ == nk_kernel_dot_k) {
        auto const [inner_product, status] = nk::dot<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return std::numeric_limits<f32_t>::max();
        return f32_t(1) - static_cast<f32_t>(inner_product);
    }
    else if constexpr (kernel_ == nk_kernel_angular_k) {
        auto const [angular_value, status] = nk::angular<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return std::numeric_limits<f32_t>::max();
        return static_cast<f32_t>(angular_value);
    }
    else if constexpr (kernel_ == nk_kernel_euclidean_k) {
        auto const [euclidean_value, status] = nk::euclidean<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return std::numeric_limits<f32_t>::max();
        return static_cast<f32_t>(euclidean_value);
    }
    else if constexpr (kernel_ == nk_kernel_hamming_k) {
        auto const [hamming_value, status] = nk::hamming<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return std::numeric_limits<f32_t>::max();
        return static_cast<f32_t>(hamming_value);
    }
    else if constexpr (kernel_ == nk_kernel_jaccard_k) {
        auto const [jaccard_value, status] = nk::jaccard<scalar_>(first_vector, second_vector);
        if (nk::failed(status)) return std::numeric_limits<f32_t>::max();
        return static_cast<f32_t>(jaccard_value);
    }
}

/** No-op overload picked when the (kernel, scalar, caps) tuple isn't wired up -
 *  lets the caller enumerate the full capability list without if-constexpr gates inline. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static int check_quads(std::size_t /*dim*/) noexcept {
    return 0;
}

/** Runs one (kernel, scalar, caps, dim) check - returns the number of mismatches
 *  across both the 1-to-1 and 1-to-4 paths. Body instantiates only for adapted tuples. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires dense_quads_supported<kernel_, scalar_, caps_>
static int check_quads(std::size_t dim) noexcept {
    // With a single capability bit requested, exactly one cascade row can match, so the
    // winner is either that tier or the serial fallback. Anything else means a row names
    // a kit from another backend - numerically indistinguishable, hence the static check.
    static_assert(dense_quads_chain<scalar_, caps_>::picked_k == caps_ ||
                      dense_quads_chain<scalar_, caps_>::picked_k == nk_cap_serial_k,
                  "Capability cascade selected a tier that was never requested");

    auto query = nk::vector<scalar_>::zeros(dim).value;
    auto first_candidate = nk::vector<scalar_>::zeros(dim).value;
    auto second_candidate = nk::vector<scalar_>::zeros(dim).value;
    auto third_candidate = nk::vector<scalar_>::zeros(dim).value;
    auto fourth_candidate = nk::vector<scalar_>::zeros(dim).value;
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

    auto query_norms = dense_norms<caps_>(query_view);
    auto first_candidate_norms = dense_norms<caps_>(first_candidate_view);
    auto second_candidate_norms = dense_norms<caps_>(second_candidate_view);
    auto third_candidate_norms = dense_norms<caps_>(third_candidate_view);
    auto fourth_candidate_norms = dense_norms<caps_>(fourth_candidate_view);

    f32_t const first_reference = reference_distance<kernel_>(query_view, first_candidate_view);
    f32_t const second_reference = reference_distance<kernel_>(query_view, second_candidate_view);
    f32_t const third_reference = reference_distance<kernel_>(query_view, third_candidate_view);
    f32_t const fourth_reference = reference_distance<kernel_>(query_view, fourth_candidate_view);

    f32_t const first_measured = distance<caps_>(kernel_, query_view, query_norms, first_candidate_view, //
                                                 first_candidate_norms);
    f32_t const second_measured = distance<caps_>(kernel_, query_view, query_norms, second_candidate_view, //
                                                  second_candidate_norms);
    f32_t const third_measured = distance<caps_>(kernel_, query_view, query_norms, third_candidate_view, //
                                                 third_candidate_norms);
    f32_t const fourth_measured = distance<caps_>(kernel_, query_view, query_norms, fourth_candidate_view, //
                                                  fourth_candidate_norms);

    std::size_t failures = 0;
    failures += expect_near(first_measured, first_reference);
    failures += expect_near(second_measured, second_reference);
    failures += expect_near(third_measured, third_reference);
    failures += expect_near(fourth_measured, fourth_reference);

    auto distances_array = distances<caps_>(
        kernel_, query_view, query_norms,
        {first_candidate_view, second_candidate_view, third_candidate_view, fourth_candidate_view},
        {first_candidate_norms, second_candidate_norms, third_candidate_norms, fourth_candidate_norms});
    failures += expect_near(distances_array[0], first_reference);
    failures += expect_near(distances_array[1], second_reference);
    failures += expect_near(distances_array[2], third_reference);
    failures += expect_near(distances_array[3], fourth_reference);
    return failures;
}

/** No-op sibling picked when no tile engine exists for the tuple. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
static int check_tiles() noexcept {
    return 0;
}

/** A tiles kit against the 4-way path as the oracle: one tile of queries plus a ragged tail packs
 *  once, stored rows stream against the pack where they lie, and every raw dot, finalized by the
 *  quads kit under the runtime metric, must match the 1-to-1 distance. Skips silently when the
 *  machine lacks the engine at runtime. */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires tiles_supported<kernel_, scalar_, caps_>
static int check_tiles() noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k || (available & caps_) != caps_) return 0;
    if (nk_cpu_configure_thread(available) != nk_success_k) return 0;

    using tiles_kit_t = typename dense_tiles_chain<scalar_, caps_>::kit_t;
    using quads_kit_t = typename dense_quads_chain<scalar_, caps_>::kit_t;
    using raw_dot_t = typename scalar_::dot_result_t::raw_t;
    constexpr std::size_t queries_k = tiles_kit_t::tile_columns_k + 3, rows_k = 37, dimensions_k = 128;

    auto storage = nk::matrix<scalar_>::zeros({queries_k + rows_k, dimensions_k}).value;
    for (std::size_t row = 0; row != queries_k + rows_k; ++row) fill_cyclic(storage.row(row).as_vector(), row);
    nk::matrix_view<scalar_> const queries = storage(nk::range(std::size_t {0}, queries_k), nk::slice);
    nk::matrix_view<scalar_> const rows = storage(nk::range(queries_k, queries_k + rows_k), nk::slice);
    std::size_t const stride = static_cast<std::size_t>(queries.stride_bytes(0));
    std::array<typename scalar_::dot_result_t, queries_k> query_sumsqs;
    for (std::size_t query = 0; query != queries_k; ++query)
        query_sumsqs[query] = dense_norms<caps_>(queries.row(query).as_vector()).sumsq;

    nk_size_t packed_bytes = 0;
    int failures = expect(tiles_kit_t::pack_size(queries_k, dimensions_k, &packed_bytes) == nk_success_k);
    auto packed = nk::vector<nk::u8_t>::zeros(packed_bytes).value;
    failures += expect(tiles_kit_t::pack(queries.raw_values_data(), queries_k, dimensions_k, stride, packed.data(), 0,
                                         queries_k, nullptr) == nk_success_k);
    auto dots = nk::matrix<raw_dot_t>::zeros({rows_k, queries_k}).value;
    failures += expect(tiles_kit_t::dots_packed(rows.raw_values_data(), packed.data(), dots.data(), rows_k, queries_k,
                                                dimensions_k, stride, queries_k * sizeof(raw_dot_t),
                                                nullptr) == nk_success_k);

    for (std::size_t row = 0; row != rows_k; ++row) {
        auto const row_vector = rows.row(row).as_vector();
        auto const row_norms = dense_norms<caps_>(row_vector);
        std::span<raw_dot_t const> const row_dots = to_span(dots.row(row).as_vector());
        for (std::size_t first = 0; first < queries_k; first += 4) {
            std::size_t const lanes = min_of<std::size_t>(4, queries_k - first);
            typename quads_kit_t::finalized_vec_t lane_dots {}, lane_sumsqs {};
            std::memcpy(&lane_dots, row_dots.subspan(first, lanes).data(), lanes * sizeof(raw_dot_t));
            std::memcpy(&lane_sumsqs, std::span {query_sumsqs}.subspan(first, lanes).data(), lanes * sizeof(raw_dot_t));
            std::array<f32_t, 4> const tiled = convert_to_distances<quads_kit_t>(kernel_, lane_dots, row_norms.sumsq,
                                                                                 lane_sumsqs);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                auto const query_vector = queries.row(first + lane).as_vector();
                failures += expect_near(tiled[lane], distance<caps_>(kernel_, row_vector, row_norms, query_vector,
                                                                     dense_norms<caps_>(query_vector)));
            }
        }
    }
    return failures;
}

template <nk_kernel_kind_t kernel_, typename scalar_>
static int check_kernel() noexcept {
    int failures = 0;
    for (std::size_t const requested : dimensions_to_test) {
        // Packed scalars hold whole bytes of dimensions, so a ragged request rounds up to the next byte.
        std::size_t const dim = round_up_to_multiple<std::size_t>(requested, nk::dimensions_per_value<scalar_>());
        // Newest silicon first, so a reader sees the strongest tier at the top: x86, then Arm,
        // each by generation, then the remaining architectures.
        failures += check_quads<kernel_, scalar_, nk_cap_diamond_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_sierra_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_genoa_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_alder_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_icelake_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_skylake_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_haswell_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_neonfp8_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_neonbfdot_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_neonfhm_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_neonsdot_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_neon_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_v128relaxed_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_powervsx_k>(dim);
        failures += check_quads<kernel_, scalar_, nk_cap_serial_k>(dim);
    }
    failures += check_tiles<kernel_, scalar_, nk_cap_graniteamx_k>();
    failures += check_tiles<kernel_, scalar_, nk_cap_sapphireamx_k>();
    failures += check_tiles<kernel_, scalar_, nk_cap_smebi32_k>();
    failures += check_tiles<kernel_, scalar_, nk_cap_sme_k>();
    return failures;
}

template <typename scalar_>
static int check_scalar() noexcept {
    return check_kernel<nk_kernel_dot_k, scalar_>() +     //
           check_kernel<nk_kernel_angular_k, scalar_>() + //
           check_kernel<nk_kernel_euclidean_k, scalar_>();
}

/** Binary inputs run dot/hamming/jaccard instead of dot/angular/euclidean. */
template <>
int check_scalar<nk::u1x8_t>() noexcept {
    return check_kernel<nk_kernel_dot_k, nk::u1x8_t>() +     //
           check_kernel<nk_kernel_hamming_k, nk::u1x8_t>() + //
           check_kernel<nk_kernel_jaccard_k, nk::u1x8_t>();
}

/** Every scalar below must resolve at least at the serial tier. Without this the
 *  no-op overload would swallow a dtype whose chain specialization went missing, and the
 *  suite would stay green while covering nothing. */
template <typename scalar_>
concept scalar_wired = dense_quads_supported<nk_kernel_dot_k, scalar_, nk_cap_serial_k>;

/** Per-dtype metric suite entries the shard translation units define and
 *  @c dense.cpp composes explicitly; each returns its shard's failure count. */
std::size_t check_ram_metric_f64();
std::size_t check_ram_metric_f32();
std::size_t check_ram_metric_bf16();
std::size_t check_ram_metric_f16();
std::size_t check_ram_metric_e5m2();
std::size_t check_ram_metric_e4m3();
std::size_t check_ram_metric_e3m2();
std::size_t check_ram_metric_e2m3();
std::size_t check_ram_metric_i8();
std::size_t check_ram_metric_i4();
std::size_t check_ram_metric_u8();
std::size_t check_ram_metric_u4();
std::size_t check_ram_metric_u1();

} // namespace unum::usearch::test::dense
