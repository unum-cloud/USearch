/**
 *  @brief Smoke tests for `inner_metric` 1-to-1 and 1-to-4 kernels against NumKong's
 *      monolithic 1-to-1 reference (`nk::dot` / `nk::angular` / `nk::sqeuclidean`).
 *
 *  @author Ash Vardanian
 *  @file test/inner_metric.cpp
 *  @date April 21, 2026
 *
 *  USearch's @c distance and @c distances entry points compose a kit (per scalar +
 *  capability) with a metric overlay (per kernel) and route through a single trampoline.
 *  The test:
 *
 *  1. Enumerates every NumKong scalar in @c main and every @c (kernel, scalar, capability)
 *     tuple inside @c check_metrics_for_type. The @c kernel_supported concept gates each
 *     call so combinations not yet adapted are no-ops at compile time without breaking
 *     the build - new kits and chains light up the matching cells automatically.
 *  2. For each supported tuple and a range of dimensions, computes a baseline distance
 *     with NumKong's monolithic API (@c nk::dot returns the raw inner product, so we
 *     wrap to @c 1 - dot for the dot kernel; @c nk::sqeuclidean is compared to USearch's
 *     squared euclidean), then calls @c distance (1-to-1) and @c distances (1-to-4) and
 *     compares with a relative tolerance.
 *  3. Returns non-zero exit code if any comparison fails. Each check returns its own
 *     failure count; counts roll up to @c main without any global mutable state.
 *
 *  The input vectors are populated with cyclic values from {0..7} - representable
 *  exactly in every supported scalar (Float64..E2M1, i8/u8, i4/u4) - at four different
 *  cyclic rotations so the four candidates differ from the query and from each other.
 */
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>

#include <fmt/format.h>

#include <numkong/dot.hpp>
#include <numkong/set.hpp>
#include <numkong/spatial.hpp>
#include <numkong/vector.hpp>

#include "test.hpp"
#include "usearch/inner_metric.hpp"

namespace nk = ashvardanian::numkong;
using namespace unum::usearch;

constexpr std::array<std::size_t, 7> dimensions_to_test = {7, 8, 32, 64, 128, 257, 1024};

/**
 *  @brief Checks whether @c kernel_chain has a specialization with a @c kit_t alias for
 *  this @c (kernel,scalar,caps) tuple - unadapted tuples are skipped at compile time.
 */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
concept kernel_supported = requires { typename kernel_chain<kernel_, scalar_, caps_>::kit_t; };

/**
 *  @brief Populates the input @p vec with small scalars - universally representable in any
 *  supported datatype from Float64 to E2M1 nibble-sized floats, but at a different cyclic rotation.
 */
template <typename scalar_>
void populate(nk::vector_span<scalar_> vec, std::size_t rotation) noexcept {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    for (std::size_t dimension_index = 0; dimension_index < vec.size(); ++dimension_index)
        vec[dimension_index] = make_scalar<scalar_>(
            universally_representable_values[(dimension_index + rotation) % universally_representable_values.size()]);
}

/**
 *  @brief Computes the reference distance via NumKong's monolithic 1-to-1 API, adjusted
 *  so the value matches USearch's convention - dot returns (1-inner), angular returns (1-cos_sim)
 *  clamped to [0,2], euclidean is sqrt'd via @c nk::euclidean.
 */
template <nk_kernel_kind_t kernel_, typename scalar_>
f32_t reference_distance(nk::vector_view<scalar_> first_vector, nk::vector_view<scalar_> second_vector) noexcept {
    if constexpr (kernel_ == nk_kernel_dot_k) {
        typename scalar_::dot_result_t inner_product {};
        nk::dot(first_vector, second_vector, first_vector.size(), &inner_product);
        return f32_t(1) - static_cast<f32_t>(inner_product);
    }
    else if constexpr (kernel_ == nk_kernel_angular_k) {
        typename scalar_::angular_result_t angular_value {};
        nk::angular(first_vector, second_vector, first_vector.size(), &angular_value);
        return static_cast<f32_t>(angular_value);
    }
    else if constexpr (kernel_ == nk_kernel_euclidean_k) {
        typename scalar_::euclidean_result_t euclidean_value {};
        nk::euclidean(first_vector, second_vector, first_vector.size(), &euclidean_value);
        return static_cast<f32_t>(euclidean_value);
    }
    else if constexpr (kernel_ == nk_kernel_hamming_k) {
        typename scalar_::hamming_result_t hamming_value {};
        nk::hamming(first_vector, second_vector, first_vector.size(), &hamming_value);
        return static_cast<f32_t>(hamming_value);
    }
    else if constexpr (kernel_ == nk_kernel_jaccard_k) {
        typename scalar_::jaccard_result_t jaccard_value {};
        nk::jaccard(first_vector, second_vector, first_vector.size(), &jaccard_value);
        return static_cast<f32_t>(jaccard_value);
    }
}

bool close_enough(f32_t actual, f32_t expected) noexcept {
    f32_t const absolute_difference = std::fabs(actual - expected);
    f32_t const scale = std::fmax(std::fmax(std::fabs(actual), std::fabs(expected)), 1.0f);
    return absolute_difference <= 1e-3f * scale;
}

/**
 *  @brief Compares one (actual, expected) pair - prints a structured failure line and
 *  returns 1 on mismatch, 0 on match. Caller accumulates the count.
 */
int check_close(f32_t actual, f32_t expected, std::string_view label) noexcept {
    if (close_enough(actual, expected)) return 0;
    fmt::print(stderr, "FAIL {:<22}  actual={:>14.6g}  expected={:>14.6g}  diff={:>12.4g}\n", label, actual, expected,
               std::fabs(actual - expected));
    return 1;
}

/**
 *  @brief No-op overload picked when the (kernel, scalar, caps) tuple isn't wired up -
 *  lets the caller enumerate the full capability list without if-constexpr gates inline.
 */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires(!kernel_supported<kernel_, scalar_, caps_>)
int check_one(std::size_t /*dim*/) noexcept {
    return 0;
}

/**
 *  @brief Runs one (kernel, scalar, caps, dim) check - returns the number of mismatches
 *  across both the 1-to-1 and 1-to-4 paths. Body instantiates only for adapted tuples.
 */
template <nk_kernel_kind_t kernel_, typename scalar_, nk_capability_t caps_>
    requires kernel_supported<kernel_, scalar_, caps_>
int check_one(std::size_t dim) noexcept {
    auto query = nk::vector<scalar_>::try_zeros(dim);
    auto first_candidate = nk::vector<scalar_>::try_zeros(dim);
    auto second_candidate = nk::vector<scalar_>::try_zeros(dim);
    auto third_candidate = nk::vector<scalar_>::try_zeros(dim);
    auto fourth_candidate = nk::vector<scalar_>::try_zeros(dim);
    populate(query.span(), 0);
    populate(first_candidate.span(), 1);
    populate(second_candidate.span(), 2);
    populate(third_candidate.span(), 3);
    populate(fourth_candidate.span(), 4);

    auto query_view = query.view();
    auto first_candidate_view = first_candidate.view();
    auto second_candidate_view = second_candidate.view();
    auto third_candidate_view = third_candidate.view();
    auto fourth_candidate_view = fourth_candidate.view();

    auto query_norms = inner_norms<caps_>(query_view);
    auto first_candidate_norms = inner_norms<caps_>(first_candidate_view);
    auto second_candidate_norms = inner_norms<caps_>(second_candidate_view);
    auto third_candidate_norms = inner_norms<caps_>(third_candidate_view);
    auto fourth_candidate_norms = inner_norms<caps_>(fourth_candidate_view);

    f32_t const first_reference = reference_distance<kernel_>(query_view, first_candidate_view);
    f32_t const second_reference = reference_distance<kernel_>(query_view, second_candidate_view);
    f32_t const third_reference = reference_distance<kernel_>(query_view, third_candidate_view);
    f32_t const fourth_reference = reference_distance<kernel_>(query_view, fourth_candidate_view);

    int failures = 0;
    failures += check_close(
        distance<kernel_, caps_>(query_view, query_norms, first_candidate_view, first_candidate_norms), first_reference,
        "1-to-1 lane 0");
    failures += check_close(
        distance<kernel_, caps_>(query_view, query_norms, second_candidate_view, second_candidate_norms),
        second_reference, "1-to-1 lane 1");
    failures += check_close(
        distance<kernel_, caps_>(query_view, query_norms, third_candidate_view, third_candidate_norms), third_reference,
        "1-to-1 lane 2");
    failures += check_close(
        distance<kernel_, caps_>(query_view, query_norms, fourth_candidate_view, fourth_candidate_norms),
        fourth_reference, "1-to-1 lane 3");

    auto distances_array = distances<kernel_, caps_>(
        query_view, query_norms,
        {first_candidate_view, second_candidate_view, third_candidate_view, fourth_candidate_view},
        {first_candidate_norms, second_candidate_norms, third_candidate_norms, fourth_candidate_norms});
    failures += check_close(distances_array[0], first_reference, "1-to-4 lane 0");
    failures += check_close(distances_array[1], second_reference, "1-to-4 lane 1");
    failures += check_close(distances_array[2], third_reference, "1-to-4 lane 2");
    failures += check_close(distances_array[3], fourth_reference, "1-to-4 lane 3");
    return failures;
}

template <nk_kernel_kind_t kernel_, typename scalar_>
int check_metric_for_type() noexcept {
    int failures = 0;
    for (std::size_t dim : dimensions_to_test) {
        failures += check_one<kernel_, scalar_, nk_cap_genoa_k>(dim);
        failures += check_one<kernel_, scalar_, nk_cap_icelake_k>(dim);
        failures += check_one<kernel_, scalar_, nk_cap_skylake_k>(dim);
        failures += check_one<kernel_, scalar_, nk_cap_haswell_k>(dim);
        failures += check_one<kernel_, scalar_, nk_cap_serial_k>(dim);
    }
    return failures;
}

template <typename scalar_>
int check_metrics_for_type() noexcept {
    return check_metric_for_type<nk_kernel_dot_k, scalar_>() +     //
           check_metric_for_type<nk_kernel_angular_k, scalar_>() + //
           check_metric_for_type<nk_kernel_euclidean_k, scalar_>();
}

/** @brief Binary inputs run dot/hamming/jaccard instead of dot/angular/euclidean. */
template <>
int check_metrics_for_type<nk::u1x8_t>() noexcept {
    return check_metric_for_type<nk_kernel_dot_k, nk::u1x8_t>() +     //
           check_metric_for_type<nk_kernel_hamming_k, nk::u1x8_t>() + //
           check_metric_for_type<nk_kernel_jaccard_k, nk::u1x8_t>();
}

int main(void) {
    int failures = 0;

    failures += check_metrics_for_type<nk::f64_t>();
    failures += check_metrics_for_type<nk::f32_t>();
    failures += check_metrics_for_type<nk::bf16_t>();
    failures += check_metrics_for_type<nk::f16_t>();
    failures += check_metrics_for_type<nk::e5m2_t>();
    failures += check_metrics_for_type<nk::e4m3_t>();
    failures += check_metrics_for_type<nk::e3m2_t>();
    failures += check_metrics_for_type<nk::e2m3_t>();

    failures += check_metrics_for_type<nk::i8_t>();
    failures += check_metrics_for_type<nk::i4x2_t>();

    failures += check_metrics_for_type<nk::u8_t>();
    failures += check_metrics_for_type<nk::u4x2_t>();
    failures += check_metrics_for_type<nk::u1x8_t>();

    if (failures != 0) fmt::print(stderr, "{} failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
