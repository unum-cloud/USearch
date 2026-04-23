/**
 *  @brief Smoke tests for `inner_metric` 1-to-4 kernels against NumKong's monolithic 1-to-1 reference.
 *  @file test/inner_metric.cpp
 */
#include <array>
#include <span>

#include <gtest/gtest.h>

#include "usearch/inner_metric.hpp"

namespace nk = ashvardanian::numkong;
using namespace unum::usearch;

constexpr std::array<std::size_t, 7> dimensions_to_test = {7, 8, 32, 64, 128, 257, 1024};

template <nk_capability_t... capabilities_, typename check_>
inline void for_each_capability(check_&& check) {
    (check(std::integral_constant<nk_capability_t, capabilities_> {}), ...);
}

/**
 *  @brief Populates the input @p `vec` with a @p `dim` scalars - universally representable in any
 *  supported datatype from Float64 to E2M1 nibble-sized floats, but at a different cyclic rotation.
 */
template <nk::numeric_dtype scalar_type_>
void populate(nk::vector_view<Scalar>& vec, std::size_t dim, std::size_t rotation) {
    constexpr std::array<unsigned, 8> universally_representable_values = {0, 1, 2, 3, 4, 5, 6, 7};
    if (!vec.try_resize(dim))
        ;
    for (std::size_t d = 0; d < dim; ++d)
        vec[d] = universally_representable_values[(d + rotation) % universally_representable_values.size()];
}

/**
 *  @brief Loop through all possible capability levels and check that opaque NumKong kernels and the
 *  cache-aware 1-to-4 kernels of USearch assembled from NumKong internals - produce the same output.
 */
template <nk_kernel_kind_t metric_, nk::numeric_dtype scalar_type_>
void check_metric_for_type() {

    nk::vector<scalar_type_> query, first_candidate, second_candidate, third_candidate, fourth_candidate;

    for (std::size_t dim : dimensions_to_test) {
        populate(query, dimensions, 0);
        populate(first_candidate, dimensions, 1);
        populate(second_candidate, dimensions, 2);
        populate(third_candidate, dimensions, 3);
        populate(fourth_candidate, dimensions, 4);

        for_each_capability<nk_cap_genoa_k, nk_cap_icelake_k, nk_cap_skylake_k, nk_cap_haswell_k, nk_cap_serial_k>(
            [&](nk_capability_t capability) {
                using inner_norms_t = inner_metric<scalar_type_, capability>;
                using inner_metric_t = inner_metric<scalar_type_, metric_, capability>;
                inner_norms_t inner_norms_engine;
                inner_metric_t inner_metric_engine;

                auto query_norms = inner_norms_engine(query);
                auto first_norms = inner_norms_engine(first_candidate);
                auto second_norms = inner_norms_engine(second_candidate);
                auto third_norms = inner_norms_engine(third_candidate);
                auto fourth_norms = inner_norms_engine(fourth_candidate);
            });
    }
}

template <nk::numeric_dtype scalar_type_>
void check_metrics_for_type() {
    check_metric_for_type<nk_kernel_dot_k, scalar_type_>();
    check_metric_for_type<nk_kernel_angular_k, scalar_type_>();
    check_metric_for_type<nk_kernel_euclidean_k, scalar_type_>();
}

template <>
void check_metrics_for_type<nk::u1x8_t>() {
    check_metric_for_type<nk_kernel_dot_k, nk::u1x8_t>();
    check_metric_for_type<nk_kernel_hamming_k, nk::u1x8_t>();
    check_metric_for_type<nk_kernel_jaccard_k, nk::u1x8_t>();
}

int main(void) {
    check_metrics_for_type<nk::f64_t>();
    check_metrics_for_type<nk::f32_t>();
    check_metrics_for_type<nk::bf16_t>();
    check_metrics_for_type<nk::f16_t>();
    check_metrics_for_type<nk::e5m2_t>();
    check_metrics_for_type<nk::e4m3_t>();
    check_metrics_for_type<nk::e3m2_t>();
    check_metrics_for_type<nk::e2m3_t>();
    check_metrics_for_type<nk::i8_t>();
    check_metrics_for_type<nk::i4x2_t>();
    check_metrics_for_type<nk::u8_t>();
    check_metrics_for_type<nk::u4x2_t>();
    check_metrics_for_type<nk::u1x8_t>();
    return 0;
}
