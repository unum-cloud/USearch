/**
 *  @file bench/dense_streaming.cpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The dataset streaming of @c dense.cuh through NumKong's C++ SDK, linked beside the units
 *      that compile NumKong header-only for their kits, which cannot call its verbs.
 */
#include <cmath> // `std::abs` for the rescale amplitude

#include <numkong/cast.hpp>   // `nk::cast`, `nk::to_f32`
#include <numkong/each.hpp>   // `nk::scale`
#include <numkong/reduce.hpp> // `nk::reduce_minmax`

#include "dense.cuh"

namespace unum::usearch::bench {

/** NumKong's @c component_max of @p scalar_type_ - the largest magnitude one stored dimension
 *  holds - extended to the plain scalars, each its own component, whose largest magnitude is their
 *  @c finite_max value. */
template <typename scalar_type_>
float component_max() noexcept {
    if constexpr (requires { scalar_type_::component_max(); }) return float(scalar_type_::component_max());
    else return nk::to_f32(scalar_type_::finite_max());
}

template <typename from_type_, typename to_type_>
nk::expected<float> downcast_scale(nk::vector_view<from_type_> payload) noexcept {
    auto const [extremes, status] = nk::reduce_minmax<from_type_>(payload);
    if (nk::failed(status)) return {1.0f, status};
    float const max_amplitude = util::max_of(std::abs(nk::to_f32(extremes.min_value)),
                                             std::abs(nk::to_f32(extremes.max_value)));
    // The stored dtype's largest magnitude; not a constant expression for nvcc's byte floats.
    float const target_amplitude = component_max<to_type_>();
    return {max_amplitude > 0 ? target_amplitude / max_amplitude : 1.0f, nk::status_t::success_k};
}

template <typename from_type_, typename to_type_>
nk::status_t stage_row(nk::vector_view<from_type_> source, float scale, nk::vector_span<nk::f32_t> rescaled,
                       nk::vector_span<to_type_> staged) noexcept {
    if (rescaled.empty()) return nk::cast<from_type_, to_type_>(source, staged);
    nk::status_t status = nk::status_t::success_k;
    // An f32 source feeds `nk::scale` straight from the mapping, skipping the widening cast.
    if constexpr (std::is_same_v<from_type_, nk::f32_t>) status = nk::scale<nk::f32_t>(source, scale, 0.0f, rescaled);
    else if (nk::succeeded(status = nk::cast<from_type_, nk::f32_t>(source, rescaled)))
        status = nk::scale<nk::f32_t>(rescaled, scale, 0.0f, rescaled);
    if (nk::succeeded(status)) status = nk::cast<nk::f32_t, to_type_>(rescaled, staged);
    return status;
}

/** The one pair @c dense_ram_index_search.cpp stages: an f32 file into bf16 storage, no rescale. */
template nk::status_t stage_row<nk::f32_t, nk::bf16_t>(nk::vector_view<nk::f32_t>, float, nk::vector_span<nk::f32_t>,
                                                       nk::vector_span<nk::bf16_t>) noexcept;

} // namespace unum::usearch::bench
