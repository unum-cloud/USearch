/**
 *  @file bench/dense/ram_metric_numkong.cpp
 *  @author Ash Vardanian
 *  @date April 23, 2026
 *  @brief The NumKong baseline rows of @c ram_metric.cpp, calling NumKong's C++ SDK, linked beside
 *      that unit, which compiles NumKong header-only for the kits and cannot call its verbs.
 */
#include <cstddef> // `std::size_t`

#include <numkong/dot.hpp>     // `nk::dot`
#include <numkong/set.hpp>     // `nk::hamming`, `nk::jaccard`
#include <numkong/spatial.hpp> // `nk::angular`, `nk::euclidean`

#include "dense_kits.hpp" // `dense_dtypes_t` - the storage dtypes the library compiles, in ladder order; `bytes_view_t`

namespace unum::usearch::bench {
using utilities::bytes_view_t;

/** NumKong's own @p kernel between the @p dimensions in @p first_bytes and @p second_bytes; a
 *  kernel the scalar lacks answers @c missing_kernel_k. */
template <typename scalar_type_>
nk::status_t numkong_distance_of(nk_kernel_kind_t kernel, bytes_view_t first_bytes, bytes_view_t second_bytes,
                                 std::size_t dimensions) noexcept {
    nk::vector_view<scalar_type_> const first {reinterpret_cast<scalar_type_ const*>(first_bytes.data()), dimensions};
    nk::vector_view<scalar_type_> const second {reinterpret_cast<scalar_type_ const*>(second_bytes.data()), dimensions};
    if constexpr (requires { nk::hamming<scalar_type_>(first, second); }) {
        if (kernel == nk_kernel_hamming_k) return nk::hamming<scalar_type_>(first, second).status;
        if (kernel == nk_kernel_jaccard_k) return nk::jaccard<scalar_type_>(first, second).status;
    }
    if constexpr (requires { nk::angular<scalar_type_>(first, second); }) {
        if (kernel == nk_kernel_angular_k) return nk::angular<scalar_type_>(first, second).status;
        if (kernel == nk_kernel_euclidean_k) return nk::euclidean<scalar_type_>(first, second).status;
    }
    if (kernel == nk_kernel_dot_k) return nk::dot<scalar_type_>(first, second).status;
    return nk::status_t::missing_kernel_k;
}

/** Picks the scalar of @p dtype among @p dtypes_; one outside them answers @c missing_kernel_k. */
template <nk_dtype_t... dtypes_>
nk::status_t numkong_distance_in(dtype_list<dtypes_...>, nk_kernel_kind_t kernel, nk_dtype_t dtype, bytes_view_t first,
                                 bytes_view_t second, std::size_t dimensions) noexcept {
    nk::status_t status = nk::status_t::missing_kernel_k;
    [[maybe_unused]] bool const picked =
        ((dtype == dtypes_ &&
          (status = numkong_distance_of<typename nk::type_for<dtypes_>::type>(kernel, first, second, dimensions),
           true)) ||
         ...);
    return status;
}

nk::status_t numkong_distance(nk_kernel_kind_t kernel, nk_dtype_t dtype, bytes_view_t first, bytes_view_t second,
                              std::size_t dimensions) noexcept {
    // NumKong's own references first, then the storage dtypes the library compiles.
    if (dtype == nk_f64_k || dtype == nk_f32_k)
        return numkong_distance_in(dtype_list<nk_f64_k, nk_f32_k> {}, kernel, dtype, first, second, dimensions);
    return numkong_distance_in(dense_dtypes_t {}, kernel, dtype, first, second, dimensions);
}

} // namespace unum::usearch::bench
