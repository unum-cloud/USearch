/**
 *  @file test/dense/ram_metric.cpp
 *  @author Ash Vardanian
 *  @date April 21, 2026
 *  @brief The metric checks of NumKong's own reference scalars, f64 and f32, which the library
 *      stores in no dtype unit and so compiles whatever @c USEARCH_DENSE_DTYPES cuts.
 */
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

std::size_t check_ram_metric(settings_t const& settings) {
    return check_named_scalar<nk::f64_t>(settings, "ram_metric f64") +
           check_named_scalar<nk::f32_t>(settings, "ram_metric f32");
}

} // namespace unum::usearch::test::dense
