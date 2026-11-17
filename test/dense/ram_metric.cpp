/**
 *  @file test/dense/ram_metric.cpp
 *  @author Ash Vardanian
 *  @date April 21, 2026
 *  @brief The metric checks of NumKong's own reference scalars, f64 and f32, which the library
 *      stores in no dtype unit and so compiles whatever @c USEARCH_DENSE_DTYPES cuts.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

void run_ram_metric_cases() {
    run_test_case("test_ram_metric<nk::f64_t>", test_ram_metric<nk::f64_t>);
    run_test_case("test_ram_metric<nk::f32_t>", test_ram_metric<nk::f32_t>);
}

} // namespace unum::usearch::test::dense
