/**
 *  @file test/dense/ram_f16.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The f16 cells of the CPU-resident dense residency: the metric alone.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps f16, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_f16_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::f16_t>", test_ram_metric<nk::f16_t>);
}

static bool const ram_f16_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_f16_cases}), true);

} // namespace unum::usearch::test::dense
