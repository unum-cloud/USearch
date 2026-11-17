/**
 *  @file test/dense/ram_e3m2.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The e3m2 cells of the CPU-resident dense residency: the metric alone.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps e3m2, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_e3m2_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::e3m2_t>", test_ram_metric<nk::e3m2_t>);
}

static bool const ram_e3m2_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_e3m2_cases}), true);

} // namespace unum::usearch::test::dense
