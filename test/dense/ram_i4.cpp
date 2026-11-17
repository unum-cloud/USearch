/**
 *  @file test/dense/ram_i4.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The i4 cells of the CPU-resident dense residency: the metric alone.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps i4, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_i4_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::i4x2_t>", test_ram_metric<nk::i4x2_t>);
}

static bool const ram_i4_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_i4_cases}), true);

} // namespace unum::usearch::test::dense
