/**
 *  @file test/dense/ram_u1.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The u1 cells of the CPU-resident dense residency: the metric and the full single-index
 *      suite under the bit kernels.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps u1, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_index.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_u1_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::u1x8_t>", test_ram_metric<nk::u1x8_t>);
    run_test_case("test_ram_index_configuration<nk_u1_k, nk_kernel_hamming_k>",
                  test_ram_index_configuration<nk_u1_k, nk_kernel_hamming_k>, test_run.settings.seed);
}

static bool const ram_u1_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_u1_cases}), true);

} // namespace unum::usearch::test::dense
