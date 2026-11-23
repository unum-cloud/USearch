/**
 *  @file test/dense/ram_i8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The i8 cells of the CPU-resident dense residency: the metric and the sharded suite under
 *      Icelake's kits.
 *
 *  Icelake's i8 finalize subtracts the row sums, so even a dot index saves its norms column.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps i8, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_indexes.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_i8_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::i8_t>", test_ram_metric<nk::i8_t>);
    run_test_case(
        "test_ram_indexes_configuration<nk_i8_k, nk_kernel_dot_k, fu::standard_yield_t, cpu_class_t::intel_icelake_k>",
        test_ram_indexes_configuration<nk_i8_k, nk_kernel_dot_k, fu::standard_yield_t, cpu_class_t::intel_icelake_k>,
        test_run.settings.seed);
    run_test_case("test_ram_scan_configuration<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>",
                  test_ram_scan_configuration<nk_i8_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>,
                  test_run.settings.seed);
}

static bool const ram_i8_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_i8_cases}), true);

} // namespace unum::usearch::test::dense
