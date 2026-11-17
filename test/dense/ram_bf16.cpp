/**
 *  @file test/dense/ram_bf16.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The bf16 cells of the CPU-resident dense residency.
 *
 *  The metric, the full single-index suite over the allocation ledger, again under a flaky waiter
 *  and over starved scratch, the sharded suite at the serial floor and under a flaky waiter, and
 *  the SME and AMX scans.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps bf16, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_index.hpp"
#include "ram_indexes.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_bf16_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::bf16_t>", test_ram_metric<nk::bf16_t>);
    u64_t const seed = test_run.settings.seed;
    run_test_case("test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k>",
                  test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k>, seed);
    run_test_case("test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>",
                  test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>, seed);
    run_test_case("test_starved_configuration<nk_bf16_k, nk_kernel_angular_k>",
                  test_starved_configuration<nk_bf16_k, nk_kernel_angular_k>, seed);
    run_test_case("test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>",
                  test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>, seed);
    run_test_case("test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>",
                  test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>, seed);
    // The tiled scans: Arm SME, Intel AMX and IBM MMA, each skipped where build or host lacks it.
    run_test_case("test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m4_k>",
                  test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m4_k>, seed);
    run_test_case("test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_sapphire_k>",
                  test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_sapphire_k>, seed);
    run_test_case("test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::ibm_power10_k>",
                  test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::ibm_power10_k>, seed);
}

static bool const ram_bf16_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_bf16_cases}), true);

} // namespace unum::usearch::test::dense
