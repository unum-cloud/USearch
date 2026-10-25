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
#include "ram_index.hpp"
#include "ram_indexes.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_bf16(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_bf16")) return 0;
    std::size_t failures_count = check_named_scalar<nk::bf16_t>(settings, "ram_bf16 metric");
    failures_count += test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k>("ram_bf16 angular", settings.seed);
    failures_count += test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>(
        "ram_bf16 angular+flaky", settings.seed);
    failures_count += test_starved_configuration<nk_bf16_k, nk_kernel_angular_k>("ram_bf16 angular+starved",
                                                                                 settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>("ram_bf16 sharded angular",
                                                                                     settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>(
        "ram_bf16 sharded angular+flaky", settings.seed);
    // The tiled scans: Arm SME and Intel AMX, each skipped where the build or the host lacks it.
    failures_count += test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::apple_m4_k>(
        "ram_bf16 sharded angular+sme", settings.seed);
    failures_count += test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_sapphire_k>(
        "ram_bf16 sharded angular+amx", settings.seed);
    return failures_count;
}

static bool const ram_bf16_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_bf16}), true);

} // namespace unum::usearch::test::dense
