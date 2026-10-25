/**
 *  @file test/dense/ram_e2m3.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The e2m3 cells of the CPU-resident dense residency: the metric alone.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps e2m3, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_e2m3(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_e2m3")) return 0;
    std::size_t failures_count = check_named_scalar<nk::e2m3_t>(settings, "ram_e2m3 metric");
    return failures_count;
}

static bool const ram_e2m3_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_e2m3}), true);

} // namespace unum::usearch::test::dense
