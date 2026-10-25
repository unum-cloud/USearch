/**
 *  @file test/dense/ram_e3m2.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The e3m2 cells of the CPU-resident dense residency: the metric alone.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps e3m2, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_e3m2(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_e3m2")) return 0;
    std::size_t failures_count = check_named_scalar<nk::e3m2_t>(settings, "ram_e3m2 metric");
    return failures_count;
}

static bool const ram_e3m2_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_e3m2}), true);

} // namespace unum::usearch::test::dense
