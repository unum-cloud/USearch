/**
 *  @file test/dense/ram_i4.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The i4 cells of the CPU-resident dense residency: the metric alone.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps i4, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_i4(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_i4")) return 0;
    std::size_t failures_count = check_named_scalar<nk::i4x2_t>(settings, "ram_i4 metric");
    return failures_count;
}

static bool const ram_i4_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_i4}), true);

} // namespace unum::usearch::test::dense
