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
#include "ram_index.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_u1(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_u1")) return 0;
    std::size_t failures_count = check_named_scalar<nk::u1x8_t>(settings, "ram_u1 metric");
    failures_count += test_ram_index_configuration<nk_u1_k, nk_kernel_hamming_k>("ram_u1 hamming", settings.seed);
    return failures_count;
}

static bool const ram_u1_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_u1}), true);

} // namespace unum::usearch::test::dense
