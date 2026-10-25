/**
 *  @file test/dense/ram_u8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The u8 cells of the CPU-resident dense residency: the metric and the full single-index
 *      suite over the AMX tiles.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps u8, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#include "ram_index.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_u8(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_u8")) return 0;
    std::size_t failures_count = check_named_scalar<nk::u8_t>(settings, "ram_u8 metric");
    constexpr nk_capability_t tiles_caps_k = static_cast<nk_capability_t>( //
        nk_cap_serial_k | nk_cap_icelake_k | nk_cap_sapphireamx_k);
    failures_count += test_ram_index_configuration<nk_u8_k, nk_kernel_euclidean_k, fu::standard_yield_t, tiles_caps_k>(
        "ram_u8 euclidean+amx", settings.seed);
    return failures_count;
}

static bool const ram_u8_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_u8}), true);

} // namespace unum::usearch::test::dense
