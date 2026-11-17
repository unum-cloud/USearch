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
#undef NDEBUG // ! A test's oracle must stay live in every build
#include "ram_index.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static void run_ram_u8_cases(gpu_device_t const&) {
    run_test_case("test_ram_metric<nk::u8_t>", test_ram_metric<nk::u8_t>);
    constexpr nk_capability_t tiles_caps_k = static_cast<nk_capability_t>( //
        nk_cap_serial_k | nk_cap_icelake_k | nk_cap_sapphireamx_k);
    run_test_case("test_ram_index_configuration<nk_u8_k, nk_kernel_euclidean_k, fu::standard_yield_t, tiles_caps_k>",
                  test_ram_index_configuration<nk_u8_k, nk_kernel_euclidean_k, fu::standard_yield_t, tiles_caps_k>,
                  test_run.settings.seed);
}

static bool const ram_u8_registered = (registered_suites({gpu_vendor_t::unknown_k, &run_ram_u8_cases}), true);

} // namespace unum::usearch::test::dense
