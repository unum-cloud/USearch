/**
 *  @file test/dense/ram_index_u8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense::ram_index over u8 cells,
 *      including the tiled matrix-engine tier where the machine offers it.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_u8(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_index_configuration<nk_u8_k, nk_kernel_euclidean_k>("u8+euclidean", settings.seed);
    failures_count += test_ram_index_configuration<nk_u8_k, nk_kernel_angular_k>("u8+angular", settings.seed);
    failures_count +=
        test_ram_index_configuration<nk_u8_k, nk_kernel_euclidean_k, fu::standard_yield_t, nk_cap_neonsdot_k>( //
            "u8+euclidean+neonsdot", settings.seed);

    // The tiled u8 tier: the serial floor, icelake quads, and sapphireamx tiles.
    constexpr nk_capability_t tiles_caps_k = static_cast<nk_capability_t>( //
        nk_cap_serial_k | nk_cap_icelake_k | nk_cap_sapphireamx_k);
    failures_count +=
        test_ram_index_configuration<nk_u8_k, nk_kernel_euclidean_k, fu::standard_yield_t, tiles_caps_k>( //
            "u8+euclidean+tiles", settings.seed);

    // The starved u8 cell: ledgered scratch, so a refused re-table has somewhere to come from.
    failures_count += test_starved_configuration<nk_u8_k, nk_kernel_euclidean_k>("u8+euclidean+starved", settings.seed);
    return failures_count;
}
