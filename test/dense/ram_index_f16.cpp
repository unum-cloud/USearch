/**
 *  @file test/dense/ram_index_f16.cpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over f16 cells.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_f16(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_f16_k, nk_kernel_angular_k>("f16+angular", settings.seed);
    return failures_count;
}
