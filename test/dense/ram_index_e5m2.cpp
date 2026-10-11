/**
 *  @file test/dense/ram_index_e5m2.cpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over e5m2 cells.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_e5m2(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_index_configuration<nk_e5m2_k, nk_kernel_angular_k>("e5m2+angular", settings.seed);
    return failures_count;
}
