/**
 *  @file test/dense/ram_index_i4.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over packed i4 cells.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_i4(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_i4_k, nk_kernel_angular_k>("i4+angular", settings.seed);
    return failures_count;
}
