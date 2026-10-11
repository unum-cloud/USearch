/**
 *  @file test/dense/ram_index_f32.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over f32 cells.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_f32(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_index_configuration<nk_f32_k, nk_kernel_dot_k>("f32+dot", settings.seed);
    failures_count += test_ram_index_configuration<nk_f32_k, nk_kernel_angular_k>("f32+angular", settings.seed);
    failures_count += test_ram_index_configuration<nk_f32_k, nk_kernel_euclidean_k>("f32+euclidean", settings.seed);
    return failures_count;
}
