/**
 *  @file test/dense/ram_index_u1.cpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over packed u1 cells,
 *      under the bit kernels.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_u1(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_index_configuration<nk_u1_k, nk_kernel_hamming_k>("u1+hamming", settings.seed);
    failures_count += test_ram_index_configuration<nk_u1_k, nk_kernel_jaccard_k>("u1+jaccard", settings.seed);
    return failures_count;
}
