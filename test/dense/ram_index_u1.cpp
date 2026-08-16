/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over packed
 *      u1 cells, under the bit kernels.
 *  @author Ash Vardanian
 *  @file test/dense/ram_index_u1.cpp
 *  @date August 15, 2026
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test_dense_index_u1() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_u1_k, nk_kernel_hamming_k>("u1+hamming");
    failures_count += test_configuration<nk_u1_k, nk_kernel_jaccard_k>("u1+jaccard");
    return failures_count;
}
