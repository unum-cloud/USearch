/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over f16 cells.
 *  @author Ash Vardanian
 *  @file test/dense/ram_index_f16.cpp
 *  @date August 15, 2026
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test_dense_index_f16() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_f16_k, nk_kernel_angular_k>("f16+angular");
    return failures_count;
}
