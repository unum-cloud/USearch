/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over packed i4 cells.
 *  @author Ash Vardanian
 *  @file test/dense_index_i4.cpp
 *  @date July 23, 2026
 */
#include "dense_index.hpp"
#include "test.hpp"

std::size_t test_dense_index_i4() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_i4_k, nk_kernel_angular_k>("i4+angular");
    return failures_count;
}
