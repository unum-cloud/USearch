/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over i8 cells,
 *      repeated under a deliberately flaky waiter to widen the interleaving windows.
 *  @author Ash Vardanian
 *  @file test/dense_index_i8.cpp
 *  @date July 23, 2026
 */
#include "dense_index.hpp"

using namespace unum::usearch;

std::size_t test_dense_index_i8() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k>("i8+euclidean");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k, flaky_waiter_t>("i8+euclidean+flaky");
    return failures_count;
}
