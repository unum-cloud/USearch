/**
 *  @brief Concurrency & misuse tests for the CPU-resident @c dense_ram_indexes, repeated
 *      under a deliberately flaky waiter to widen the interleaving windows.
 *  @author Ash Vardanian
 *  @file test/dense/ram_indexes.cpp
 *  @date August 31, 2026
 */
#include "dense/ram_indexes.hpp"

using namespace unum::usearch;

std::size_t test_dense_indexes() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular");
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>("sharded bf16+angular+flaky");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k>("sharded i8+euclidean");
    return failures_count;
}
