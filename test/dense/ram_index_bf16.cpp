/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over bf16 cells,
 *      repeated under a deliberately flaky waiter to widen the interleaving windows.
 *  @author Ash Vardanian
 *  @file test/dense/ram_index_bf16.cpp
 *  @date July 23, 2026
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test_dense_index_bf16() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k>("bf16+angular");
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>("bf16+angular+flaky");
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k, dummy_waiter_t, nk_cap_neonbfdot_k>( //
        "bf16+angular+neonbfdot");
    return failures_count;
}
