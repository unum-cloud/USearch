/**
 *  @file test/dense/ram_indexes.cpp
 *  @author Ash Vardanian
 *  @date August 31, 2026
 *  @brief Concurrency & misuse tests for the CPU-resident @c dense_ram_indexes, repeated under a
 *      deliberately flaky waiter to widen the interleaving windows.
 */
#include "dense/ram_indexes.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_indexes() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular");
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>("sharded bf16+angular+flaky");
    failures_count +=
        test_configuration<nk_bf16_k, nk_kernel_angular_k, dummy_waiter_t, nk_cap_serial_k, fu::preferred_atomic_ref>(
            "sharded bf16+angular+preferred-atomics");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k>("sharded i8+euclidean");
    return failures_count;
}
