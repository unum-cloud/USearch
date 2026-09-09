/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over i8 cells,
 *      repeated under a deliberately flaky waiter to widen the interleaving windows.
 *  @author Ash Vardanian
 *  @file test/dense/ram_index_i8.cpp
 *  @date July 23, 2026
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test_dense_index_i8() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k>("i8+euclidean");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k, flaky_waiter_t>("i8+euclidean+flaky");
    failures_count +=
        test_configuration<nk_i8_k, nk_kernel_euclidean_k, dummy_waiter_t, nk_cap_serial_k, fu::preferred_atomic_ref>(
            "i8+euclidean+preferred-atomics");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k, dummy_waiter_t, nk_cap_neonsdot_k>( //
        "i8+euclidean+neonsdot");
    return failures_count;
}
