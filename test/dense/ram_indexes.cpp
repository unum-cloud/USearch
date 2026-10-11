/**
 *  @file test/dense/ram_indexes.cpp
 *  @author Ash Vardanian
 *  @date August 31, 2026
 *  @brief Concurrency & misuse tests for the CPU-resident @c dense_ram_indexes, repeated under a
 *      deliberately flaky waiter to widen the interleaving windows.
 */
#include "dense/ram_indexes.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_indexes(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular",
                                                                                     settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>(
        "sharded bf16+angular+flaky", settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, dummy_waiter_t, nk_cap_serial_k,
                                                     fu::preferred_atomic_ref>("sharded bf16+angular+preferred-atomics",
                                                                               settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_dot_k>("sharded bf16+dot", settings.seed);
    failures_count += test_ram_indexes_configuration<nk_i8_k, nk_kernel_dot_k>("sharded i8+dot", settings.seed);
    failures_count += test_ram_indexes_configuration<nk_i8_k, nk_kernel_euclidean_k>("sharded i8+euclidean",
                                                                                     settings.seed);
    return failures_count;
}
