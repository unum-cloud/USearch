/**
 *  @file test/dense/ram_index_bf16.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense_ram_index over bf16 cells,
 *      repeated under a deliberately flaky waiter to widen the interleaving windows.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_bf16(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_index_configuration<nk_bf16_k, nk_kernel_dot_k>("bf16+dot", settings.seed);
    failures_count += test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k>("bf16+angular", settings.seed);
    failures_count += test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>("bf16+angular+flaky",
                                                                                                   settings.seed);
    failures_count +=
        test_ram_index_configuration<nk_bf16_k, nk_kernel_angular_k, dummy_waiter_t, nk_cap_neonbfdot_k>( //
            "bf16+angular+neonbfdot", settings.seed);
    return failures_count;
}
