/**
 *  @file test/dense/ram_index_i8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the CPU-resident @c dense::ram_index over i8 cells,
 *      repeated under a deliberately flaky waiter to widen the interleaving windows.
 */
#include "dense/ram_index.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_index_i8(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_index_configuration<nk_i8_k, nk_kernel_dot_k>("i8+dot", settings.seed);
    failures_count += test_ram_index_configuration<nk_i8_k, nk_kernel_euclidean_k>("i8+euclidean", settings.seed);
    failures_count += test_ram_index_configuration<nk_i8_k, nk_kernel_euclidean_k, flaky_waiter_t>("i8+euclidean+flaky",
                                                                                                   settings.seed);
    failures_count += test_ram_index_configuration<nk_i8_k, nk_kernel_euclidean_k, fu::standard_yield_t,
                                                   nk_cap_serial_k, fu::preferred_atomic_ref>(
        "i8+euclidean+preferred-atomics", settings.seed);
    failures_count +=
        test_ram_index_configuration<nk_i8_k, nk_kernel_euclidean_k, fu::standard_yield_t, nk_cap_neonsdot_k>( //
            "i8+euclidean+neonsdot", settings.seed);
    return failures_count;
}
