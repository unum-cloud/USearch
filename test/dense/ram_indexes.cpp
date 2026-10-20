/**
 *  @file test/dense/ram_indexes.cpp
 *  @author Ash Vardanian
 *  @date August 31, 2026
 *  @brief Concurrency & misuse tests for the CPU-resident @c dense::ram_indexes, repeated under
 *      a deliberately flaky waiter to widen the interleaving windows.
 */
#include "dense/ram_indexes.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_ram_indexes(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular",
                                                                                     settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>(
        "sharded bf16+angular+flaky", settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, fu::standard_yield_t,
                                                     nk_cap_serial_k, fu::preferred_atomic_ref>(
        "sharded bf16+angular+preferred-atomics", settings.seed);
    failures_count += test_ram_indexes_configuration<nk_bf16_k, nk_kernel_dot_k>("sharded bf16+dot", settings.seed);
    // The tiled scans: Arm SME and Intel AMX, each skipped where the build or the host lacks it.
    constexpr nk_capability_t sme_caps_k = static_cast<nk_capability_t>(nk_cap_serial_k | nk_cap_sme_k);
    constexpr nk_capability_t amx_caps_k = static_cast<nk_capability_t>(nk_cap_serial_k | nk_cap_sapphireamx_k);
    failures_count += test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, sme_caps_k>(
        "sharded bf16+angular+sme", settings.seed);
    failures_count += test_ram_scan_configuration<nk_bf16_k, nk_kernel_angular_k, amx_caps_k>(
        "sharded bf16+angular+amx", settings.seed);
#if USEARCH_DENSE_DTYPES >= 3
    failures_count += test_ram_indexes_configuration<nk_i8_k, nk_kernel_dot_k>("sharded i8+dot", settings.seed);
    // Icelake's i8 finalize subtracts the row sums, so even a dot index saves its norms column.
    failures_count += test_ram_indexes_configuration<nk_i8_k, nk_kernel_dot_k, fu::standard_yield_t, nk_cap_icelake_k>(
        "sharded i8+dot+icelake", settings.seed);
    failures_count += test_ram_indexes_configuration<nk_i8_k, nk_kernel_euclidean_k>("sharded i8+euclidean",
                                                                                     settings.seed);
#endif
    return failures_count;
}
