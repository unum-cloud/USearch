/**
 *  @file test/dense/nvme_indexes.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Every read verb of @c dense::nvme_indexes against the RAM group its parts came from,
 *      and every rewrite against a fresh RAM build of its rows, once per I/O engine the host
 *      admits.
 */
#include "nvme_indexes.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_nvme_indexes(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_nvme_configuration<blocking_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+blocking", settings.seed);
#if USEARCH_DENSE_DTYPES >= 3
    failures_count += test_nvme_configuration<blocking_files_engine<>, nk_i8_k, nk_kernel_angular_k>(
        "nvme i8+angular+blocking", settings.seed);
#endif
    failures_count += test_nvme_uuid_configuration<blocking_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+blocking+uuid", settings.seed);
    failures_count += test_nvme_widened_keys<blocking_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+blocking+widened", settings.seed);
    // The tiled scans: Arm SME and Intel AMX, each skipped where the build or the host lacks it.
    constexpr nk_capability_t sme_caps_k = static_cast<nk_capability_t>(nk_cap_serial_k | nk_cap_sme_k);
    constexpr nk_capability_t amx_caps_k = static_cast<nk_capability_t>(nk_cap_serial_k | nk_cap_sapphireamx_k);
    failures_count += test_nvme_scan_configuration<blocking_files_engine<>, nk_bf16_k, nk_kernel_angular_k, sme_caps_k>(
        "nvme bf16+angular+blocking+sme", settings.seed);
    failures_count += test_nvme_scan_configuration<blocking_files_engine<>, nk_bf16_k, nk_kernel_angular_k, amx_caps_k>(
        "nvme bf16+angular+blocking+amx", settings.seed);
#if USEARCH_WITH_LINUX_AIO
    failures_count += test_nvme_configuration<linux_aio_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+linux-aio", settings.seed);
#if USEARCH_DENSE_DTYPES >= 3
    failures_count += test_nvme_configuration<linux_aio_files_engine<>, nk_i8_k, nk_kernel_angular_k>(
        "nvme i8+angular+linux-aio", settings.seed);
#endif
    failures_count += test_nvme_uuid_configuration<linux_aio_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+linux-aio+uuid", settings.seed);
#endif // USEARCH_WITH_LINUX_AIO
#if USEARCH_WITH_LINUX_URING
    failures_count += test_nvme_configuration<linux_uring_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+linux-uring", settings.seed);
#if USEARCH_DENSE_DTYPES >= 3
    failures_count += test_nvme_configuration<linux_uring_files_engine<>, nk_i8_k, nk_kernel_angular_k>(
        "nvme i8+angular+linux-uring", settings.seed);
#endif
    failures_count += test_nvme_uuid_configuration<linux_uring_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
        "nvme bf16+dot+linux-uring+uuid", settings.seed);
#endif // USEARCH_WITH_LINUX_URING
#if USEARCH_WITH_LINUX_SPDK
    failures_count += test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>("nvme bf16+dot+linux-spdk",
                                                                               settings.seed);
#endif
    return failures_count;
}

std::size_t test::dense::check_nvme_rewrites(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_nvme_rewrites_configuration<blocking_files_engine<>, nk_bf16_k, nk_kernel_angular_k>(
        "rewrites bf16+angular+blocking", settings.seed);
#if USEARCH_WITH_LINUX_AIO
    failures_count += test_nvme_rewrites_configuration<linux_aio_files_engine<>, nk_bf16_k, nk_kernel_angular_k>(
        "rewrites bf16+angular+linux-aio", settings.seed);
#endif
#if USEARCH_WITH_LINUX_URING
    failures_count += test_nvme_rewrites_configuration<linux_uring_files_engine<>, nk_bf16_k, nk_kernel_angular_k>(
        "rewrites bf16+angular+linux-uring", settings.seed);
#endif
    return failures_count;
}
