/**
 *  @file test/dense/nvme_indexes.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief One bf16 cell of @c dense::nvme_indexes per I/O engine the build carries: every read verb
 *      against the RAM group its parts came from and every rewrite against a fresh RAM build.
 */
#include "nvme_indexes.hpp"

using namespace unum::usearch;

std::size_t test::dense::check_nvme_indexes(settings_t const& settings) {
    // The family's name selects every cell, and a cell's own name selects it alone.
    auto const selects = [&](char const* name) {
        return settings.filter.selects("nvme_indexes") || settings.filter.selects(name);
    };
    std::size_t failures_count = 0;
    if (char const* name = "nvme bf16+dot+blocking"; selects(name))
        failures_count += test_nvme_configuration<blocking_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(name,
                                                                                                       settings.seed);
#if USEARCH_WITH_LINUX_AIO
    if (char const* name = "nvme bf16+dot+linux-aio"; selects(name))
        failures_count += test_nvme_configuration<linux_aio_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(name,
                                                                                                        settings.seed);
#endif
#if USEARCH_WITH_LINUX_URING
    if (char const* name = "nvme bf16+dot+linux-uring"; selects(name))
        failures_count += test_nvme_configuration<linux_uring_files_engine<>, nk_bf16_k, nk_kernel_dot_k>(
            name, settings.seed);
#endif
#if USEARCH_WITH_LINUX_SPDK
    if (char const* name = "nvme bf16+dot+linux-spdk"; selects(name))
        failures_count += test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>(name, settings.seed,
                                                                                   settings.nvme_paths);
#endif
    return failures_count;
}
