/**
 *  @file test/dense/nvme_indexes.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief One bf16 cell of @c dense::nvme_indexes per I/O engine the build carries: every read verb
 *      against the RAM group its parts came from and every rewrite against a fresh RAM build.
 */
#include "nvme_indexes.hpp"

#if USEARCH_WITH_LINUX_URING
#include <sys/resource.h> // `getrlimit`, `RLIMIT_MEMLOCK` - whether the pieces may be pinned at all
#endif

using namespace unum::usearch;

#if USEARCH_WITH_LINUX_URING

namespace unum::usearch::test::dense {

/** A staging region past the kernel's 1 GiB cap per registered buffer registers in whole-buffer
 *  pieces, and a fixed read into the second piece lands as a plain read would; skipped where the
 *  memlock limit could not pin the region at all. */
static std::size_t test_uring_registered_pieces(char const* name) {
    // Two buffers of half a GiB and a page, so the second one opens the second piece.
    constexpr std::size_t buffer_bytes = (std::size_t {1} << 29) + io_alignment_k;
    struct rlimit pinnable {};
    if (::getrlimit(RLIMIT_MEMLOCK, &pinnable) != 0 ||
        (pinnable.rlim_cur != RLIM_INFINITY && pinnable.rlim_cur < 2 * buffer_bytes + (std::size_t {1} << 20)))
        return report(name, "RLIMIT_MEMLOCK below the GiB the region pins");
    std::size_t failures = 0;
    std::string const path = temporary_path(fmt::format("usearch-uring-pieces-{:x}", ::getpid()));
    std::vector<byte_t> written(io_alignment_k);
    for (std::size_t offset = 0; offset != written.size(); ++offset) written[offset] = static_cast<byte_t>(offset * 7);
    {
        auto [file, open_status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k);
        failures += expect(succeeded(open_status) && succeeded(file.write_at(0, written)) && succeeded(file.flush()));
    }
    {
        auto [medium, medium_status] = file_medium_t::open(path);
        failures += expect_eq(medium_status, status_t::success_k);
        linux_uring_files_engine<fu::standard_yield_t> engine {forkunion_allocator_t {}};
        io_engine_config_t config;
        config.transport = io_transport_t::linux_uring_k, config.queue_depth = 2;
        if (failed(engine.open(config)))
            return std::remove(path.c_str()), report(name, "io_uring refused by this host");
        failures += expect_eq(engine.reserve(2, buffer_bytes), status_t::success_k);
        failures += expect(engine.buffer_registration() == linux_uring_buffer_registration_t::fixed_k);
        failures += expect_eq(engine.attach(std::span<file_medium_t const> {&medium, 1}), status_t::success_k);
        failures += expect_eq(engine.enqueue_read(io_request_t {0, static_cast<u32_t>(io_alignment_k), 0, 1, 7}),
                              status_t::success_k);
        failures += expect_eq(engine.submit(io_wait_t::blocking_k), status_t::success_k);
        io_completion_t completion {};
        failures += expect_eq(engine.poll_completions(std::span<io_completion_t> {&completion, 1}), 1u);
        failures += expect_eq(completion.result, static_cast<i32_t>(io_alignment_k));
        failures += expect(std::memcmp(engine.buffer(1), written.data(), written.size()) == 0);
        engine.detach();
    }
    std::remove(path.c_str());
    return report(name, failures);
}

} // namespace unum::usearch::test::dense

#endif

std::size_t test::dense::check_nvme_indexes(settings_t const& settings) {
    // The family's name selects every cell, and a cell's own name selects it alone.
    auto const selects = [&](char const* name) {
        return settings.filter.selects("nvme_indexes") || settings.filter.selects(name);
    };
    std::size_t failures_count = 0;
    if (char const* name = "nvme bf16+dot+blocking"; selects(name))
        failures_count +=
            test_nvme_configuration<blocking_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>(
                name, settings.seed);
#if USEARCH_WITH_LINUX_AIO
    if (char const* name = "nvme bf16+dot+linux-aio"; selects(name))
        failures_count +=
            test_nvme_configuration<linux_aio_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>(
                name, settings.seed);
#endif
#if USEARCH_WITH_LINUX_URING
    if (char const* name = "nvme bf16+dot+linux-uring"; selects(name))
        failures_count +=
            test_nvme_configuration<linux_uring_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>(
                name, settings.seed);
    if (char const* name = "nvme linux-uring registered pieces"; selects(name))
        failures_count += test_uring_registered_pieces(name);
#endif
#if USEARCH_WITH_LINUX_SPDK
    if (char const* name = "nvme bf16+dot+linux-spdk"; selects(name))
        failures_count += test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>(name, settings.seed,
                                                                                   settings.nvme_paths);
#endif
    return failures_count;
}
