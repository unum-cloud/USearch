/**
 *  @file test/dense/nvme_indexes.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief One bf16 cell of @c dense::nvme_indexes per I/O engine the build carries: every read verb
 *      against the RAM group its parts came from and every rewrite against a fresh RAM build.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
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
static case_result_t test_uring_registered_pieces() {
    // Two buffers of half a GiB and a page, so the second one opens the second piece.
    constexpr std::size_t buffer_bytes = (std::size_t {1} << 29) + io_alignment_k;
    struct rlimit pinnable {};
    if (::getrlimit(RLIMIT_MEMLOCK, &pinnable) != 0 ||
        (pinnable.rlim_cur != RLIM_INFINITY && pinnable.rlim_cur < 2 * buffer_bytes + (std::size_t {1} << 20)))
        return case_result_t::skipped("needs RLIMIT_MEMLOCK");
    scratch_directory_t const scratch;
    std::string const path = scratch.path("pieces");
    std::vector<byte_t> written(io_alignment_k);
    for (std::size_t offset = 0; offset != written.size(); ++offset) written[offset] = static_cast<byte_t>(offset * 7);
    {
        auto [file, open_status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k);
        expect_true(succeeded(open_status) && succeeded(file.write_at(0, written)) && succeeded(file.flush()));
    }
    {
        auto [medium, medium_status] = file_medium_t::open(path);
        expect_equal(medium_status, status_t::success_k);
        linux_uring_files_engine<fu::standard_yield_t> engine {forkunion_allocator_t {}};
        io_engine_config_t config;
        config.transport = io_transport_t::linux_uring_k, config.queue_depth = 2;
        if (failed(engine.open(config)))
            return case_result_t::skipped("linux_uring_files_engine is refused on this machine");
        expect_equal(engine.reserve(2, buffer_bytes), status_t::success_k);
        expect_true(engine.buffer_registration() == linux_uring_buffer_registration_t::fixed_k);
        expect_equal(engine.attach(std::span<file_medium_t const> {&medium, 1}), status_t::success_k);
        expect_equal(engine.enqueue_read(io_request_t {0, static_cast<u32_t>(io_alignment_k), 0, 1, 7}),
                     status_t::success_k);
        expect_equal(engine.submit(io_wait_t::blocking_k), status_t::success_k);
        io_completion_t completion {};
        expect_equal(engine.poll_completions(std::span<io_completion_t> {&completion, 1}), 1u);
        expect_equal(completion.result, static_cast<i32_t>(io_alignment_k));
        expect_true(std::memcmp(engine.buffer(1), written.data(), written.size()) == 0);
        engine.detach();
    }
    return {};
}

} // namespace unum::usearch::test::dense

#endif

void test::dense::run_nvme_indexes_cases() {
    u64_t const seed = test_run.settings.seed;
    run_test_case("test_nvme_configuration<blocking_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_configuration<blocking_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>,
                  seed, "blocking_files_engine is refused on this machine");
#if USEARCH_WITH_LINUX_AIO
    run_test_case("test_nvme_configuration<linux_aio_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_configuration<linux_aio_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>,
                  seed, "linux_aio_files_engine is refused on this machine");
#endif
#if USEARCH_WITH_LINUX_URING
    run_test_case("test_nvme_configuration<linux_uring_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_configuration<linux_uring_files_engine<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>,
                  seed, "linux_uring_files_engine is refused on this machine");
    run_test_case("test_uring_registered_pieces", test_uring_registered_pieces);
#endif
#if USEARCH_WITH_LINUX_SPDK
    run_test_case("test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>, seed,
                  std::span<std::string_view const> {test_run.settings.nvme_paths});
#endif
}
