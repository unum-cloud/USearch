/**
 *  @file test/dense/nvme_indexes.cpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief One bf16 cell of @c dense::nvme_indexes per I/O queue the build holds: every read verb
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
        expect_equal(medium_status, status_code_t::success_k);
        linux_uring_files_queue<fu::standard_yield_t> queue {forkunion_allocator_t {}};
        io_queue_config_t config;
        config.transport = io_transport_t::linux_uring_k, config.queue_depth = 2;
        if (failed(queue.open(config)))
            return case_result_t::skipped("linux_uring_files_queue is refused on this machine");
        expect_equal(queue.reserve(2, buffer_bytes), status_code_t::success_k);
        expect_true(queue.buffer_registration() == linux_uring_buffer_registration_t::fixed_k);
        expect_equal(queue.register_media(std::span<file_medium_t const> {&medium, 1}, media_list_key_t {1, 0}),
                     status_code_t::success_k);
        expect_equal(queue.enqueue_read(io_request_t {0, static_cast<u32_t>(io_alignment_k), 0, 1, 7}),
                     status_code_t::success_k);
        expect_equal(queue.submit(io_wait_t::blocking_k), status_code_t::success_k);
        io_completion_t completion {};
        expect_equal(queue.poll_completions(std::span<io_completion_t> {&completion, 1}), 1u);
        expect_equal(completion.result, static_cast<i32_t>(io_alignment_k));
        expect_true(std::memcmp(queue.buffer(1), written.data(), written.size()) == 0);
    }
    return {};
}

/** A list keyed without an identity answers uninitialized, as a reshard target missing its draw
 *  would; a second list read after a first reads its own file, not the first one the ring still
 *  holds, as a reshard's targets read after its sources; and the ring grows to a deeper call once
 *  and keeps that depth through a shallower one. */
static case_result_t test_uring_media_keys() {
    using executor_t = nvme_executor<linux_uring_files_queue<fu::standard_yield_t>>;
    scratch_directory_t const scratch;
    std::array<file_medium_t, 2> media;
    for (std::size_t list = 0; list != media.size(); ++list) {
        std::string const path = scratch.path(fmt::format("list-{}", list));
        std::vector<byte_t> const written(io_alignment_k, static_cast<byte_t>(list + 1));
        auto [file, open_status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k);
        expect_true(succeeded(open_status) && succeeded(file.write_at(0, written)) && succeeded(file.flush()));
        auto [medium, medium_status] = file_medium_t::open(path);
        expect_equal(medium_status, status_code_t::success_k);
        media[list] = std::move(medium);
    }
    io_queue_config_t config;
    config.transport = io_transport_t::linux_uring_k;
    auto [threads, threads_status] = executor_t::threads_t::on_calling_thread();
    expect_equal(threads_status, status_code_t::success_k);
    auto [executor, executor_status] = executor_t::acquire(std::move(threads), config);
    if (failed(executor_status)) return case_result_t::skipped("linux_uring_files_queue is refused on this machine");

    // The first byte of the list holding medium `list` alone, or the status its call answered.
    auto first_byte = [&](std::size_t list, media_list_key_t key, std::size_t depth) noexcept {
        auto [call, call_status] = nvme_begin_call(executor, std::span<file_medium_t const> {&media[list], 1}, key,
                                                   depth, 0, 1, io_alignment_k);
        if (failed(call_status)) return expected<byte_t> {byte_t {}, call_status};
        status_t read = status_code_t::success_k;
        status_t const dispatched = executor.for_threads([&](std::size_t thread, std::size_t) noexcept {
            executor_t::queue_t& queue = executor.queue(thread);
            io_completion_t completion {};
            read = queue.enqueue_read(io_request_t {0, static_cast<u32_t>(io_alignment_k), 0, 0, 0});
            if (succeeded(read)) read = queue.submit(io_wait_t::blocking_k);
            if (succeeded(read) && queue.poll_completions(std::span<io_completion_t> {&completion, 1}) != 1)
                read = status_code_t::io_k;
        });
        if (failed(dispatched) || failed(read)) return expected<byte_t> {byte_t {}, failed(read) ? read : dispatched};
        return expected<byte_t> {static_cast<byte_t>(*executor.queue(0).buffer(0)), status_code_t::success_k};
    };
    expect_equal(first_byte(0, media_list_key_t {}, 1).status, status_code_t::uninitialized_k);
    auto const [first, first_status] = first_byte(0, media_list_key_t {3, 0}, 2);
    expect_equal(first_status, status_code_t::success_k);
    expect_equal(static_cast<unsigned>(first), 1u);
    expect_equal(executor.queue_depth(), 2u);
    auto const [second, second_status] = first_byte(1, media_list_key_t {5, 0}, 1);
    expect_equal(second_status, status_code_t::success_k);
    expect_equal(static_cast<unsigned>(second), 2u);
    expect_equal(executor.queue_depth(), 2u);
    return {};
}

} // namespace unum::usearch::test::dense

#endif

void test::dense::run_nvme_indexes_cases() {
    u64_t const seed = test_run.settings.seed;
    run_test_case("test_nvme_configuration<blocking_files_queue<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_configuration<blocking_files_queue<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>, seed,
                  "blocking_files_queue is refused on this machine");
#if USEARCH_WITH_LINUX_AIO
    run_test_case("test_nvme_configuration<linux_aio_files_queue<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_configuration<linux_aio_files_queue<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>,
                  seed, "linux_aio_files_queue is refused on this machine");
#endif
#if USEARCH_WITH_LINUX_URING
    run_test_case("test_nvme_configuration<linux_uring_files_queue<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_configuration<linux_uring_files_queue<fu::standard_yield_t>, nk_bf16_k, nk_kernel_dot_k>,
                  seed, "linux_uring_files_queue is refused on this machine");
    run_test_case("test_uring_registered_pieces", test_uring_registered_pieces);
    run_test_case("test_uring_media_keys", test_uring_media_keys);
#endif
#if USEARCH_WITH_LINUX_SPDK
    run_test_case("test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>",
                  test_nvme_spdk_configuration<nk_bf16_k, nk_kernel_dot_k>, seed,
                  std::span<std::string_view const> {test_run.settings.nvme_paths});
#endif
}
