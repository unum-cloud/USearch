/**
 *  @file test/dense/vram_nvidia.cpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA suite's entry for the one VRAM runner: selects the device, runs every dtype
 *      cell and the sharded suite the settings select, then device attention, whose value lookups
 *      run on the device.
 *
 *  Clang compiles it host-only into the cells' index types, which the shared shards see only through
 *  the faces; NVCC compiles each dtype's kernels and the attends over the suite's value lookups
 *  from @c vram_nvidia_<dtype>.cu into the fatbins this unit embeds.
 */
#include <fmt/format.h>

#include "usearch/dense/vram_nvidia_index.cuh"

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::dense {

using test::dense::counting_cuda_allocator_t;
using test::dense::dense_nvidia_fatbin;

// The ledgered index of every NVIDIA cell and the collections the sharded suite drives.
template struct vram_nvidia_index<nk_bf16_k, dense_nvidia_fatbin<nk_bf16_k>, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
template struct vram_nvidia_indexes<u32_t, nk_bf16_k, dense_nvidia_fatbin<nk_bf16_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct vram_nvidia_index<nk_u1_k, dense_nvidia_fatbin<nk_u1_k>, dummy_cache_hints_t, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct vram_nvidia_index<nk_i8_k, dense_nvidia_fatbin<nk_i8_k>, dummy_cache_hints_t, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, cuda_device_allocator_t>;
template struct vram_nvidia_indexes<u32_t, nk_i8_k, dense_nvidia_fatbin<nk_i8_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct vram_nvidia_index<nk_e4m3_k, dense_nvidia_fatbin<nk_e4m3_k>, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct vram_nvidia_index<nk_f16_k, dense_nvidia_fatbin<nk_f16_k>, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct vram_nvidia_index<nk_u8_k, dense_nvidia_fatbin<nk_u8_k>, dummy_cache_hints_t, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct vram_nvidia_index<nk_e5m2_k, dense_nvidia_fatbin<nk_e5m2_k>, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct vram_nvidia_index<nk_i4_k, dense_nvidia_fatbin<nk_i4_k>, dummy_cache_hints_t, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct vram_nvidia_index<nk_u4_k, dense_nvidia_fatbin<nk_u4_k>, dummy_cache_hints_t, counting_cuda_allocator_t,
                                  counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct vram_nvidia_index<nk_e3m2_k, dense_nvidia_fatbin<nk_e3m2_k>, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct vram_nvidia_index<nk_e2m3_k, dense_nvidia_fatbin<nk_e2m3_k>, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
} // namespace unum::usearch::dense

namespace unum::usearch::test::dense {

/** Each dtype's fatbin, linked in as raw bytes beside the suite's host half. */
extern "C" byte_t const test_vram_nvidia_bf16_bundle_start[], test_vram_nvidia_bf16_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_bf16_k>::bytes() noexcept {
    return {test_vram_nvidia_bf16_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_bf16_bundle_end - test_vram_nvidia_bf16_bundle_start)};
}
#if USEARCH_DENSE_DTYPES >= 2
extern "C" byte_t const test_vram_nvidia_u1_bundle_start[], test_vram_nvidia_u1_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_u1_k>::bytes() noexcept {
    return {test_vram_nvidia_u1_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_u1_bundle_end - test_vram_nvidia_u1_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern "C" byte_t const test_vram_nvidia_i8_bundle_start[], test_vram_nvidia_i8_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_i8_k>::bytes() noexcept {
    return {test_vram_nvidia_i8_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_i8_bundle_end - test_vram_nvidia_i8_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern "C" byte_t const test_vram_nvidia_e4m3_bundle_start[], test_vram_nvidia_e4m3_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_e4m3_k>::bytes() noexcept {
    return {test_vram_nvidia_e4m3_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_e4m3_bundle_end - test_vram_nvidia_e4m3_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern "C" byte_t const test_vram_nvidia_f16_bundle_start[], test_vram_nvidia_f16_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_f16_k>::bytes() noexcept {
    return {test_vram_nvidia_f16_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_f16_bundle_end - test_vram_nvidia_f16_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern "C" byte_t const test_vram_nvidia_u8_bundle_start[], test_vram_nvidia_u8_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_u8_k>::bytes() noexcept {
    return {test_vram_nvidia_u8_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_u8_bundle_end - test_vram_nvidia_u8_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern "C" byte_t const test_vram_nvidia_e5m2_bundle_start[], test_vram_nvidia_e5m2_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_e5m2_k>::bytes() noexcept {
    return {test_vram_nvidia_e5m2_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_e5m2_bundle_end - test_vram_nvidia_e5m2_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern "C" byte_t const test_vram_nvidia_i4_bundle_start[], test_vram_nvidia_i4_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_i4_k>::bytes() noexcept {
    return {test_vram_nvidia_i4_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_i4_bundle_end - test_vram_nvidia_i4_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern "C" byte_t const test_vram_nvidia_u4_bundle_start[], test_vram_nvidia_u4_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_u4_k>::bytes() noexcept {
    return {test_vram_nvidia_u4_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_u4_bundle_end - test_vram_nvidia_u4_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern "C" byte_t const test_vram_nvidia_e3m2_bundle_start[], test_vram_nvidia_e3m2_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_e3m2_k>::bytes() noexcept {
    return {test_vram_nvidia_e3m2_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_e3m2_bundle_end - test_vram_nvidia_e3m2_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern "C" byte_t const test_vram_nvidia_e2m3_bundle_start[], test_vram_nvidia_e2m3_bundle_end[];
template <>
bytes_view_t dense_nvidia_fatbin<nk_e2m3_k>::bytes() noexcept {
    return {test_vram_nvidia_e2m3_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_e2m3_bundle_end - test_vram_nvidia_e2m3_bundle_start)};
}
#endif

/** A value table copied onto the device, its rows a 16-byte pitch apart; null rows when the copy
 *  failed. */
struct dense_cuda_value_table_t {
    using lookup_t = cuda_test_lookup_t;

    cuda_unique_ptr<byte_t> rows;
    std::size_t pitch;

    /** Copies @p values onto the device of @p stream as the table every attend lookup in these
     *  suites reads. */
    template <typename scalar_type_>
    static dense_cuda_value_table_t upload(nk::matrix_view<scalar_type_> values,
                                           cuda_executor_t::stream_t const& stream) {
        std::size_t const rows_count = values.extent(0);
        std::size_t const row_bytes = divide_round_up<std::size_t>(
            values.extent(1) * nk_dtype_bits(scalar_type_::dtype()), 8);
        std::size_t const pitch = round_up_to_multiple<std::size_t>(row_bytes, 16);
        dense_cuda_value_table_t table {cuda_make_unique(cuda_device_allocator_t {stream}, rows_count * pitch), pitch};
        if (table.rows && (cuda_copy_2d_async(table.rows.get(), pitch, values.byte_data(), values.stride_bytes(0),
                                              row_bytes, rows_count, stream.handle) != cuda_success_k ||
                           cuda_api().stream_synchronize(stream.handle) != cuda_success_k))
            table.rows.reset();
        return table;
    }

    /** The device lookup over the first @p count rows. */
    lookup_t lookup(std::size_t count) const noexcept { return {rows.get(), pitch, count}; }
};

/** Device attention over every dtype cell the settings select, then over the sharded collection,
 *  all on @p executor 's device. */
static std::size_t check_vram_nvidia_attend(settings_t const& settings, cuda_executor_t& executor) {
    std::size_t failures_count = check_vram_attend_cells<dense_cuda_value_table_t>(settings, executor);
    if (settings.selects("vram_indexes"))
        failures_count += report(
            "cuda-sharded bf16+attend",
            dense_expect_vram_indexes_attend<vram_nvidia_indexes<u32_t, nk_bf16_k, dense_nvidia_fatbin<nk_bf16_k>>,
                                             dense_cuda_value_table_t>(settings.seed, executor));
    return failures_count;
}

std::size_t check_vram_nvidia(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    if (cuda_set_device(ordinal) != cuda_success_k)
        return fmt::println("- CUDA: device #{} does not answer", ordinal), 1;
    fmt::println("- CUDA: {} {} #{}, {} SMs, {} MiB", device.name,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)), ordinal,
                 device.compute_units, device.memory_bytes >> 20);
    auto [executor, spawn_status] = cuda_executor_t::spawn(std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println("- CUDA: device #{} opens no stream", ordinal), 1;
    std::size_t failures_count = check_vram_index_cells(settings, executor);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes(settings, executor);
    failures_count += check_vram_nvidia_attend(settings, executor);
    return failures_count;
}

/** Reads device bytes back through the driver, which unified addressing points the right way. */
bool download_vram_nvidia(void* host, void const* device, std::size_t bytes) {
    return cuda_copy(host, device, bytes) == cuda_success_k;
}

static bool const vram_nvidia_registered =
    (vram_suites({gpu_vendor_t::nvidia_k, &check_vram_nvidia, &download_vram_nvidia}), true);

} // namespace unum::usearch::test::dense
