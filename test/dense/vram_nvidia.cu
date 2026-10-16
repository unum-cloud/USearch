/**
 *  @file test/dense/vram_nvidia.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA suite's entry for the one VRAM runner: selects the device, runs every dtype
 *      cell and the sharded suite the settings select, then device attention, whose value lookups
 *      run on the device.
 *
 *  Compiled twice, like the library's class units: by NVCC device-only into the suite's own fatbin,
 *  carrying every cell's kernels and the attends over this file's value lookups, and by Clang
 *  host-only into the cells' index types, which the shared shards see only through the faces.
 */
#include <fmt/format.h>

#include "usearch/dense/vram_nvidia_index.cuh"

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::dense {

/** A device lookup over a value table whose rows sit a 16-byte pitch apart: an ID below @c count
 *  names its row, a node ID on one GPU and a key on shards, and any other has none. */
struct cuda_test_lookup_t {

    /** Names the lookup to the attend kernels the suite's fatbin carries. */
    inline static constexpr u32_t device_lookup_k = 1;

    byte_t const* rows;
    std::size_t pitch;
    std::size_t count;

    __device__ void const* operator()(u32_t id) const noexcept { return id < count ? rows + id * pitch : nullptr; }
};

using test::dense::counting_cuda_allocator_t;
using test::dense::dense_nvidia_capabilities_k;
using test::dense::dense_nvidia_fatbin_t;

/** The lookups the suite attends through: by node ID on one GPU, and by key on shards. */
using cuda_test_keyed_lookup_t = vram_keyed_lookup<u32_t, cuda_test_lookup_t>;

template <>
struct cuda_value_lookup_of<cuda_test_lookup_t::device_lookup_k> {
    using type = cuda_test_lookup_t;
};
template <>
struct cuda_value_lookup_of<cuda_test_keyed_lookup_t::device_lookup_k> {
    using type = cuda_test_keyed_lookup_t;
};

// Every cell's kernels and attends, and the keyed attends of the sharded suite's collection.
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_bf16_k, cuda_test_lookup_t::device_lookup_k,
                                          cuda_test_keyed_lookup_t::device_lookup_k>();
#if USEARCH_DENSE_DTYPES >= 2
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_u1_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 3
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_i8_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 4
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e4m3_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 5
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_f16_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 6
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_u8_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 7
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e5m2_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 8
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_i4_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 9
template __device__ void cuda_instantiate<dense_nvidia_capabilities_k, nk_u4_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 10
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e3m2_k, cuda_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 11
template __device__ void
cuda_instantiate<dense_nvidia_capabilities_k, nk_e2m3_k, cuda_test_lookup_t::device_lookup_k>();
#endif

// The ledgered index of every NVIDIA cell and the collections the sharded suite drives.
template struct vram_nvidia_index<nk_bf16_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
template struct vram_nvidia_indexes<u32_t, nk_bf16_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t>;
#if USEARCH_DENSE_DTYPES >= 2
template struct vram_nvidia_index<nk_u1_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct vram_nvidia_index<nk_i8_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
template struct vram_nvidia_indexes<u32_t, nk_i8_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct vram_nvidia_index<nk_e4m3_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct vram_nvidia_index<nk_f16_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct vram_nvidia_index<nk_u8_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct vram_nvidia_index<nk_e5m2_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct vram_nvidia_index<nk_i4_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct vram_nvidia_index<nk_u4_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct vram_nvidia_index<nk_e3m2_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct vram_nvidia_index<nk_e2m3_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
#endif

} // namespace unum::usearch::dense

namespace unum::usearch::test::dense {

/** The suite's fatbin, linked in as raw bytes beside its host half. */
extern "C" byte_t const test_vram_nvidia_bundle_start[], test_vram_nvidia_bundle_end[];

bytes_view_t dense_nvidia_fatbin_t::bytes() noexcept {
    return {test_vram_nvidia_bundle_start,
            static_cast<std::size_t>(test_vram_nvidia_bundle_end - test_vram_nvidia_bundle_start)};
}

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
            dense_expect_vram_indexes_attend<
                vram_nvidia_indexes<u32_t, nk_bf16_k, dense_nvidia_capabilities_k, dense_nvidia_fatbin_t>,
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
