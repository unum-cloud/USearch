/**
 *  @file test/dense/vram_amd.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD suite's entry for the one VRAM runner: selects the device, runs every dtype cell
 *      and the sharded suite the settings select, then device attention, whose value lookups run
 *      on the device.
 *
 *  Compiled host-only by HIP-Clang into the cells' index types, which the shared shards see only
 *  through the faces; @c vram_kernels_amd.hip includes it and compiles every cell's kernels, with
 *  the attends over this file's value lookups, device-only into the suite's own code object.
 */
#include <fmt/format.h>

#include "usearch/dense/vram_amd_index.hip"

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::dense {

/** A device lookup over a value table whose rows sit a 16-byte pitch apart: an ID below @c count
 *  names its row, a node ID on one GPU and a key on shards, and any other has none. */
struct hip_test_lookup_t {

    /** Names the lookup to the attend kernels the suite's code object carries. */
    inline static constexpr u32_t device_lookup_k = 1;

    byte_t const* rows;
    std::size_t pitch;
    std::size_t count;

    __device__ void const* operator()(u32_t id) const noexcept { return id < count ? rows + id * pitch : nullptr; }
};

using test::dense::dense_amd_capabilities_k;
using test::dense::dense_amd_code_object_t;

/** The lookups the suite attends through: by node ID on one GPU, and by key on shards. */
using hip_test_keyed_lookup_t = vram_keyed_lookup<u32_t, hip_test_lookup_t>;

template <>
struct amd_value_lookup_of<hip_test_lookup_t::device_lookup_k> {
    using type = hip_test_lookup_t;
};
template <>
struct amd_value_lookup_of<hip_test_keyed_lookup_t::device_lookup_k> {
    using type = hip_test_keyed_lookup_t;
};

} // namespace unum::usearch::dense

namespace unum::usearch::test::dense {

/** The suite's code object, linked in as raw bytes beside its host half. */
extern "C" byte_t const test_vram_amd_bundle_start[], test_vram_amd_bundle_end[];

bytes_view_t dense_amd_code_object_t::bytes() noexcept {
    return {test_vram_amd_bundle_start,
            static_cast<std::size_t>(test_vram_amd_bundle_end - test_vram_amd_bundle_start)};
}

/** A value table copied onto the device, its rows a 16-byte pitch apart; null rows when the copy
 *  failed. */
struct dense_hip_value_table_t {
    using lookup_t = hip_test_lookup_t;

    hip_unique_ptr<byte_t> rows;
    std::size_t pitch;

    /** Copies @p values onto the device of @p stream as the table every attend lookup in these
     *  suites reads. */
    template <typename scalar_type_>
    static dense_hip_value_table_t upload(nk::matrix_view<scalar_type_> values,
                                          hip_executor_t::stream_t const& stream) {
        std::size_t const rows_count = values.extent(0);
        std::size_t const row_bytes = divide_round_up<std::size_t>(
            values.extent(1) * nk_dtype_bits(scalar_type_::dtype()), 8);
        std::size_t const pitch = round_up_to_multiple<std::size_t>(row_bytes, 16);
        dense_hip_value_table_t table {hip_make_unique(hip_device_allocator_t {stream}, rows_count * pitch), pitch};
        hip_api_t const& runtime = hip_api();
        if (table.rows &&
            (runtime.copy_2d_async(table.rows.get(), pitch, values.byte_data(), values.stride_bytes(0), row_bytes,
                                   rows_count, hip_copy_host_to_device_k, stream.handle) != hip_success_k ||
             runtime.stream_synchronize(stream.handle) != hip_success_k))
            table.rows.reset();
        return table;
    }

    /** The device lookup over the first @p count rows. */
    lookup_t lookup(std::size_t count) const noexcept { return {rows.get(), pitch, count}; }
};

/** Device attention over every dtype cell the settings select, then over the sharded collection,
 *  all on @p executor 's device. */
static std::size_t check_vram_amd_attend(settings_t const& settings, hip_executor_t& executor) {
    std::size_t failures_count = check_vram_attend_cells<dense_hip_value_table_t>(settings, executor);
    if (settings.selects("vram_indexes"))
        failures_count += report(
            "rocm-sharded bf16+attend",
            dense_expect_vram_indexes_attend<vram_amd_indexes<u32_t, nk_bf16_k, dense_amd_code_object_t>,
                                             dense_hip_value_table_t>(settings.seed, executor));
    return failures_count;
}

std::size_t check_vram_amd(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    if (hip_set_device(ordinal) != hip_success_k) return fmt::println("- ROCm: device #{} does not answer", ordinal), 1;
    fmt::println("- ROCm: device #{} {}, {}, {} compute units, {} MiB", ordinal, device.name,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)), device.compute_units,
                 device.memory_bytes >> 20);
    auto [executor, spawn_status] = hip_executor_t::spawn(std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println("- ROCm: device #{} opens no stream", ordinal), 1;
    std::size_t failures_count = check_vram_index_cells(settings, executor);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes(settings, executor);
    failures_count += check_vram_amd_attend(settings, executor);
    return failures_count;
}

/** Reads device bytes back through the HIP runtime. */
bool download_vram_amd(void* host, void const* device, std::size_t bytes) {
    hip_api_t const& runtime = hip_api();
    return runtime && runtime.copy(host, device, bytes, hip_copy_device_to_host_k) == hip_success_k;
}

static bool const vram_amd_registered = (vram_suites({gpu_vendor_t::amd_k, &check_vram_amd, &download_vram_amd}), true);

} // namespace unum::usearch::test::dense
