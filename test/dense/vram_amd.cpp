/**
 *  @file test/dense/vram_amd.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD suite's entry for the one VRAM runner: selects the device, then runs every dtype
 *      cell and the sharded suite the settings select over the AMD index.
 */
#include <fmt/format.h>

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::test::dense {

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
    return failures_count;
}

/** Reads device bytes back through the HIP runtime. */
bool download_vram_amd(void* host, void const* device, std::size_t bytes) {
    hip_api_t const& runtime = hip_api();
    return runtime && runtime.copy(host, device, bytes, hip_copy_device_to_host_k) == hip_success_k;
}

static bool const vram_amd_registered = (vram_suites({gpu_vendor_t::amd_k, &check_vram_amd, &download_vram_amd}), true);

} // namespace unum::usearch::test::dense
