/**
 *  @file test/dense/vram_apple.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple suite's entry for the one VRAM runner: selects the device, then runs every
 *      dtype cell and the sharded suite the settings select over the Apple index.
 */
#include <cstring> // `std::memcpy` - the download, over unified memory

#include <fmt/format.h>

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::test::dense {

std::size_t check_vram_apple(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    fmt::println("- Metal: device #{} {}, {}, {} compute units, {} MiB", ordinal, device.name,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)), device.compute_units,
                 device.memory_bytes >> 20);
    auto [executor, spawn_status] = metal_executor_t::spawn(std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println("- Metal: device #{} opens no queue", ordinal), 1;
    std::size_t failures_count = check_vram_index_cells(settings, executor);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes(settings, executor);
    return failures_count;
}

/** Reads device bytes back, which unified memory makes a plain copy. */
bool download_vram_apple(void* host, void const* device, std::size_t bytes) {
    std::memcpy(host, device, bytes);
    return true;
}

static bool const vram_apple_registered =
    (vram_suites({gpu_vendor_t::apple_k, &check_vram_apple, &download_vram_apple}), true);

} // namespace unum::usearch::test::dense
