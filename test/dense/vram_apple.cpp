/**
 *  @file test/dense/vram_apple.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple suite's entry for the one VRAM runner: selects the device, then runs the dense
 *      rung cells the settings select over the Apple index.
 */

#include <fmt/format.h>

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::test::dense {

std::size_t check_vram_apple(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    fmt::println("- Metal: device #{} {}, Apple{} family, {} compute units, {} MiB", ordinal, device.name,
                 device.architecture_major, device.compute_units, device.memory_bytes >> 20);
    auto [executor, spawn_status] = metal_executor_t::spawn(gpu_drivers().metal_contexts,
                                                            std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println("- Metal: device #{} opens no queue", ordinal), 1;
    return check_vram_indexes(settings, executor);
}

static bool const vram_apple_registered = (registered_suites({gpu_vendor_t::apple_k, &check_vram_apple}), true);

} // namespace unum::usearch::test::dense
