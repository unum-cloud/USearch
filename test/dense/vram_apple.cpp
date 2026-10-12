/**
 *  @file test/dense/vram_apple.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple suite's entry for the one VRAM runner: selects the device, then runs every
 *      dtype cell and the sharded suite the settings select over the Apple index.
 */
#include <fmt/format.h>

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::test::dense {

std::size_t check_vram_apple(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    fmt::println("- Metal: device #{} {}, {}, {} compute units, {} MiB", ordinal, device.name,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)), device.compute_units,
                 device.memory_bytes >> 20);
    selected_device = ordinal;
    std::size_t failures_count = check_vram_index_cells<gpu_vendor_t::apple_k>(settings);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes<gpu_vendor_t::apple_k>(settings);
    return failures_count;
}

static bool const vram_apple_registered = (vram_suites({gpu_vendor_t::apple_k, &check_vram_apple}), true);

} // namespace unum::usearch::test::dense
