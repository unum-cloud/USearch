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
    auto const [specs, status] = metal_device_specs_t::query(ordinal);
    if (failed(status)) return fmt::println("- Metal: device #{} does not answer", ordinal), 1;
    fmt::println("- Metal: device #{} {}", ordinal,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(specs.gpu_class)));
    selected_device = ordinal;
    std::size_t failures_count = check_vram_index_cells<gpu_vendor_t::apple_k>(settings);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes<gpu_vendor_t::apple_k>(settings);
    return failures_count;
}

static bool const vram_apple_registered = (vram_suites({gpu_vendor_t::apple_k, &check_vram_apple}), true);

} // namespace unum::usearch::test::dense
