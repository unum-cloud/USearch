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
    selected_device = ordinal;
    std::size_t failures_count = check_vram_index_cells<gpu_vendor_t::amd_k>(settings);
    if (settings.selects("vram_indexes")) failures_count += check_vram_indexes<gpu_vendor_t::amd_k>(settings);
    return failures_count;
}

static bool const vram_amd_registered = (vram_suites({gpu_vendor_t::amd_k, &check_vram_amd}), true);

} // namespace unum::usearch::test::dense
