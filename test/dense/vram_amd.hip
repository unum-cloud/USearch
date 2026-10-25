/**
 *  @file test/dense/vram_amd.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD suite's entry for the one VRAM runner: selects the device, then runs the dense
 *      rung cells the settings select.
 *
 *  Compiled host-only by HIP-Clang into the cells' index types; @c vram_kernels_amd.hip includes it
 *  and compiles the cells' kernels device-only into the suite's own code object.
 */
#include <fmt/format.h>

#include "usearch/dense/vram_amd_index.hip"

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::dense {
using test::dense::dense_amd_capabilities_k;
} // namespace unum::usearch::dense

namespace unum::usearch::test::dense {

/** The suite's code object, linked in as raw bytes beside its host half. */
extern "C" byte_t const test_vram_amd_bundle_start[], test_vram_amd_bundle_end[];

bytes_view_t dense_amd_code_object_t::bytes() noexcept {
    return {test_vram_amd_bundle_start,
            static_cast<std::size_t>(test_vram_amd_bundle_end - test_vram_amd_bundle_start)};
}

std::size_t check_vram_amd(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    if (hip_set_device(ordinal) != hip_success_k) return fmt::println("- ROCm: device #{} does not answer", ordinal), 1;
    fmt::println("- ROCm: device #{} {}, {}, {} compute units, {} MiB", ordinal, device.name,
                 usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(device.gpu_class)), device.compute_units,
                 device.memory_bytes >> 20);
    auto [executor, spawn_status] = hip_executor_t::spawn(std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println("- ROCm: device #{} opens no stream", ordinal), 1;
    return check_vram_indexes(settings, executor);
}

static bool const vram_amd_registered = (registered_suites({gpu_vendor_t::amd_k, &check_vram_amd}), true);

} // namespace unum::usearch::test::dense
