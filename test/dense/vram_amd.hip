/**
 *  @file test/dense/vram_amd.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The AMD suite's entry for the one VRAM runner: selects the device, then runs the dense
 *      rung cases and the sequence, shapes and geospatial cases.
 *
 *  Compiled host-only by HIP-Clang into the cells' index types; @c vram_kernels_amd.hip includes it
 *  and compiles the cells' kernels device-only into the suite's own code object.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
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

void run_vram_amd_cases(gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    hip_api_t const& api = gpu_drivers().hip;
    if (!expect_true(hip_set_device(api, ordinal) == hip_success_k, "ROCm device #{} does not answer", ordinal)) return;
    fmt::println("- ROCm: device #{} {}, gfx {}.{}, {}-lane waves, {} compute units, {} MiB", ordinal, device.name,
                 device.architecture_major, device.architecture_minor, device.warp_lanes, device.compute_units,
                 device.memory_bytes >> 20);
    auto [executor, spawn_status] = hip_executor_t::spawn(api, std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "ROCm device #{} opens no stream", ordinal)) return;
    run_vram_indexes_cases(executor);
    test::sequence::run_sequence_vram_cases<hip_executor_t>(device);
    test::shapes::run_shapes_vram_cases<hip_executor_t>(device);
    test::geospatial::run_geospatial_vram_cases<hip_executor_t>(device);
}

static bool const vram_amd_registered = (registered_suites({gpu_vendor_t::amd_k, &run_vram_amd_cases}), true);

} // namespace unum::usearch::test::dense
