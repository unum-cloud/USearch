/**
 *  @file test/dense/vram_apple.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple suite's entry for the one VRAM runner: selects the device, then runs the dense
 *      rung cases over the Apple index and the basic modalities' cases.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#include <fmt/format.h>

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::test::dense {

void run_vram_apple_cases(gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    fmt::println("- Metal: device #{} {}, Apple{} family, {} compute units, {} MiB", ordinal, device.name,
                 device.architecture_major, device.compute_units, device.memory_bytes >> 20);
    auto [executor, spawn_status] = metal_executor_t::spawn(gpu_drivers().metal_contexts,
                                                            std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "Metal device #{} opens no queue", ordinal)) return;
    run_vram_indexes_cases(executor);
    test::sequence::run_sequence_vram_cases<metal_executor_t>(device);
    test::shapes::run_shapes_vram_cases<metal_executor_t>(device);
    test::geospatial::run_geospatial_vram_cases<metal_executor_t>(device);
}

static bool const vram_apple_registered = (registered_suites({gpu_vendor_t::apple_k, &run_vram_apple_cases}), true);

} // namespace unum::usearch::test::dense
