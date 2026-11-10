/**
 *  @file test/dense/vram_nvidia.cpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA suite's entry for the one VRAM runner: selects the device, then runs the dense
 *      rung cells and every modality's VRAM cells the settings select.
 *
 *  Clang compiles it host-only into the cells' index types, whose kernels the library's class
 *  fatbins carry.
 */
#include <fmt/format.h>

#include "usearch/dense/vram_nvidia_index.cuh"

#include "vram_index.cuh"
#include "vram_indexes.cuh"

namespace unum::usearch::dense {

using test::dense::counting_cuda_allocator_t;
using test::dense::dense_nvidia_fatbin;

template struct vram_nvidia_index<nk_bf16_k, dense_nvidia_fatbin<nk_bf16_k>, fu::standard_yield_t, dummy_cache_hints_t,
                                  counting_cuda_allocator_t, counting_cuda_allocator_t, cuda_device_allocator_t>;
template struct vram_nvidia_indexes<uuid128_t, nk_bf16_k, dense_nvidia_fatbin<nk_bf16_k>>;

} // namespace unum::usearch::dense

namespace unum::usearch::test::dense {

std::size_t check_vram_nvidia(settings_t const& settings, gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    cuda_api_t const& api = test::gpu_drivers().cuda;
    if (cuda_set_device(api, ordinal) != cuda_success_k)
        return fmt::println("- CUDA: device #{} does not answer", ordinal), 1;
    fmt::println("- CUDA: {} sm_{}{} #{}, {} SMs, {} MiB", device.name, device.architecture_major,
                 device.architecture_minor, ordinal, device.compute_units, device.memory_bytes >> 20);
    auto [executor, spawn_status] = cuda_executor_t::spawn(api, std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println("- CUDA: device #{} opens no stream", ordinal), 1;
    std::size_t failures_count = 0;
    if (dense_nvidia_runs(device.architecture_major, device.architecture_minor))
        failures_count += check_vram_indexes(settings, executor);
    else
        fmt::println("- CUDA: dense rung cells skipped, the build names {} for them",
                     usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(dense_nvidia_class_k)));
    failures_count += test::sequence::check_sequence_vram(settings, device);
    failures_count += test::shapes::check_shapes_vram(settings, device);
    failures_count += test::geospatial::check_geospatial_vram(settings, device);
    return failures_count;
}

static bool const vram_nvidia_registered = (registered_suites({gpu_vendor_t::nvidia_k, &check_vram_nvidia}), true);

} // namespace unum::usearch::test::dense
