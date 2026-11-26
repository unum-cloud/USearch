/**
 *  @file test/dense/vram_nvidia.cpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The NVIDIA suite's entry for the one VRAM runner: selects the device, then runs the dense
 *      rung cases and every modality's VRAM cases.
 *
 *  Clang compiles it host-only into the cells' index types, whose kernels the library builds into
 *  its class fatbins.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
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

void run_vram_nvidia_cases(gpu_device_t const& device) {
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    cuda_api_t const& api = test::gpu_drivers().cuda;
    if (!expect_true(cuda_set_device(api, ordinal) == cuda_success_k, "CUDA device #{} does not answer", ordinal))
        return;
    fmt::println("- CUDA: {} sm_{}{} #{}, {} SMs, {} MiB", device.name, device.architecture_major,
                 device.architecture_minor, ordinal, device.compute_units, device.memory_bytes >> 20);
    auto [executor, spawn_status] = spawn_harness_executor<cuda_executor_t>(std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "CUDA device #{} opens no stream", ordinal)) return;
    if (dense_nvidia_runs(device)) run_vram_indexes_cases(executor);
    else
        run_test_case("run_vram_indexes_cases(gpu_vendor_t::nvidia_k)",
                      [] { return case_result_t::skipped(needs_dense_nvidia_class_k); });
    test::sequence::run_sequence_vram_cases<cuda_executor_t>(device);
    test::shapes::run_shapes_vram_cases<cuda_executor_t>(device);
    test::geospatial::run_geospatial_vram_cases<cuda_executor_t>(device);
}

static bool const vram_nvidia_registered = (registered_suites({gpu_vendor_t::nvidia_k, &run_vram_nvidia_cases}), true);

} // namespace unum::usearch::test::dense
