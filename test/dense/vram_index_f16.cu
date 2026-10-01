/**
 *  @file test/dense/vram_index_f16.cu
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over f16 cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test::dense::check_vram_index_f16(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_dot_k, nk_f16_k>("cuda-f16+dot", settings.seed);
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_f16_k>("cuda-f16+angular", settings.seed);
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_f16_k>("cuda-f16+euclidean", settings.seed);
    return failures_count;
}
