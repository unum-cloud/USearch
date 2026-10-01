/**
 *  @file test/dense/vram_index_u8.cu
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over u8 cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test::dense::check_vram_index_u8(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_dot_k, nk_u8_k>("cuda-u8+dot", settings.seed);
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_u8_k>("cuda-u8+angular", settings.seed);
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_u8_k>("cuda-u8+euclidean", settings.seed);
    return failures_count;
}
