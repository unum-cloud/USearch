/**
 *  @file test/dense/vram_index_e4m3.cu
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over e4m3 cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_e4m3_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_dot_k, nk_e4m3_k>("cuda-e4m3+dot");
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_e4m3_k>("cuda-e4m3+angular");
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_e4m3_k>("cuda-e4m3+euclidean");
    return failures_count;
}
