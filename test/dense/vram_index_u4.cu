/**
 *  @file test/dense/vram_index_u4.cu
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed u4 cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_u4_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_dot_k, nk_u4_k>("cuda-u4+dot");
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_u4_k>("cuda-u4+angular");
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_u4_k>("cuda-u4+euclidean");
    return failures_count;
}
