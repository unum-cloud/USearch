/**
 *  @file test/dense/vram_index_i4.cu
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed i4 cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_i4_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_i4_k>("cuda-i4+angular");
    return failures_count;
}
