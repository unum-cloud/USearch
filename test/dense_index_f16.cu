/**
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over f16 cells.
 *  @author Ash Vardanian
 *  @file test/dense_index_f16.cu
 *  @date July 23, 2026
 */
#include "dense_index_cuda.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_f16_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_f16_k>("cuda-f16+angular");
    return failures_count;
}
