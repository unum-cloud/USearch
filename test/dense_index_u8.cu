/**
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over u8 cells.
 *  @author Ash Vardanian
 *  @file test/dense_index_u8.cu
 *  @date July 23, 2026
 */
#include "dense_index_cuda.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_u8_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_u8_k>("cuda-u8+euclidean");
    return failures_count;
}
