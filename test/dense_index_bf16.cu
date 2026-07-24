/**
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over bf16 cells.
 *  @author Ash Vardanian
 *  @file test/dense_index_bf16.cu
 *  @date July 23, 2026
 */
#include "dense_index_cuda.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_bf16_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_bf16_k>("cuda-bf16+euclidean");
    return failures_count;
}
