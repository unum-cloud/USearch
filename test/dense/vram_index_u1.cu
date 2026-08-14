/**
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed binary cells.
 *  @author Ash Vardanian
 *  @file test/dense/vram_index_u1.cu
 *  @date July 23, 2026
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test_dense_index_u1_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_hamming_k, nk_u1_k>("cuda-u1+hamming");
    failures_count += test_cuda_configuration<nk_kernel_jaccard_k, nk_u1_k>("cuda-u1+jaccard");
    return failures_count;
}
