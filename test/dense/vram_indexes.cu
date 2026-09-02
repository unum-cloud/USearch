/**
 *  @brief Concurrency & misuse tests for the GPU-resident @c dense_vram_indexes over the
 *      cells the single-index CUDA suite already wires.
 *  @author Ash Vardanian
 *  @file test/dense/vram_indexes.cu
 *  @date September 1, 2026
 */
#include "dense/vram_indexes.cuh"

using namespace unum::usearch;

std::size_t test_dense_indexes_cuda() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>("cuda-sharded bf16+angular");
    failures_count += test_cuda_indexes_configuration<nk_i8_k, nk_kernel_euclidean_k>("cuda-sharded i8+euclidean");
    return failures_count;
}
