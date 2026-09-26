/**
 *  @file test/dense/vram_index_e3m2.cu
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over e3m2 cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test::dense::check_vram_index_e3m2() {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_dot_k, nk_e3m2_k>("cuda-e3m2+dot");
    failures_count += test_cuda_configuration<nk_kernel_angular_k, nk_e3m2_k>("cuda-e3m2+angular");
    failures_count += test_cuda_configuration<nk_kernel_euclidean_k, nk_e3m2_k>("cuda-e3m2+euclidean");
    return failures_count;
}
