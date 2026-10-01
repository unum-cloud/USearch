/**
 *  @file test/dense/vram_index_u1.cu
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed binary cells.
 */
#include "dense/vram_index.cuh"

using namespace unum::usearch;

std::size_t test::dense::check_vram_index_u1(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_cuda_configuration<nk_kernel_dot_k, nk_u1_k>("cuda-u1+dot", settings.seed);
    failures_count += test_cuda_configuration<nk_kernel_hamming_k, nk_u1_k>("cuda-u1+hamming", settings.seed);
    failures_count += test_cuda_configuration<nk_kernel_jaccard_k, nk_u1_k>("cuda-u1+jaccard", settings.seed);
    return failures_count;
}
