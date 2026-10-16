/**
 *  @file test/dense/vram_index_u8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over u8 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <typename executor_type_>
std::size_t test::dense::check_vram_index_u8(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<nk_kernel_dot_k, nk_u8_k>("u8+dot", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_angular_k, nk_u8_k>("u8+angular", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_euclidean_k, nk_u8_k>("u8+euclidean", settings.seed, executor);
    return failures_count;
}

#if USEARCH_ARCH_CUDA_
template std::size_t test::dense::check_vram_index_u8<cuda_executor_t>(settings_t const&, cuda_executor_t&);
#endif
#if USEARCH_ARCH_ROCM_
template std::size_t test::dense::check_vram_index_u8<hip_executor_t>(settings_t const&, hip_executor_t&);
#endif
#if USEARCH_ARCH_METAL_
template std::size_t test::dense::check_vram_index_u8<metal_executor_t>(settings_t const&, metal_executor_t&);
#endif
