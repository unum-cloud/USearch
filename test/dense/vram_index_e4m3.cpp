/**
 *  @file test/dense/vram_index_e4m3.cpp
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over e4m3 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <typename executor_type_>
std::size_t test::dense::check_vram_index_e4m3(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<nk_kernel_dot_k, nk_e4m3_k>("e4m3+dot", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_angular_k, nk_e4m3_k>("e4m3+angular", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_euclidean_k, nk_e4m3_k>("e4m3+euclidean", settings.seed,
                                                                                executor);
    return failures_count;
}

#if USEARCH_ARCH_CUDA_
template std::size_t test::dense::check_vram_index_e4m3<cuda_executor_t>(settings_t const&, cuda_executor_t&);
#endif
#if USEARCH_ARCH_ROCM_
template std::size_t test::dense::check_vram_index_e4m3<hip_executor_t>(settings_t const&, hip_executor_t&);
#endif
#if USEARCH_ARCH_METAL_
template std::size_t test::dense::check_vram_index_e4m3<metal_executor_t>(settings_t const&, metal_executor_t&);
#endif
