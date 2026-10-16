/**
 *  @file test/dense/vram_index_e5m2.cpp
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over e5m2 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <typename executor_type_>
std::size_t test::dense::check_vram_index_e5m2(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<nk_kernel_dot_k, nk_e5m2_k>("e5m2+dot", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_angular_k, nk_e5m2_k>("e5m2+angular", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_euclidean_k, nk_e5m2_k>("e5m2+euclidean", settings.seed,
                                                                                executor);
    return failures_count;
}

#if USEARCH_ARCH_CUDA_
template std::size_t test::dense::check_vram_index_e5m2<cuda_executor_t>(settings_t const&, cuda_executor_t&);
#endif
#if USEARCH_ARCH_ROCM_
template std::size_t test::dense::check_vram_index_e5m2<hip_executor_t>(settings_t const&, hip_executor_t&);
#endif
#if USEARCH_ARCH_METAL_
template std::size_t test::dense::check_vram_index_e5m2<metal_executor_t>(settings_t const&, metal_executor_t&);
#endif
