/**
 *  @file test/dense/vram_index_f16.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over f16 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <typename executor_type_>
std::size_t test::dense::check_vram_index_f16(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<nk_kernel_dot_k, nk_f16_k>("f16+dot", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_angular_k, nk_f16_k>("f16+angular", settings.seed, executor);
    failures_count += test_vram_configuration<nk_kernel_euclidean_k, nk_f16_k>("f16+euclidean", settings.seed,
                                                                               executor);
    return failures_count;
}

template std::size_t test::dense::check_vram_index_f16<cuda_executor_t>(settings_t const&, cuda_executor_t&);
template std::size_t test::dense::check_vram_index_f16<hip_executor_t>(settings_t const&, hip_executor_t&);
template std::size_t test::dense::check_vram_index_f16<metal_executor_t>(settings_t const&, metal_executor_t&);
