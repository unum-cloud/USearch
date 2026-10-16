/**
 *  @file test/dense/vram_indexes.cpp
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief Concurrency & misuse tests for the GPU-resident sharded index over the cells the
 *      single-index suite already wires.
 */
#include "vram_indexes.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <typename executor_type_>
std::size_t test::dense::check_vram_indexes(settings_t const& settings, executor_type_& executor) {
    std::size_t failures_count = 0;
    failures_count += test_vram_indexes_configuration<nk_bf16_k, nk_kernel_dot_k>("sharded bf16+dot", settings.seed,
                                                                                  executor);
    failures_count += test_vram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular",
                                                                                      settings.seed, executor);
#if USEARCH_DENSE_DTYPES >= 3
    failures_count += test_vram_indexes_configuration<nk_i8_k, nk_kernel_euclidean_k>("sharded i8+euclidean",
                                                                                      settings.seed, executor);
#endif
    return failures_count;
}

#if USEARCH_ARCH_CUDA_
template std::size_t test::dense::check_vram_indexes<cuda_executor_t>(settings_t const&, cuda_executor_t&);
#endif
#if USEARCH_ARCH_ROCM_
template std::size_t test::dense::check_vram_indexes<hip_executor_t>(settings_t const&, hip_executor_t&);
#endif
#if USEARCH_ARCH_METAL_
template std::size_t test::dense::check_vram_indexes<metal_executor_t>(settings_t const&, metal_executor_t&);
#endif
