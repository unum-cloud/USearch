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

template <gpu_vendor_t vendor_>
std::size_t test::dense::check_vram_indexes(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_vram_indexes_configuration<vendor_, nk_bf16_k, nk_kernel_dot_k>("sharded bf16+dot",
                                                                                           settings.seed);
    failures_count += test_vram_indexes_configuration<vendor_, nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular",
                                                                                               settings.seed);
#if USEARCH_DENSE_DTYPES >= 3
    failures_count += test_vram_indexes_configuration<vendor_, nk_i8_k, nk_kernel_euclidean_k>("sharded i8+euclidean",
                                                                                               settings.seed);
#endif
    return failures_count;
}

template std::size_t test::dense::check_vram_indexes<test::dense::vram_object_vendor_k>(settings_t const&);
