/**
 *  @file test/dense/vram_index_u4.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed u4 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;

template <gpu_vendor_t vendor_>
std::size_t test::dense::check_vram_index_u4(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<vendor_, nk_kernel_dot_k, nk_u4_k>("u4+dot", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_angular_k, nk_u4_k>("u4+angular", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_euclidean_k, nk_u4_k>("u4+euclidean", settings.seed);
    return failures_count;
}

template std::size_t test::dense::check_vram_index_u4<test::dense::vram_object_vendor_k>(settings_t const&);
