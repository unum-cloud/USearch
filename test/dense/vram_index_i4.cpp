/**
 *  @file test/dense/vram_index_i4.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed i4 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;

template <gpu_vendor_t vendor_>
std::size_t test::dense::check_vram_index_i4(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<vendor_, nk_kernel_dot_k, nk_i4_k>("i4+dot", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_angular_k, nk_i4_k>("i4+angular", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_euclidean_k, nk_i4_k>("i4+euclidean", settings.seed);
    return failures_count;
}

template std::size_t test::dense::check_vram_index_i4<test::dense::vram_object_vendor_k>(settings_t const&);
