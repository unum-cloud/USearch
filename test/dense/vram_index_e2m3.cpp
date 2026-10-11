/**
 *  @file test/dense/vram_index_e2m3.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over e2m3 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;

template <gpu_vendor_t vendor_>
std::size_t test::dense::check_vram_index_e2m3(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<vendor_, nk_kernel_dot_k, nk_e2m3_k>("e2m3+dot", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_angular_k, nk_e2m3_k>("e2m3+angular", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_euclidean_k, nk_e2m3_k>("e2m3+euclidean",
                                                                                         settings.seed);
    return failures_count;
}

template std::size_t test::dense::check_vram_index_e2m3<test::dense::vram_object_vendor_k>(settings_t const&);
