/**
 *  @file test/dense/vram_index_u1.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over packed binary cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <gpu_vendor_t vendor_>
std::size_t test::dense::check_vram_index_u1(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<vendor_, nk_kernel_dot_k, nk_u1_k>("u1+dot", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_hamming_k, nk_u1_k>("u1+hamming", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_jaccard_k, nk_u1_k>("u1+jaccard", settings.seed);
    return failures_count;
}

template std::size_t test::dense::check_vram_index_u1<gpu_vendor_t::nvidia_k>(settings_t const&);
template std::size_t test::dense::check_vram_index_u1<gpu_vendor_t::amd_k>(settings_t const&);
template std::size_t test::dense::check_vram_index_u1<gpu_vendor_t::apple_k>(settings_t const&);
