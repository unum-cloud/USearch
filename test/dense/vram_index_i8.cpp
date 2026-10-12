/**
 *  @file test/dense/vram_index_i8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief Smoke & concurrency tests for the GPU-resident dense index over i8 cells.
 */
#include "vram_index.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <gpu_vendor_t vendor_>
std::size_t test::dense::check_vram_index_i8(settings_t const& settings) {
    std::size_t failures_count = 0;
    failures_count += test_vram_configuration<vendor_, nk_kernel_dot_k>("i8+dot", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_sqeuclidean_k>("i8+sqeuclidean", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_euclidean_k>("i8+euclidean", settings.seed);
    failures_count += test_vram_configuration<vendor_, nk_kernel_angular_k>("i8+angular", settings.seed);
    return failures_count;
}

template std::size_t test::dense::check_vram_index_i8<test::dense::vram_object_vendor_k>(settings_t const&);
