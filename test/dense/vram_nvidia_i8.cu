/**
 *  @file test/dense/vram_nvidia_i8.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The i8 cells' kernels, with the attends over the suite's value lookup.
 *
 *  NVCC compiles it device-only into the suite's i8 fatbin, which @c vram_nvidia.cpp embeds.
 */
#include "vram_nvidia_kernels.cuh"

namespace unum::usearch::dense {
template __device__ void
cuda_instantiate<test::dense::dense_nvidia_capabilities_k, nk_i8_k, test::dense::cuda_test_lookup_t::device_lookup_k>();
} // namespace unum::usearch::dense
