/**
 *  @file test/dense/vram_nvidia_i4.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The i4 cells' kernels, with the attends over the suite's value lookup.
 *
 *  NVCC compiles it device-only into the suite's i4 fatbin, which @c vram_nvidia.cpp embeds.
 */
#include "vram_nvidia_kernels.cuh"

namespace unum::usearch::dense {
template __device__ void
cuda_instantiate<test::dense::dense_nvidia_capabilities_k, nk_i4_k, test::dense::cuda_test_lookup_t>();
} // namespace unum::usearch::dense
