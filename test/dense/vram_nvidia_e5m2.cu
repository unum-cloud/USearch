/**
 *  @file test/dense/vram_nvidia_e5m2.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The e5m2 cells' kernels, with the attends over the suite's value lookup.
 *
 *  NVCC compiles it device-only into the suite's e5m2 fatbin, which @c vram_nvidia.cpp embeds.
 */
#include "vram_nvidia_kernels.cuh"

namespace unum::usearch::dense {
template __device__ void
cuda_instantiate<test::dense::dense_nvidia_capabilities_k, nk_e5m2_k, test::dense::cuda_test_lookup_t>();
} // namespace unum::usearch::dense
