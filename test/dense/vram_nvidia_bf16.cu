/**
 *  @file test/dense/vram_nvidia_bf16.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The bf16 cells' kernels, with the attends over the suite's value lookup and the keyed one
 *      the sharded suite attends through.
 *
 *  NVCC compiles it device-only into the suite's bf16 fatbin, which @c vram_nvidia.cpp embeds.
 */
#include "vram_nvidia_kernels.cuh"

namespace unum::usearch::dense {
template __device__ void cuda_instantiate<test::dense::dense_nvidia_capabilities_k, nk_bf16_k,
                                          test::dense::cuda_test_lookup_t, test::dense::cuda_test_keyed_lookup_t>();
} // namespace unum::usearch::dense
