/**
 *  @file bench/dense/vram_kernels_f16.cu
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The f16 kernels every CUDA benchmark loads, under Ampere's ceiling and the newest
 *      compiled generation's, with the attends over both algebra lookups.
 *
 *  NVCC compiles it device-only, once for every benchmark, into the f16 fatbin
 *  @c vram_kernels.cpp names.
 */
#include "dense/vram_kernels.cuh"

namespace unum::usearch::bench::dense {
template __device__ void instantiate_cuda_ceilings<nk_f16_k>();
} // namespace unum::usearch::bench::dense
