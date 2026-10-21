/**
 *  @file bench/dense/vram_kernels_u4.cu
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The u4 kernels every CUDA benchmark loads, under Ampere's ceiling and the newest
 *      compiled generation's, with the attends over both algebra lookups.
 *
 *  NVCC compiles it device-only, once for every benchmark, into the u4 fatbin
 *  @c vram_kernels.cpp names.
 */
#include "dense/vram_kernels.cuh"

namespace unum::usearch::bench::dense {
template __device__ void instantiate_cuda_ceilings<nk_u4_k>();
} // namespace unum::usearch::bench::dense
