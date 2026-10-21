/**
 *  @file src/dense/vram_kernels_nvidia_e5m2.cu
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The e5m2 kernels of one NVIDIA class, under that class's ceiling.
 *
 *  NVCC compiles it device-only once per class, with the class's codes and NumKong target, into the
 *  class's e5m2 fatbin, which @c vram_variants_nvidia_<class>.cpp finds linked in beside its
 *  variants.
 */
#include "dense_kernels.cuh"
#include "usearch/dense/vram_nvidia_index.cuh"

template __device__ void dense::cuda_instantiate<cuda_class_capabilities_k, nk_e5m2_k, vram_attend_lookup_t>();
