/**
 *  @file src/shapes/vram_kernels_nvidia_f16.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The f16 shapes kernels at the Ampere floor, for every NVIDIA class.
 *
 *  NVCC compiles it device-only into the f16 fatbin, which @c vram_variants_nvidia.cpp finds linked
 *  in beside its variants.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_nvidia_metric.cuh"

template __device__ void us::shapes::cuda_instantiate<shapes_nvidia_capabilities_k, nk_f16_k, u32_t>();
