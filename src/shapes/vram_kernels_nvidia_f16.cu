/**
 *  @file src/shapes/vram_kernels_nvidia_f16.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The f16 shapes kernels of one NVIDIA class, under that class's ceiling.
 *
 *  NVCC compiles it device-only once per class, with the class's codes and NumKong target, into the
 *  class's f16 fatbin: the vendor-wide Ampere floor, which @c vram_variants_nvidia.cpp finds linked
 *  in beside its variants, and datacenter Blackwell, which @c vram_variants_nvidia_blackwell.cpp
 *  finds.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_nvidia_metric.cuh"

template __device__ void us::shapes::cuda_instantiate<cuda_class_capabilities_k, nk_f16_k, u32_t>();
