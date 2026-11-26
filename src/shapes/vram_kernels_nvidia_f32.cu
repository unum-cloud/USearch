/**
 *  @file src/shapes/vram_kernels_nvidia_f32.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The f32 shapes kernels at the Ampere floor, or a monoversioned build's one class.
 *
 *  NVCC compiles it device-only into the f32 fatbin, which @c vram_variants_nvidia.cpp finds linked
 *  in beside its variants.
 */
#include "usearch/shapes/vram_nvidia_metric.cuh" // `shapes::cuda_instantiate`, the NVIDIA rungs
#include "shapes_kits.hpp"                       // `shapes_vram_kit`
#include "classes.hpp"                           // `cuda_class_k`

template __device__ void us::shapes::cuda_instantiate<shapes_vram_kit<nk::f32_t, cuda_class_k>, util::u32_t>();
