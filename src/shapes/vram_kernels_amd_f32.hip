/**
 *  @file src/shapes/vram_kernels_amd_f32.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The f32 shapes kernels of one AMD class, under that class's ceiling.
 *
 *  HIP-Clang compiles it device-only once per class, with the class's processors and class target,
 *  into the class's f32 code object, which @c vram_variants_amd.hip finds linked in beside the
 *  variants of its class.
 */
#include "shapes_specializations.cuh"
#include "usearch/shapes/vram_amd_metric.hip"

template void us::shapes::amd_instantiate<
    us::shapes::shapes_ladder<nk::f32_t, numkong_capabilities_of(hip_class_k)>::kit_t, util::u32_t>();
