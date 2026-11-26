/**
 *  @file src/shapes/vram_kernels_amd_f16.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The f16 shapes kernels of one AMD class, under that class's ceiling.
 *
 *  HIP-Clang compiles it device-only once per class, with the class's processors and class target,
 *  into the class's f16 code object, which @c vram_variants_amd.hip finds linked in beside the
 *  variants of its class.
 */
#include "usearch/shapes/vram_amd_metric.hip" // `shapes::amd_instantiate`, the AMD rungs
#include "shapes_kits.hpp"                    // `shapes_vram_kit`
#include "classes.hpp"                        // `hip_class_k`

template void us::shapes::amd_instantiate<shapes_vram_kit<nk::f16_t, hip_class_k>, util::u32_t>();
