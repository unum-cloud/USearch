/**
 *  @file src/dense/vram_kernels_amd_u4.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The u4 kernels of one AMD class, under that class's ceiling.
 *
 *  HIP-Clang compiles it device-only once per class, with the class's processors and NumKong
 *  target, into the class's u4 code object, which @c vram_variants_amd_<class>.hip finds
 *  linked in beside its variants.
 */
#include "dense_kernels.cuh"                // `vram_attend_lookup_t`, `hip_class_capabilities_k`
#include "usearch/dense/vram_amd_index.hip" // `dense::amd_instantiate`

template void us::dense::amd_instantiate<hip_class_capabilities_k, nk_u4_k, vram_attend_lookup_t>();
