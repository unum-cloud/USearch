/**
 *  @file src/dense/vram_kernels_amd_e3m2.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The e3m2 kernels of one AMD class, under that class's ceiling.
 *
 *  HIP-Clang compiles it device-only once per class, with the class's processors and NumKong
 *  target, into the class's e3m2 code object, which @c vram_variants_amd_<class>.hip finds
 *  linked in beside its variants.
 */
#include "dense_kernels.cuh"
#include "usearch/dense/vram_amd_index.hip"

template void dense::amd_instantiate<amd_class_capabilities_k, nk_e3m2_k, vram_attend_lookup_t::device_lookup_k>();
