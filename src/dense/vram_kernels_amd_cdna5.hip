/**
 *  @file src/dense/vram_kernels_amd_cdna5.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_cdna5 unit's device entry: the kernels of every compiled dtype, for its code object.
 *
 *  HIP-Clang compiles it device-only, once over every compiled dtype, into the class's code object,
 *  which @c vram_variants_amd_cdna5.hip compiles host-only and finds linked in beside its variants.
 */
#include "dense_kernels.cuh"
#include "usearch/dense/vram_amd_index.hip"

template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_bf16_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#if USEARCH_DENSE_DTYPES >= 2
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_u1_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 3
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_i8_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 4
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_e4m3_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 5
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_f16_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 6
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_u8_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 7
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_e5m2_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 8
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_i4_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 9
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_u4_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 10
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_e3m2_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 11
template void dense::amd_instantiate<numkong_capabilities_of(gpu_class_t::amd_cdna5_k), nk_e2m3_k,
                                     vram_attend_lookup_t::device_lookup_k>();
#endif
