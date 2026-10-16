/**
 *  @file src/dense/vram_variants_apple_metal7.cpp
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The Apple7 unit: one variant per dtype, each compiling its kernels from the
 *      embedded source for the class's capability set at the first @c reserve.
 */
#include "dense_specializations.cuh"

/* apple_metal7, the Apple floor by metal: every cell, 11 variants */
template struct specialized_index<vram_apple_indexes<nk_bf16_k, gpu_class_t::apple_metal7_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_apple_indexes<nk_u1_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_apple_indexes<nk_i8_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_apple_indexes<nk_e4m3_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_apple_indexes<nk_f16_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_apple_indexes<nk_u8_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_apple_indexes<nk_e5m2_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_apple_indexes<nk_i4_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_apple_indexes<nk_u4_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_apple_indexes<nk_e3m2_k, gpu_class_t::apple_metal7_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_apple_indexes<nk_e2m3_k, gpu_class_t::apple_metal7_k>>;
#endif
