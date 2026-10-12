/**
 *  @file src/dense/vram_variants_apple.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The Apple unit: one arm per dtype serving every Apple class, Apple7 through Apple10,
 *      until a family's kernels differ.
 */
#include "dense_specializations.cuh"

template struct specialized_index<vram_apple_indexes<nk_bf16_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_apple_indexes<nk_u1_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_apple_indexes<nk_i8_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_apple_indexes<nk_e4m3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_apple_indexes<nk_f16_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_apple_indexes<nk_u8_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_apple_indexes<nk_e5m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_apple_indexes<nk_i4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_apple_indexes<nk_u4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_apple_indexes<nk_e3m2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_apple_indexes<nk_e2m3_k>>;
#endif
