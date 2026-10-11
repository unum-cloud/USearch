/**
 *  @file src/dense/vram_variants_nvidia_blackwell.cu
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The nvidia_blackwell unit: its arms inline the mixed-precision multiply-add & fp6-conversion
 *      kits, so the unit compiles for "100f" alone.
 */
#include "dense_specializations.cuh"
#include "usearch/dense/vram_nvidia_indexes.hpp"
#include "usearch/dense/vram_nvidia_index.cuh"

template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_blackwell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
