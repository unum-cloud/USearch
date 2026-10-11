/**
 *  @file src/dense/vram_variants_nvidia_ampere.cu
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The nvidia_ampere unit: the floor kits, compiled for "80" alone with the PTX every GPU without a
 *      class of its own runs.
 */
#include "dense_specializations.cuh"
#include "usearch/dense/vram_nvidia_indexes.hpp"
#include "usearch/dense/vram_nvidia_index.cuh"

template struct specialized_index<vram_index<nk_bf16_k, gpu_class_t::nvidia_ampere_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_index<nk_u1_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_index<nk_i8_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_index<nk_e4m3_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_index<nk_f16_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_index<nk_u8_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_index<nk_e5m2_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_index<nk_i4_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_index<nk_u4_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_index<nk_e3m2_k, gpu_class_t::nvidia_ampere_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_index<nk_e2m3_k, gpu_class_t::nvidia_ampere_k>>;
#endif
