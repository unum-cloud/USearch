/**
 *  @file src/dense/nvme_variants_apple_m4.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The apple_m4 NVMe unit: storage scans streaming through the SME outer-product tiles.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_indexes<nk_bf16_k, cpu_class_t::apple_m4_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<nvme_indexes<nk_e4m3_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<nvme_indexes<nk_e3m2_k, cpu_class_t::apple_m4_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<nvme_indexes<nk_e2m3_k, cpu_class_t::apple_m4_k>>;
#endif
