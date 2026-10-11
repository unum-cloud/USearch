/**
 *  @file src/dense/nvme_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_haswell NVMe unit: storage walks scoring with the AVX2 kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_bf16_k, cpu_class_t::intel_haswell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<nvme_index<nk_u1_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<nvme_index<nk_i8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<nvme_index<nk_e4m3_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<nvme_index<nk_f16_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<nvme_index<nk_u8_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<nvme_index<nk_e5m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<nvme_index<nk_i4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<nvme_index<nk_u4_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<nvme_index<nk_e3m2_k, cpu_class_t::intel_haswell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<nvme_index<nk_e2m3_k, cpu_class_t::intel_haswell_k>>;
#endif
