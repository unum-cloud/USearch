/**
 *  @file src/dense/nvme_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_skylake NVMe unit: storage walks scoring with the AVX-512 kits, the cells they
 *      change.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_bf16_k, cpu_class_t::intel_skylake_k>>;
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<nvme_index<nk_e4m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<nvme_index<nk_f16_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<nvme_index<nk_e5m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<nvme_index<nk_e3m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<nvme_index<nk_e2m3_k, cpu_class_t::intel_skylake_k>>;
#endif
