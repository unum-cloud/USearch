/**
 *  @file src/dense/nvme_variants_intel_icelake.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_icelake NVMe unit: storage walks scoring with the AVX-512 VNNI & VPOPCNT kits.
 */
#include "dense_specializations.cuh"

#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<nvme_indexes<nk_u1_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<nvme_indexes<nk_i8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<nvme_indexes<nk_u8_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<nvme_indexes<nk_i4_k, cpu_class_t::intel_icelake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<nvme_indexes<nk_u4_k, cpu_class_t::intel_icelake_k>>;
#endif
