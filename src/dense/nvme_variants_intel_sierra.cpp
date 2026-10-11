/**
 *  @file src/dense/nvme_variants_intel_sierra.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_sierra NVMe unit: storage walks scoring with the AVX-VNNI kits, the cells they
 *      change.
 */
#include "dense_specializations.cuh"

#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<nvme_index<nk_i8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<nvme_index<nk_u8_k, cpu_class_t::intel_sierra_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<nvme_index<nk_e2m3_k, cpu_class_t::intel_sierra_k>>;
#endif
