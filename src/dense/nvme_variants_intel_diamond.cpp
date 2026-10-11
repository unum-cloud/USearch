/**
 *  @file src/dense/nvme_variants_intel_diamond.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_diamond NVMe unit: storage walks scoring with the AVX10.2 FP8 kits.
 */
#include "dense_specializations.cuh"

#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<nvme_index<nk_e4m3_k, cpu_class_t::intel_diamond_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<nvme_index<nk_e5m2_k, cpu_class_t::intel_diamond_k>>;
#endif
