/**
 *  @file src/dense/nvme_variants_intel_granite.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_granite NVMe unit: storage scans streaming through the AMX-FP16 tiles, the
 *      cells they change.
 */
#include "dense_specializations.cuh"

#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<nvme_indexes<nk_f16_k, cpu_class_t::intel_granite_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<nvme_indexes<nk_e5m2_k, cpu_class_t::intel_granite_k>>;
#endif
