/**
 *  @file src/dense/ram_variants_intel_granite.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_granite unit: the AMX-FP16 tiles of @c f16 and @c e5m2.
 */
#include "dense_specializations.cuh"

#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_granite_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_granite_k>>;
#endif
