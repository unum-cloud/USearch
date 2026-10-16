/**
 *  @file src/dense/ram_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The intel_skylake unit: its variants inline the AVX-512 kits, so the unit compiles
 *      with them.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_skylake_k>>;
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_skylake_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_skylake_k>>;
#endif
