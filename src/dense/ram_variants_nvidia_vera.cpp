/**
 *  @file src/dense/ram_variants_nvidia_vera.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The nvidia_vera unit: the NEON FP8 kits.
 */
#include "dense_specializations.cuh"

#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::nvidia_vera_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::nvidia_vera_k>>;
#endif
