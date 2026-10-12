/**
 *  @file src/dense/ram_variants_amd_genoa.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The amd_genoa unit: the AVX-512 BF16 kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::amd_genoa_k>>;
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::amd_genoa_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::amd_genoa_k>>;
#endif
