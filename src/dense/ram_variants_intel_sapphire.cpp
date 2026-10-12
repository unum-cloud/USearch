/**
 *  @file src/dense/ram_variants_intel_sapphire.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_sapphire unit: the Genoa kits plus the AMX-BF16 & AMX-INT8 tiles.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::intel_sapphire_k>>;
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::intel_sapphire_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::intel_sapphire_k>>;
#endif
