/**
 *  @file src/dense/ram_variants_amd_genoa.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The amd_genoa unit: the AVX-512 BF16 kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::amd_genoa_k>>;
