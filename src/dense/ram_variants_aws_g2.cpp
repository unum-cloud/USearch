/**
 *  @file src/dense/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The aws_g2 unit: the NEON floor.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::aws_g2_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::aws_g2_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::aws_g2_k>>;
#endif
