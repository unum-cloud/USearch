/**
 *  @file src/dense/ram_variants_aws_g5.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The aws_g5 unit: the apple_m2 kits under WFET waits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::aws_g5_k>>;
