/**
 *  @brief The aws_g2 unit: the NEON floor.
 *  @author Ash Vardanian
 *  @file src/dense/ram_variants_aws_g2.cpp
 *  @date September 7, 2026
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>>;
