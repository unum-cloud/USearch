/**
 *  @brief The u1 shard of the dense index library: its RAM arms' explicit instantiations.
 *  @author Ash Vardanian
 *  @file src/dense/ram_indexes_u1.cpp
 *  @date September 1, 2026
 */
#include "dense_specializations.cuh"

#if NK_TARGET_X8664_
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>>;
#elif NK_TARGET_ARM64_
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g2_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::aws_g5_k>>;
#endif
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::serial_k>>;
