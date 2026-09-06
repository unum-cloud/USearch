/**
 *  @brief The i4 shard of the dense index library: its RAM arms' explicit instantiations.
 *  @author Ash Vardanian
 *  @file src/dense/ram_indexes_i4.cpp
 *  @date September 1, 2026
 */
#include "dense_specializations.cuh"

#if NK_TARGET_X8664_
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_diamond_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_diamond_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_diamond_k>>;
#elif NK_TARGET_ARM64_
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::apple_m2_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::apple_m2_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::apple_m2_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::aws_g5_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::aws_g5_k>>;
#elif NK_TARGET_RISCV64_
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::alibaba_c930_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::alibaba_c930_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::alibaba_c930_k>>;
#endif
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
