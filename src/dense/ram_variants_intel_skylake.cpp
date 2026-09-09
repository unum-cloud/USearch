/**
 *  @brief The intel_skylake unit: its arms inline the AVX-512 kits, so the unit compiles with them.
 *  @author Ash Vardanian
 *  @file src/dense/ram_variants_intel_skylake.cpp
 *  @date September 7, 2026
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_skylake_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_skylake_k>>;
