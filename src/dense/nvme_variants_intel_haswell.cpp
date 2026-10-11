/**
 *  @file src/dense/nvme_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_haswell NVMe unit: storage walks scoring with the AVX2 kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_haswell_k>>;
template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_haswell_k>>;
