/**
 *  @file src/dense/nvme_variants_intel_icelake.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_icelake NVMe unit: storage walks scoring with the AVX-512 VNNI & VPOPCNT kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::intel_icelake_k>>;
template struct specialized_index<nvme_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::intel_icelake_k>>;
