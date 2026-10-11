/**
 *  @file src/dense/nvme_variants_intel_sierra.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_sierra NVMe unit: storage walks scoring with the AVX-VNNI kits, the cells they
 *      change.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::intel_sierra_k>>;
template struct specialized_index<nvme_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::intel_sierra_k>>;
