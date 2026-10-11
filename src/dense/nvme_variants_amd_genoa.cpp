/**
 *  @file src/dense/nvme_variants_amd_genoa.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_genoa NVMe unit: storage walks scoring with the AVX-512 BF16 kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::amd_genoa_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::amd_genoa_k>>;
