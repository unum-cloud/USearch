/**
 *  @file src/dense/nvme_variants_nvidia_vera.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The nvidia_vera NVMe unit: storage walks scoring with the NEON FP8 kits.
 */
#include "dense_specializations.cuh"

template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<nvme_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
