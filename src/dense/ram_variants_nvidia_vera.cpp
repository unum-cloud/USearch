/**
 *  @brief The nvidia_vera unit: the NEON FP8 kits.
 *  @author Ash Vardanian
 *  @file src/dense/ram_variants_nvidia_vera.cpp
 *  @date September 7, 2026
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::nvidia_vera_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::nvidia_vera_k>>;
