/**
 *  @brief The serial unit: every cell at the portable floor, no instruction set beyond the baseline.
 *  @author Ash Vardanian
 *  @file src/dense/ram_variants_serial.cpp
 *  @date September 7, 2026
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::serial_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::serial_k>>;
