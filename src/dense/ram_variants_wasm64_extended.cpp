/**
 *  @brief The wasm64_extended unit: relaxed SIMD, shared memory, and 64-bit addressing.
 *  @author Ash Vardanian
 *  @file src/dense/ram_variants_wasm64_extended.cpp
 *  @date September 12, 2026
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_f16_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i4_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u4_k, nk_kernel_dot_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u4_k, nk_kernel_angular_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u4_k, nk_kernel_euclidean_k, cpu_class_t::wasm64_extended_k>>;
