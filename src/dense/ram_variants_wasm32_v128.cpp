/**
 *  @brief The wasm32_v128 unit: fixed-width SIMD128, one thread, the cells with a strict kit.
 *  @author Ash Vardanian
 *  @file src/dense/ram_variants_wasm32_v128.cpp
 *  @date September 12, 2026
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_bf16_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_i8_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_angular_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u8_k, nk_kernel_euclidean_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_dot_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_hamming_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u1_k, nk_kernel_jaccard_k, cpu_class_t::wasm32_v128_k>>;
