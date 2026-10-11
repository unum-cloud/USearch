/**
 *  @file src/dense/ram_variants_wasm32_v128.cpp
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief The wasm32_v128 unit: fixed-width SIMD128, one thread, the cells with a strict kit.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u1_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::wasm32_v128_k>>;
template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::wasm32_v128_k>>;
