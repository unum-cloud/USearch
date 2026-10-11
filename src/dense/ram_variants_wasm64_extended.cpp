/**
 *  @file src/dense/ram_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief The wasm64_extended unit: relaxed SIMD, shared memory, and 64-bit addressing.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_index<nk_bf16_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i8_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e4m3_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_f16_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u8_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e5m2_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_i4_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_u4_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e3m2_k, cpu_class_t::wasm64_extended_k>>;
template struct specialized_index<ram_index<nk_e2m3_k, cpu_class_t::wasm64_extended_k>>;
