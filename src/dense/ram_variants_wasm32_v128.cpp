/**
 *  @file src/dense/ram_variants_wasm32_v128.cpp
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief The wasm32_v128 unit: fixed-width SIMD128, one thread, the variants with a strict kit.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::wasm32_v128_k>;
