/**
 *  @file src/dense/ram_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief The wasm64_extended unit: relaxed SIMD, shared memory, and 64-bit addressing.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::wasm64_extended_k>;
