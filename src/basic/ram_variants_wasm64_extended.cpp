/**
 *  @file src/basic/ram_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The wasm64_extended unit of the basic modalities: its shapes arms inline the relaxed-SIMD
 *      kits, so the unit compiles with them.
 */
#include "basic_specializations.cuh"

#if NUMKONG_TARGET_V128RELAXED
template struct shapes_arm<nk::f32_t, cpu_class_t::wasm64_extended_k>;
template struct shapes_arm<nk::f16_t, cpu_class_t::wasm64_extended_k>;
template struct geospatial_arm<cpu_class_t::wasm64_extended_k>;
#endif
