/**
 *  @file src/shapes/ram_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The wasm64_extended unit of the shapes variants: they inline the relaxed-SIMD kits, so
 *      the unit compiles with them.
 */
#include "shapes_specializations.cuh"

template struct shapes_ram_variant<nk::f32_t, cpu_class_t::wasm64_extended_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::wasm64_extended_k>;
