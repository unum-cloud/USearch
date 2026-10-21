/**
 *  @file src/geospatial/ram_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The wasm64_extended unit of the geospatial variants: they inline the relaxed-SIMD kit, so
 *      the unit compiles with it.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_terrain<cpu_class_t::wasm64_extended_k>;
template struct geospatial_ram_variant<cpu_class_t::wasm64_extended_k>;
