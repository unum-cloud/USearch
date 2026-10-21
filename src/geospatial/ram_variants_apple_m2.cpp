/**
 *  @file src/geospatial/ram_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The apple_m2 unit of the geospatial variants: they fall back to the NEON kit they inline
 *      like aws_g2.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_terrain<cpu_class_t::apple_m2_k>;
template struct geospatial_ram_variant<cpu_class_t::apple_m2_k>;
