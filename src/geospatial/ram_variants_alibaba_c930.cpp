/**
 *  @file src/geospatial/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The alibaba_c930 unit of the geospatial variants: they inline the RVV kit, so the unit
 *      compiles with it.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_terrain<cpu_class_t::alibaba_c930_k>;
template struct geospatial_ram_variant<cpu_class_t::alibaba_c930_k>;
