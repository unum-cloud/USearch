/**
 *  @file src/geospatial/cpu_variants_wasm64_extended.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The wasm64_extended unit of the geospatial variants, on the host and on storage: they
 *      inline the relaxed-SIMD kit, so the unit compiles with it.
 */
#include "geospatial_specializations.cuh"

template struct basic_class_tables<geospatial_traits_t, dense_home_t::ram_k, cpu_class_t::wasm64_extended_k>;
template struct basic_class_tables<geospatial_traits_t, dense_home_t::nvme_k, cpu_class_t::wasm64_extended_k>;
