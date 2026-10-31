/**
 *  @file src/geospatial/cpu_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The intel_haswell unit of the geospatial variants, on the host and on storage: they
 *      inline the AVX2 kit, so the unit compiles with it.
 */
#include "geospatial_specializations.cuh"

template struct basic_class_tables<geospatial_traits_t, vendored_residency_t::ram_k, cpu_class_t::intel_haswell_k>;
template struct basic_class_tables<geospatial_traits_t, vendored_residency_t::nvme_k, cpu_class_t::intel_haswell_k>;
