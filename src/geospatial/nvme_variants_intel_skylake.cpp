/**
 *  @file src/geospatial/nvme_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The intel_skylake NVMe unit of the geospatial variants: they inline the AVX-512 kit, so
 *      the unit compiles with it.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_nvme_variant<cpu_class_t::intel_skylake_k>;
