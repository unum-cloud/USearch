/**
 *  @file src/shapes/cpu_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_skylake unit of the shapes variants, on the host and on storage: they inline
 *      the AVX-512 kits, so the unit compiles with them.
 */
#include "shapes_specializations.cuh"

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::ram_k, cpu_class_t::intel_skylake_k>;
template struct basic_class_tables<shapes_traits_t, vendored_residency_t::nvme_k, cpu_class_t::intel_skylake_k>;
