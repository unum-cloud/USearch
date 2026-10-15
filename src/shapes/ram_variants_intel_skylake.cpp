/**
 *  @file src/shapes/ram_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_skylake unit of the shapes arms: they inline the AVX-512 kits, so the unit
 *      compiles with them.
 */
#include "shapes_specializations.cuh"

template struct shapes_ram_variant<nk::f32_t, cpu_class_t::intel_skylake_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::intel_skylake_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::intel_skylake_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::intel_skylake_k>;
