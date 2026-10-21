/**
 *  @file src/shapes/nvme_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_haswell NVMe unit of the shapes variants: they inline the AVX2 kits, so the
 *      unit compiles with them.
 */
#include "shapes_specializations.cuh"

template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::intel_haswell_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::intel_haswell_k>;
