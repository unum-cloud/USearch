/**
 *  @file src/basic/ram_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_skylake unit of the basic modalities: its arms inline the AVX-512 kits, so the
 *      unit compiles with them.
 */
#include "basic_specializations.cuh"

#if STRINGZILLA_TARGET_SKYLAKE
template struct sequence_arm<cpu_class_t::intel_skylake_k>;
#endif
#if NUMKONG_TARGET_SKYLAKE
template struct shapes_arm<nk::f32_t, cpu_class_t::intel_skylake_k>;
template struct shapes_arm<nk::f16_t, cpu_class_t::intel_skylake_k>;
template struct geospatial_arm<cpu_class_t::intel_skylake_k>;
#endif
