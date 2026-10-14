/**
 *  @file src/basic/ram_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The intel_haswell unit of the basic modalities: its arms inline the AVX2 kits, so the
 *      unit compiles with them.
 */
#include "basic_specializations.cuh"

#if STRINGZILLA_TARGET_HASWELL
template struct sequence_arm<cpu_class_t::intel_haswell_k>;
#endif
#if NUMKONG_TARGET_HASWELL
template struct shapes_arm<nk::f32_t, cpu_class_t::intel_haswell_k>;
template struct shapes_arm<nk::f16_t, cpu_class_t::intel_haswell_k>;
template struct geospatial_arm<cpu_class_t::intel_haswell_k>;
#endif
