/**
 *  @file src/basic/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The aws_g2 unit of the basic modalities: its arms inline the NEON kits, so the unit
 *      compiles with them.
 */
#include "basic_specializations.cuh"

#if STRINGZILLA_TARGET_NEON
template struct sequence_arm<cpu_class_t::aws_g2_k>;
#endif
#if NUMKONG_TARGET_NEON
template struct shapes_arm<nk::f32_t, cpu_class_t::aws_g2_k>;
template struct shapes_arm<nk::f16_t, cpu_class_t::aws_g2_k>;
template struct geospatial_arm<cpu_class_t::aws_g2_k>;
#endif
