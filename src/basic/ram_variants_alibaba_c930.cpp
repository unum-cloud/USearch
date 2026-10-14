/**
 *  @file src/basic/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The alibaba_c930 unit of the basic modalities: its shapes arms inline the RVV kits, so
 *      the unit compiles with them.
 */
#include "basic_specializations.cuh"

#if NUMKONG_TARGET_RVV
template struct shapes_arm<nk::f32_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_arm<nk::f16_t, cpu_class_t::alibaba_c930_k>;
template struct geospatial_arm<cpu_class_t::alibaba_c930_k>;
#endif
