/**
 *  @file src/shapes/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The alibaba_c930 unit of the shapes arms: they inline the RVV kits, so the unit compiles
 *      with them.
 */
#include "shapes_specializations.cuh"

template struct shapes_ram_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>;
