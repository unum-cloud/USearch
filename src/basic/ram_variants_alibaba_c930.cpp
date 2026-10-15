/**
 *  @file src/basic/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The alibaba_c930 unit of the basic modalities: its shapes and geospatial arms inline the
 *      RVV kits, so the unit compiles with them, and its sequence arms fall back to serial kits.
 */
#include "basic_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::alibaba_c930_k>;
template struct sequence_nvme_variant<cpu_class_t::alibaba_c930_k>;
template struct shapes_ram_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::alibaba_c930_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::alibaba_c930_k>;
template struct geospatial_terrain<cpu_class_t::alibaba_c930_k>;
template struct geospatial_ram_variant<cpu_class_t::alibaba_c930_k>;
template struct geospatial_nvme_variant<cpu_class_t::alibaba_c930_k>;
