/**
 *  @file src/basic/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The aws_g2 unit of the basic modalities: its arms inline the NEON kits, so the unit
 *      compiles with them.
 */
#include "basic_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::aws_g2_k>;
template struct sequence_nvme_variant<cpu_class_t::aws_g2_k>;
template struct shapes_ram_variant<nk::f32_t, cpu_class_t::aws_g2_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::aws_g2_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::aws_g2_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::aws_g2_k>;
template struct geospatial_terrain<cpu_class_t::aws_g2_k>;
template struct geospatial_ram_variant<cpu_class_t::aws_g2_k>;
template struct geospatial_nvme_variant<cpu_class_t::aws_g2_k>;
