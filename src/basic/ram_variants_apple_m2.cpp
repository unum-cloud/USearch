/**
 *  @file src/basic/ram_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The apple_m2 unit of the basic modalities: its f16 shapes arms widen through the NEONFHM
 *      kits, and every other arm falls back to the NEON kits it inlines like aws_g2.
 */
#include "basic_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::apple_m2_k>;
template struct sequence_nvme_variant<cpu_class_t::apple_m2_k>;
template struct shapes_ram_variant<nk::f32_t, cpu_class_t::apple_m2_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::apple_m2_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::apple_m2_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::apple_m2_k>;
template struct geospatial_terrain<cpu_class_t::apple_m2_k>;
template struct geospatial_ram_variant<cpu_class_t::apple_m2_k>;
template struct geospatial_nvme_variant<cpu_class_t::apple_m2_k>;
