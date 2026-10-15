/**
 *  @file src/shapes/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The aws_g2 unit of the shapes arms: they inline the NEON kits, so the unit compiles with
 *      them.
 */
#include "shapes_specializations.cuh"

template struct shapes_ram_variant<nk::f32_t, cpu_class_t::aws_g2_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::aws_g2_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::aws_g2_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::aws_g2_k>;
