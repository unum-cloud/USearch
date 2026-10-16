/**
 *  @file src/shapes/ram_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The apple_m2 unit of the shapes variants: the f16 ones widen through the NEONFHM kits,
 *      and the f32 ones fall back to the NEON kits they inline like aws_g2.
 */
#include "shapes_specializations.cuh"

template struct shapes_ram_variant<nk::f32_t, cpu_class_t::apple_m2_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::apple_m2_k>;
template struct shapes_nvme_variant<nk::f32_t, cpu_class_t::apple_m2_k>;
template struct shapes_nvme_variant<nk::f16_t, cpu_class_t::apple_m2_k>;
