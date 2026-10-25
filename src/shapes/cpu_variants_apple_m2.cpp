/**
 *  @file src/shapes/cpu_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The apple_m2 unit of the shapes variants, on the host and on storage: the f16 ones widen
 *      through the NEONFHM kits, and the f32 ones fall back to the NEON kits they inline like
 *      aws_g2.
 */
#include "shapes_specializations.cuh"

template struct basic_class_tables<shapes_traits_t, dense_home_t::ram_k, cpu_class_t::apple_m2_k>;
template struct basic_class_tables<shapes_traits_t, dense_home_t::nvme_k, cpu_class_t::apple_m2_k>;
