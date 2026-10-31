/**
 *  @file src/dense/ram_variants_apple_m4.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The apple_m4 unit: the Armv8.6 NEON kits plus the SME outer-product tiles.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::apple_m4_k>;
