/**
 *  @file src/dense/ram_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The apple_m2 unit: the NEON dot-product, BF16 & FP16 kits under RCpc atomics.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::apple_m2_k>;
