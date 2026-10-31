/**
 *  @file src/dense/ram_variants_intel_icelake.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The intel_icelake unit: the AVX-512 VNNI & VPOPCNT kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::intel_icelake_k>;
