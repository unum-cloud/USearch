/**
 *  @file src/dense/ram_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The intel_skylake unit: its variants inline the AVX-512 kits, so the unit compiles
 *      with them.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::intel_skylake_k>;
