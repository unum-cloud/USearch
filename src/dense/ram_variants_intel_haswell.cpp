/**
 *  @file src/dense/ram_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The intel_haswell unit: its variants inline the AVX2 kits, so the unit compiles with
 *      them.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::intel_haswell_k>;
