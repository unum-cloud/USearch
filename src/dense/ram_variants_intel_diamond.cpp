/**
 *  @file src/dense/ram_variants_intel_diamond.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The intel_diamond unit: the AVX10.2 kits under TPAUSE waits & CMPCCXADD atomics.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::intel_diamond_k>;
