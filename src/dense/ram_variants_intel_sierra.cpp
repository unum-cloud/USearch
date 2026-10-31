/**
 *  @file src/dense/ram_variants_intel_sierra.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The intel_sierra unit: the AVX2 & AVX-VNNI kits under TPAUSE waits & CMPCCXADD atomics.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::intel_sierra_k>;
