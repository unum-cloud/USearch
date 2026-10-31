/**
 *  @file src/dense/ram_variants_intel_sapphire.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_sapphire unit: the Genoa kits plus the AMX-BF16 & AMX-INT8 tiles.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::intel_sapphire_k>;
