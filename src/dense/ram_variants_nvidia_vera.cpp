/**
 *  @file src/dense/ram_variants_nvidia_vera.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The nvidia_vera unit: the NEON FP8 kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::nvidia_vera_k>;
