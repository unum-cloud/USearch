/**
 *  @file src/dense/ram_variants_ibm_power10.cpp
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief The ibm_power10 unit: VSX and MMA kits under @c pause_short waits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::ibm_power10_k>;
