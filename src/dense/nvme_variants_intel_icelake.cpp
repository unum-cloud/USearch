/**
 *  @file src/dense/nvme_variants_intel_icelake.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_icelake NVMe unit: storage walks scoring with the AVX-512 VNNI & VPOPCNT kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::intel_icelake_k>;
