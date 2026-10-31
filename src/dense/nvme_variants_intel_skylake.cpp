/**
 *  @file src/dense/nvme_variants_intel_skylake.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_skylake NVMe unit: storage walks scoring with the AVX-512 kits, the variants
 *      they change.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::intel_skylake_k>;
