/**
 *  @file src/dense/nvme_variants_intel_sierra.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_sierra NVMe unit: storage walks scoring with the AVX-VNNI kits, the variants
 *      they change.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::intel_sierra_k>;
