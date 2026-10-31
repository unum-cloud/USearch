/**
 *  @file src/dense/nvme_variants_amd_genoa.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_genoa NVMe unit: storage walks scoring with the AVX-512 BF16 kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::amd_genoa_k>;
