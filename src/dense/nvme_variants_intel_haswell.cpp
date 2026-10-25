/**
 *  @file src/dense/nvme_variants_intel_haswell.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_haswell NVMe unit: storage walks scoring with the AVX2 kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::intel_haswell_k>;
