/**
 *  @file src/dense/nvme_variants_intel_diamond.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_diamond NVMe unit: storage walks scoring with the AVX10.2 FP8 kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::intel_diamond_k>;
