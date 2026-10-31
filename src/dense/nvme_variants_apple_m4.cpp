/**
 *  @file src/dense/nvme_variants_apple_m4.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The apple_m4 NVMe unit: storage scans streaming through the SME outer-product tiles.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::apple_m4_k>;
