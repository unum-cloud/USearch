/**
 *  @file src/dense/nvme_variants_intel_sapphire.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_sapphire NVMe unit: storage scans streaming through the AMX-BF16 & AMX-INT8
 *      tiles.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::intel_sapphire_k>;
