/**
 *  @file src/dense/nvme_variants_intel_granite.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_granite NVMe unit: storage scans streaming through the AMX-FP16 tiles, the
 *      variants they change.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::intel_granite_k>;
