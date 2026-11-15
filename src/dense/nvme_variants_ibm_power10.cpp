/**
 *  @file src/dense/nvme_variants_ibm_power10.cpp
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief The ibm_power10 NVMe unit: storage walks scoring with the VSX and MMA kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::ibm_power10_k>;
