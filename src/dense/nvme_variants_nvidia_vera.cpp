/**
 *  @file src/dense/nvme_variants_nvidia_vera.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The nvidia_vera NVMe unit: storage walks scoring with the NEON FP8 kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::nvidia_vera_k>;
