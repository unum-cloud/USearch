/**
 *  @file src/dense/nvme_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The aws_g2 NVMe unit: storage walks scoring with the NEON floor.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::aws_g2_k>;
