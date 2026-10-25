/**
 *  @file src/dense/nvme_variants_apple_m2.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The apple_m2 NVMe unit: storage walks scoring with the NEON dot-product, BF16 & FP16
 *      kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::apple_m2_k>;
