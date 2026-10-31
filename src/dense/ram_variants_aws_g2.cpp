/**
 *  @file src/dense/ram_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The aws_g2 unit: the NEON floor.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::aws_g2_k>;
