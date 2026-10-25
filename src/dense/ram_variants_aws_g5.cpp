/**
 *  @file src/dense/ram_variants_aws_g5.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The aws_g5 unit: the apple_m2 kits under WFET waits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::aws_g5_k>;
