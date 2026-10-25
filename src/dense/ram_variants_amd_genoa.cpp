/**
 *  @file src/dense/ram_variants_amd_genoa.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The amd_genoa unit: the AVX-512 BF16 kits.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::amd_genoa_k>;
