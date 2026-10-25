/**
 *  @file src/dense/ram_variants_intel_granite.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The intel_granite unit: the AMX-FP16 tiles of @c f16 and @c e5m2.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::intel_granite_k>;
