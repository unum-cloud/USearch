/**
 *  @file src/dense/ram_variants_monoversioned.cpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The monoversioned unit: every variant with the kits the build's floor and vouched
 *      capabilities compiled.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::monoversioned_k>;
