/**
 *  @file src/dense/nvme_variants_monoversioned.cpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The monoversioned NVMe unit: every variant's storage walk with the kits the build's
 *      floor and vouched capabilities compiled.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::nvme_k, cpu_class_t::monoversioned_k>;
