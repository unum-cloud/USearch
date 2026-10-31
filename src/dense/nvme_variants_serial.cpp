/**
 *  @file src/dense/nvme_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The serial NVMe unit: every variant's storage walk at the portable floor.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::nvme_k, cpu_class_t::serial_k>;
