/**
 *  @file src/dense/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The serial unit: every variant at the portable floor, no instruction set beyond the
 *      baseline.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<vendored_residency_t::ram_k, cpu_class_t::serial_k>;
