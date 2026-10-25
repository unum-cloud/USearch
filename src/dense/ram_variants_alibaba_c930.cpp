/**
 *  @file src/dense/ram_variants_alibaba_c930.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The alibaba_c930 unit: the serial kits under Zawrs waits & Zacas atomics.
 */

#include "dense_specializations.cuh"

template struct dense_class_tables<dense_home_t::ram_k, cpu_class_t::alibaba_c930_k>;
