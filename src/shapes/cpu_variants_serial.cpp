/**
 *  @file src/shapes/cpu_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The serial unit of the shapes variants, on the host and on storage: the floor every
 *      ladder ends on, at the baseline flags.
 */
#include "shapes_specializations.cuh"

template struct basic_class_tables<shapes_traits_t, vendored_residency_t::ram_k, cpu_class_t::serial_k>;
template struct basic_class_tables<shapes_traits_t, vendored_residency_t::nvme_k, cpu_class_t::serial_k>;
