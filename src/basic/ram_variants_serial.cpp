/**
 *  @file src/basic/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The serial unit of the basic modalities: the floor every ladder ends on, at the baseline
 *      flags.
 */
#include "basic_specializations.cuh"

template struct sequence_arm<cpu_class_t::serial_k>;
template struct shapes_arm<nk::f32_t, cpu_class_t::serial_k>;
template struct shapes_arm<nk::f16_t, cpu_class_t::serial_k>;
template struct geospatial_arm<cpu_class_t::serial_k>;
