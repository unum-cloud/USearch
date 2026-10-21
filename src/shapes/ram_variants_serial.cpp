/**
 *  @file src/shapes/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The serial unit of the shapes variants: the floor every ladder ends on, at the baseline
 *      flags.
 */
#include "shapes_specializations.cuh"

template struct shapes_ram_variant<nk::f32_t, cpu_class_t::serial_k>;
template struct shapes_ram_variant<nk::f16_t, cpu_class_t::serial_k>;
