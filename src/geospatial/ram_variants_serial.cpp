/**
 *  @file src/geospatial/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The serial unit of the geospatial arms: the floor every ladder ends on, at the baseline
 *      flags.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_terrain<cpu_class_t::serial_k>;
template struct geospatial_ram_variant<cpu_class_t::serial_k>;
template struct geospatial_nvme_variant<cpu_class_t::serial_k>;
