/**
 *  @file src/geospatial/nvme_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The serial NVMe unit of the geospatial variants: the floor every ladder ends on, at the
 *      baseline flags.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_nvme_variant<cpu_class_t::serial_k>;
