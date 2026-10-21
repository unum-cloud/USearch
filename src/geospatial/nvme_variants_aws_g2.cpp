/**
 *  @file src/geospatial/nvme_variants_aws_g2.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The aws_g2 NVMe unit of the geospatial variants: they inline the NEON kit, so the unit
 *      compiles with it.
 */
#include "geospatial_specializations.cuh"

template struct geospatial_nvme_variant<cpu_class_t::aws_g2_k>;
