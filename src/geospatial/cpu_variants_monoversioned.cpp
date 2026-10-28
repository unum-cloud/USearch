/**
 *  @file src/geospatial/cpu_variants_monoversioned.cpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief The monoversioned unit of the geospatial variants, on the host and on storage: the
 *      kits the build's floor and vouched capabilities compiled.
 */
#include "geospatial_specializations.cuh"

template struct basic_class_tables<geospatial_traits_t, dense_home_t::ram_k, cpu_class_t::monoversioned_k>;
template struct basic_class_tables<geospatial_traits_t, dense_home_t::nvme_k, cpu_class_t::monoversioned_k>;
